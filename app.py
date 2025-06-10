import os
import time
import logging
import asyncio
import psutil
from datetime import datetime
from zoneinfo import ZoneInfo
from contextlib import asynccontextmanager
from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv
from databases import Database

from langchain_nomic import NomicEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chat_models import init_chat_model

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=LOG_LEVEL,
)
logger = logging.getLogger("app")

DATABASE_URL = os.getenv("DATABASE_URL")
db = Database(DATABASE_URL)

embeddings = None
vectorstores = {}
llm = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embeddings, vectorstores, llm

    logger.info("Inicializando embeddings Nomic...")
    embeddings = NomicEmbeddings(model="gte-multilingual-base")
    logger.info("Embeddings Nomic inicializados.")

    for name in ("laserum", "salud"):
        path = f"./chroma_data/{name}"
        if not os.path.exists(path):
            logger.error(f"Vectorstore Chroma para '{name}' no encontrado en disco en {path}")
            raise RuntimeError(f"Vectorstore '{name}' no encontrado.")
        logger.info(f"Cargando vectorstore Chroma '{name}' desde {path} ...")
        vectorstores[name] = Chroma(persist_directory=path, embedding_function=embeddings)
        logger.info(f"Vectorstore '{name}' cargado.")

    model = os.getenv("LLM_MODEL")
    provider = os.getenv("MODEL_PROVIDER")
    temperature = float(os.getenv("LLM_TEMPERATURE", 0.0))
    if not model or not provider:
        logger.error("Variables LLM_MODEL o MODEL_PROVIDER no definidas en entorno")
        raise RuntimeError("Faltan variables LLM_MODEL o MODEL_PROVIDER en entorno")
    logger.info(f"Iniciando modelo LLM '{model}' con temperatura={temperature}...")
    llm = init_chat_model(model, model_provider=provider, temperature=temperature)
    logger.info("Modelo LLM inicializado.")

    logger.info("Conectando a la base de datos...")
    await db.connect()
    logger.info("Base de datos conectada.")

    yield

    logger.info("Desconectando base de datos...")
    await db.disconnect()
    logger.info("Base de datos desconectada.")


app = FastAPI(lifespan=lifespan)

class QueryRequest(BaseModel):
    id: str = Field(..., description="ID para identificar la consulta")
    question: str = Field(..., min_length=3, max_length=300, description="Texto de la pregunta")
    k: int = Field(5, ge=1, le=10, description="Número de fragmentos a recuperar")

    @field_validator("question")
    @classmethod
    def validate_question(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("La pregunta no puede estar vacía o solo espacios")
        return v


def truncate_context(context: str, max_chars: int = 100000) -> str:
    if len(context) <= max_chars:
        return context
    trunc = context[:max_chars]
    last_space = trunc.rfind(" ")
    if last_space > 0:
        trunc = trunc[:last_space]
    return trunc + "..."

async def log_query_to_db(
    request: Request,
    input_id: str,
    input_text: str,
    output_text: str,
    infer_time: float,
    total_time: float,
):
    now = datetime.now(ZoneInfo("Europe/Madrid"))
    query = """
        INSERT INTO "api-rag" (
            ip, date, time, input_text, input_id, output_generation, infer_time, total_time, model, provider, temperature
        ) VALUES (
            :ip, :date, :time, :input_text, :input_id, :output_generation, :infer_time, :total_time, :model, :provider, :temperature
        )
    """
    values = {
        "ip": request.client.host,
        "date": now.date().isoformat(),
        "time": now.time().isoformat(),
        "input_text": input_text,
        "input_id": input_id,
        "output_generation": output_text,
        "infer_time": f"{infer_time:.3f}",
        "total_time": f"{total_time:.3f}",
        "model": os.getenv("LLM_MODEL"),
        "provider": os.getenv("MODEL_PROVIDER"),
        "temperature": f"{float(os.getenv('LLM_TEMPERATURE', 0.0)):.1f}",
    }
    try:
        await db.execute(query=query, values=values)
        logger.debug("Consulta registrada en la base de datos correctamente.")
    except Exception as e:
        logger.error(f"Error guardando log en DB: {e}")

@app.post("/query")
async def query(request: QueryRequest, raw_request: Request):
    valid_ids = {"rag_salud", "rag_laserum", "construccion", "out_of_scope"}
    if request.id not in valid_ids:
        logger.warning(f"ID no permitido recibido: {request.id}")
        raise HTTPException(status_code=400, detail="ID no permitido")

    start_time = time.time()
    logger.info(f"Consulta recibida: id={request.id} pregunta='{request.question}' (k={request.k})")

    try:
        docs = []
        prompt = ""

        if request.id.startswith("rag_"):
            vs_name = request.id.replace("rag_", "")
            if vs_name not in vectorstores:
                logger.error(f"Vectorstore '{vs_name}' no cargado.")
                raise HTTPException(status_code=500, detail=f"Vectorstore '{vs_name}' no cargado")
            vs = vectorstores[vs_name]
            docs = await asyncio.to_thread(vs.similarity_search, request.question, request.k)
            logger.info(f"Vector search para '{request.id}' completado en {time.time() - start_time:.3f}s, {len(docs)} docs encontrados.")
            if not docs:
                elapsed = time.time() - start_time
                logger.info(f"Sin resultados para {request.id}. Tiempo total: {elapsed:.3f}s")
                return {
                    "id": request.id,
                    "answer": "No hay información disponible para responder a esta pregunta.",
                    "time": round(elapsed, 3),
                    "fragments": [],
                }
            context = truncate_context("\n\n".join(doc.page_content for doc in docs))
            prompt = SYSTEM_PROMPT_RAG_SALUD.format(question=request.question, context=context)

        elif request.id == "construccion":
            prompt = SYSTEM_PROMPT_CONSTRUCCION.format(question=request.question)
            logger.info("Respuesta de construcción solicitada.")

        elif request.id == "out_of_scope":
            prompt = SYSTEM_PROMPT_OUT_OF_SCOPE.format(question=request.question)
            logger.info("Respuesta de fuera de alcance solicitada.")

        if llm is None:
            logger.error("LLM no inicializado.")
            raise HTTPException(status_code=500, detail="Modelo LLM no inicializado")

        async def invoke_llm_with_timeout(prompt: str, timeout: float = 10.0):
            return await asyncio.wait_for(asyncio.to_thread(llm.invoke, prompt), timeout=timeout)

        inference_start = time.time()
        try:
            llm_response = await invoke_llm_with_timeout(prompt)
        except asyncio.TimeoutError:
            logger.error("Timeout en llamada al modelo LLM")
            raise HTTPException(status_code=504, detail="Tiempo de respuesta del modelo agotado")
        inference_duration = time.time() - inference_start

        output_text = llm_response.content if hasattr(llm_response, "content") else str(llm_response)
        total_duration = time.time() - start_time

        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024**2

        logger.info(
            f"LLM completado en {inference_duration:.3f}s | Tiempo total: {total_duration:.3f}s | RAM uso: {memory_usage:.1f} MiB"
        )

        asyncio.create_task(
            log_query_to_db(
                raw_request,
                request.id,
                request.question,
                output_text,
                inference_duration,
                total_duration,
            )
        )

        return {
            "id": request.id,
            "answer": output_text,
            "time": round(total_duration, 3),
            "fragments": [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs] if docs else [],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en endpoint /query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error interno en el servidor")


@app.get("/health")
async def health():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024**2
    uptime = time.time() - process.create_time()
    return {
        "status": "ok",
        "model": os.getenv("LLM_MODEL"),
        "provider": os.getenv("MODEL_PROVIDER"),
        "temperature": float(os.getenv("LLM_TEMPERATURE", 0.0)),
        "ram_mib": round(mem, 1),
        "uptime_s": round(uptime, 1),
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level=LOG_LEVEL.lower())
