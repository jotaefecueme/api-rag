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
from langchain_community.vectorstores import FAISS
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Conectando a la base de datos...")
    await db.connect()
    logger.info("Base de datos conectada.")

    logger.info("Cargando vectorstores FAISS en memoria (preload)...")
    await asyncio.to_thread(get_vector_store, "laserum")
    await asyncio.to_thread(get_vector_store, "salud")
    logger.info("Vectorstores FAISS cargados en memoria.")

    yield

    logger.info("Desconectando base de datos...")
    await db.disconnect()
    logger.info("Base de datos desconectada.")


app = FastAPI(lifespan=lifespan)

def get_embeddings() -> NomicEmbeddings:
    api_key = os.getenv("NOMIC_API_KEY")
    if not api_key:
        raise RuntimeError("Falta NOMIC_API_KEY en entorno o .env")
    logger.info("Inicializando embeddings Nomic...")
    emb = NomicEmbeddings(model="gte-multilingual-base")
    logger.info("Embeddings Nomic inicializados.")
    return emb

def get_vector_store(name: str) -> FAISS:
    embeddings = get_embeddings()
    path = f"./faiss_data/{name}"
    index_path = os.path.join(path, "index.faiss")
    metadata_path = os.path.join(path, "index.pkl")

    logger.debug(f"Cargando vector_store FAISS '{name}' desde {path} ...")

    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        raise RuntimeError(f"Vectorstore FAISS para '{name}' no encontrado en disco.")

    vs, _ = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    logger.debug(f"Vector store FAISS '{name}' cargado.")
    return vs



def get_llm():
    model = os.getenv("LLM_MODEL")
    provider = os.getenv("MODEL_PROVIDER")
    temperature = float(os.getenv("LLM_TEMPERATURE", 0.0))
    if not model or not provider:
        raise RuntimeError("Faltan variables LLM_MODEL o MODEL_PROVIDER en entorno")
    logger.info(f"Iniciando modelo LLM '{model}' con temperatura={temperature}...")
    llm = init_chat_model(model, model_provider=provider, temperature=temperature)
    logger.info("Modelo LLM inicializado.")
    return llm

SYSTEM_PROMPT_RAG_SALUD = (
    "Eres un asistente experto en responder preguntas usando solo la información proporcionada.\n"
    "Tu misión es maximizar la utilidad al usuario, sin desviarte jamás del contexto.\n\n"
    "OBJETIVOS:\n"
    "1. Responder con precisión y brevedad (≤ 40 palabras).\n"
    "2. Ayudar al usuario al máximo con la información disponible.\n\n"
    "RESTRICCIONES:\n"
    "- No menciones la fuente, el contexto ni uses expresiones tipo “según…”, “documentación”.\n"
    "- No especules, conjetures ni inventes datos.\n"
    "- No uses saludos, despedidas ni frases de cortesía.\n"
    "- Si no hay información suficiente, responde EXACTAMENTE:\n"
    "  “No hay información disponible para responder a esta pregunta.”\n\n"
    "FORMATO:\n"
    "- Texto plano, máximo 40 palabras.\n"
    "- Si aportas listas o viñetas, que sean muy breves (≤ 3 ítems).\n\n"
    "Pregunta: {question}\n"
    "Información: {context}\n"
    "Respuesta:"
)

SYSTEM_PROMPT_CONSTRUCCION = (
    "Eres un asistente que siempre responde con una sola frase breve indicando que la funcionalidad esta actualmente en construcción.\n"
    "- No expliques el motivo ni des detalles.\n"
    "- No añadas cortesía ni relleno.\n"
    "- La frase debe demostrar que has entendido el tema, pero indicar que esa funcionalidad esta actualmente en construcción.\n\n"
    "Pregunta: {question}\n"
    "Respuesta:"
)

SYSTEM_PROMPT_OUT_OF_SCOPE = (
    "Eres un asistente que siempre responde que cualquier pregunta está fuera de tu alcance.\n"
    "- Para todas las preguntas que recibas, responde con una sola frase corta diciendo que ese asunto no está cubierto.\n"
    "- Incluye una disculpa breve.\n"
    "- No añadas detalles innecesarios.\n"
    "- En la respuesta menciona de forma genérica y breve el tópico o área al que se refiere la pregunta, evitando repetir la pregunta literal para dar un feedback claro.\n\n"
    "- La estructura de la frase sería algo así: | <disculpa>, el <asunto genérico> está fuera de mi alcance. |\n\n"
    "Pregunta: {question}\n"
    "Respuesta:"
)

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
            vs = get_vector_store(vs_name)
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

        llm = get_llm()

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
