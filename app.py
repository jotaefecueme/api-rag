import os
import time
import logging
import asyncio
import uuid
import psutil
from datetime import datetime
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv
from databases import Database
from langchain_nomic import NomicEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.chat_models import init_chat_model
from contextlib import asynccontextmanager
from zoneinfo import ZoneInfo

load_dotenv()

# Setup Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    level=LOG_LEVEL
)
logger = logging.getLogger("app")

# Database
DATABASE_URL = os.getenv("DATABASE_URL")
db = Database(DATABASE_URL)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Conectando a la base de datos...")
    await db.connect()
    logger.info("DB conectada.")
    yield
    logger.info("Desconectando de la base de datos...")
    await db.disconnect()
    logger.info("DB desconectada.")

# App Initialization con lifespan
app = FastAPI(lifespan=lifespan)

# Embeddings Setup
logger.info("Inicializando embeddings Nomic...")
if not os.getenv("NOMIC_API_KEY"):
    raise ValueError("Falta NOMIC_API_KEY en el entorno o en .env")
embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")
logger.info("Embeddings Nomic inicializados.")

# Vector Stores (FAISS)
logger.info("Cargando vector_store_salud...")
vector_store_salud = FAISS.load_local("./faiss_data/salud", embeddings, allow_dangerous_deserialization=True)
logger.info("vector_store_salud cargado.")

logger.info("Cargando vector_store_laserum...")
vector_store_laserum = FAISS.load_local("./faiss_data/laserum", embeddings, allow_dangerous_deserialization=True)
logger.info("vector_store_laserum cargado.")

# LLM Initialization
temperature = float(os.getenv("LLM_TEMPERATURE"))
logger.info(f"Iniciando modelo LLM '{os.getenv('LLM_MODEL')}' con temperatura={temperature}...")
llm = init_chat_model(
    os.getenv("LLM_MODEL"),
    model_provider=os.getenv("MODEL_PROVIDER"),
    temperature=temperature
)
logger.info("Modelo LLM inicializado.")

# System Prompts
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
    "- La estructura de la frase sería algo así: | <disculpa>, el <asunto genérico> está fuera de mi alcance. | No tienes que seguir este formato estrictamente, pero esa es la idea.\n\n"
    "Pregunta: {question}\n"
    "Respuesta:"
)

# Models
class QueryRequest(BaseModel):
    id: str = Field(..., description="ID para identificar la consulta")
    question: str = Field(..., min_length=3, max_length=300, description="Texto de la pregunta")
    k: int = Field(5, ge=1, le=10, description="Número de fragmentos a recuperar")

    @field_validator("question")
    @classmethod
    def validate_question(cls, v):
        text = v.strip()
        if not text:
            raise ValueError("La pregunta no puede estar vacía o solo espacios")
        return text

# Utils
def truncate_context(context: str, max_chars: int = 3000) -> str:
    return context if len(context) <= max_chars else context[:max_chars] + "..."

async def log_query_to_db(request: Request, input_id, input_text, output_text, infer_time, total_time):
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
        "infer_time": str(round(infer_time, 3)),
        "total_time": str(round(total_time, 3)),
        "model": os.getenv("LLM_MODEL"),
        "provider": os.getenv("MODEL_PROVIDER"),
        "temperature": str(float(os.getenv("LLM_TEMPERATURE", 0.0)))
    }
    try:
        await db.execute(query=query, values=values)
        logger.debug("Consulta registrada en la base de datos correctamente.")
    except Exception as e:
        logger.error(f"Error guardando log en DB: {e}")

# Routes
@app.post("/query")
async def query(request: QueryRequest, raw_request: Request):
    if request.id not in ["rag_salud", "rag_laserum", "construccion", "out_of_scope"]:
        logger.warning(f"ID no permitido recibido: {request.id}")
        raise HTTPException(status_code=400, detail="ID no permitido")

    start_time = time.time()
    logger.info(f"Consulta recibida: id={request.id} pregunta='{request.question}' (k={request.k})")

    try:
        vector_search_start = time.time()
        if request.id == "rag_salud":
            docs = await asyncio.to_thread(vector_store_salud.similarity_search, request.question, request.k)
            logger.info(f"Vector search para 'rag_salud' completado en {time.time() - vector_search_start:.3f}s, {len(docs)} docs encontrados.")
            prompt = SYSTEM_PROMPT_RAG_SALUD.format(
                question=request.question,
                context=truncate_context("\n\n".join(doc.page_content for doc in docs))
            )
        elif request.id == "rag_laserum":
            docs = await asyncio.to_thread(vector_store_laserum.similarity_search, request.question, request.k)
            logger.info(f"Vector search para 'rag_laserum' completado en {time.time() - vector_search_start:.3f}s, {len(docs)} docs encontrados.")
            prompt = SYSTEM_PROMPT_RAG_SALUD.format(
                question=request.question,
                context=truncate_context("\n\n".join(doc.page_content for doc in docs))
            )
        elif request.id == "construccion":
            docs = []
            prompt = SYSTEM_PROMPT_CONSTRUCCION.format(question=request.question)
            logger.info("Respuesta de construcción solicitada.")
        elif request.id == "out_of_scope":
            docs = []
            prompt = SYSTEM_PROMPT_OUT_OF_SCOPE.format(question=request.question)
            logger.info("Respuesta de fuera de alcance solicitada.")

        if request.id.startswith("rag_") and not docs:
            elapsed = time.time() - start_time
            logger.info(f"Sin resultados para {request.id}. Tiempo total: {elapsed:.3f}s")
            return {
                "id": request.id,
                "answer": "No hay información disponible para responder a esta pregunta.",
                "time": round(elapsed, 3),
                "fragments": []
            }

        inference_start = time.time()
        llm_response = await asyncio.to_thread(llm.invoke, prompt)
        inference_duration = time.time() - inference_start

        total_duration = time.time() - start_time
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024**2

        logger.info(f"LLM completado en {inference_duration:.3f}s | Tiempo total desde consulta: {total_duration:.3f}s | RAM usada: {memory_usage:.2f} MB")

        fragments = [{"id": i + 1, "content": doc.page_content} for i, doc in enumerate(docs)] if docs else []

        await log_query_to_db(raw_request, request.id, request.question, llm_response.content.strip(), inference_duration, total_duration)

        return {
            "id": request.id,
            "answer": llm_response.content.strip(),
            "time": round(total_duration, 3),
            "fragments": fragments
        }

    except Exception as e:
        logger.exception(f"Error en procesamiento de /query: {e}")
        raise HTTPException(status_code=500, detail="Error interno en procesamiento de la consulta")

@app.get("/health")
async def health_check(_: Request):
    start = time.time()
    response = {"status": "ok"}
    logger.debug(f"/health OK en {time.time() - start:.3f}s")
    return response
