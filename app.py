import os
import time
import logging
import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List
from dotenv import load_dotenv

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")
app = FastAPI()

embeddings = HuggingFaceEmbeddings(
    model_name=os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base"),
    model_kwargs={"device": os.getenv("DEVICE", "cpu")},
    encode_kwargs={"normalize_embeddings": False}
)
vector_store = Chroma(
    collection_name=os.getenv("CHROMA_COLLECTION", "example_collection"),
    embedding_function=embeddings,
    persist_directory=os.getenv("CHROMA_DIR", "./chroma_langchain_db"),
)

tem = float(os.getenv("LLM_TEMPERATURE", "0.0"))
llm = init_chat_model(
    os.getenv("LLM_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
    model_provider=os.getenv("MODEL_PROVIDER", "groq"),
    temperature=tem
)

SYSTEM_PROMPT = (
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

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=300, description="Texto de la pregunta")
    k: int = Field(5, ge=1, le=10, description="Número de fragmentos a recuperar")

    @validator("question")
    def validate_question(cls, v):
        text = v.strip()
        if not text:
            raise ValueError("La pregunta no puede estar vacía o solo espacios")
        return text


def truncate_context(context: str, max_chars: int = 3000) -> str:
    """Limita el contexto a un tamaño máximo para el prompt."""
    return context if len(context) <= max_chars else context[:max_chars] + "..."

@app.post("/query")
async def query(request: QueryRequest):
    start_time = time.time()
    logger.info(f"Consulta recibida: '{request.question}' (k={request.k})")

    try:
        docs = await asyncio.to_thread(vector_store.similarity_search, request.question, request.k)
    except Exception:
        logger.exception("Error en búsqueda vectorial")
        raise HTTPException(status_code=500, detail="Error interno en vector search")

    if not docs:
        elapsed = time.time() - start_time
        logger.info(f"Sin resultados. Tiempo de respuesta: {elapsed:.3f}s")
        return {"answer": "No hay información disponible para responder a esta pregunta.",
                "time": round(elapsed, 3),
                "fragments": []}

    context = "\n\n".join(doc.page_content for doc in docs)
    context = truncate_context(context)
    prompt = SYSTEM_PROMPT.format(question=request.question, context=context)

    try:
        llm_response = await asyncio.to_thread(lambda: llm.invoke(prompt))
        answer = llm_response.content.strip()
    except Exception:
        logger.exception("Error al invocar el LLM")
        raise HTTPException(status_code=500, detail="Error interno en generación de respuesta")

    elapsed = time.time() - start_time
    logger.info(f"Respuesta generada en {elapsed:.3f}s")

    return {
        "answer": answer,
        "time": round(elapsed, 3),
        "fragments": [{"id": i+1, "content": doc.page_content} for i, doc in enumerate(docs)]
    }

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
