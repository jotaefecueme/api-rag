import os
import time
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List
from dotenv import load_dotenv
import uvicorn
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

app = FastAPI()

embeddings = HuggingFaceEmbeddings(
    model_name=os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large"),
    model_kwargs={"device": os.getenv("DEVICE", "cpu")},
    encode_kwargs={"normalize_embeddings": False}
)

vector_store = Chroma(
    collection_name=os.getenv("CHROMA_COLLECTION", "example_collection"),
    embedding_function=embeddings,
    persist_directory=os.getenv("CHROMA_DIR", "./chroma_langchain_db"),
)

llm = init_chat_model(
    os.getenv("LLM_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
    model_provider=os.getenv("MODEL_PROVIDER", "groq"),
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
        docs: List[Document] = vector_store.similarity_search(request.question, k=request.k)
    except Exception:
        logger.exception("Error en búsqueda vectorial")
        raise HTTPException(status_code=500, detail="Error interno en vector search")

    if not docs:
        elapsed = round(time.time() - start_time, 3)
        logger.info(f"Sin resultados. Tiempo de respuesta: {elapsed}s")
        return {"answer": "No hay información disponible para responder a esta pregunta.", "time": elapsed, "fragments": []}

    context = "\n\n".join(doc.page_content for doc in docs)
    context = truncate_context(context)
    prompt = SYSTEM_PROMPT.format(question=request.question, context=context)

    try:
        response = llm.invoke(prompt).content.strip()
    except Exception:
        logger.exception("Error al invocar el LLM")
        raise HTTPException(status_code=500, detail="Error interno en generación de respuesta")

    elapsed = round(time.time() - start_time, 3)
    logger.info(f"Respuesta generada en {elapsed}s")

    return {
        "answer": response,
        "time": elapsed,
        "fragments": [{"id": i+1, "content": doc.page_content} for i, doc in enumerate(docs)]
    }

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
