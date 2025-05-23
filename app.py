import os
import time
import logging
import asyncio
import uvicorn
import psutil
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from pydantic import field_validator

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


def truncate_context(context: str, max_chars: int = 3000) -> str:
    return context if len(context) <= max_chars else context[:max_chars] + "..."


@app.post("/query")
async def query(request: QueryRequest):
    if request.id not in ["rag_salud", "construccion", "out_of_scope"]:
        raise HTTPException(status_code=400, detail="ID no permitido")

    request_start = time.time()
    logger.info(f"Consulta recibida: id={request.id} pregunta='{request.question}' (k={request.k})")

    if request.id == "rag_salud":
        try:
            docs = await asyncio.to_thread(vector_store.similarity_search, request.question, request.k)
        except Exception:
            logger.exception("Error en búsqueda vectorial")
            raise HTTPException(status_code=500, detail="Error interno en vector search")

        if not docs:
            elapsed = time.time() - request_start
            logger.info(f"Sin resultados. Tiempo de respuesta: {elapsed:.3f}s")
            return {
                "id": request.id,
                "answer": "No hay información disponible para responder a esta pregunta.",
                "time": round(elapsed, 3),
                "fragments": []
            }

        context = "\n\n".join(doc.page_content for doc in docs)
        context = truncate_context(context)
        prompt = SYSTEM_PROMPT_RAG_SALUD.format(question=request.question, context=context)

    elif request.id == "construccion":
        prompt = SYSTEM_PROMPT_CONSTRUCCION.format(question=request.question)

    elif request.id == "out_of_scope":
        prompt = SYSTEM_PROMPT_OUT_OF_SCOPE.format(question=request.question)

    try:
        inference_start = time.time()
        llm_response = await asyncio.to_thread(lambda: llm.invoke(prompt))
        inference_time = time.time() - inference_start
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024**2
        logger.info(f"Inference time: {inference_time:.3f}s | Memory usage: {memory_usage:.2f} MB")
        answer = llm_response.content.strip()
    except Exception:
        logger.exception("Error al invocar el LLM")
        raise HTTPException(status_code=500, detail="Error interno en generación de respuesta")

    total_time = time.time() - request_start
    logger.info(f"POST /query - {total_time:.3f}s")

    fragments = []
    if request.id == "rag_salud":
        fragments = [{"id": i + 1, "content": doc.page_content} for i, doc in enumerate(docs)]

    return {
        "id": request.id,
        "answer": answer,
        "time": round(total_time, 3),
        "fragments": fragments
    }

@app.get("/health")
async def health_check(request: Request):
    logger.info("GET /health - 0.001s")
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
