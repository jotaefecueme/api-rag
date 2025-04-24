# api_query.py

import os
import time
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

from langchain_cohere import CohereEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document

load_dotenv()

app = FastAPI()

embeddings = CohereEmbeddings(
    model="embed-multilingual-v3.0",
    cohere_api_key=os.getenv("COHERE_API_KEY"),
    user_agent="lekta-rag/0.1"
)
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

llm = init_chat_model(
    "meta-llama/llama-4-scout-17b-16e-instruct",
    model_provider="groq"
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
    question: str
    k: int = 5

@app.post("/query")
def query(request: QueryRequest):
    start_time = time.time()

    docs: List[Document] = vector_store.similarity_search(
        request.question, k=request.k
    )

    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = SYSTEM_PROMPT.format(question=request.question, context=context)

    response = llm.invoke(prompt).content

    elapsed = round(time.time() - start_time, 3)

    return {
        "answer": response,
        "time": elapsed,
        "fragments": [
            {"id": i + 1, "content": doc.page_content}
            for i, doc in enumerate(docs)
        ],
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}
