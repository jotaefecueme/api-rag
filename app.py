# api_query.py

import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model

load_dotenv()

app = FastAPI()

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

llm: ChatModel = init_chat_model(
    "meta-llama/llama-4-scout-17b-16e-instruct",
    model_provider="groq"
)

# Prompt del sistema
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

    docs: List[Document] = vector_store.similarity_search(request.question, k=request.k)

    if not docs:
        return {
            "answer": "No hay información disponible para responder a esta pregunta.",
            "time": round(time.time() - start_time, 3),
            "fragments": [],
        }

    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = SYSTEM_PROMPT.format(question=request.question, context=context)

    try:
        response = llm.invoke(prompt).content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al invocar el modelo: {e}")

    elapsed = round(time.time() - start_time, 3)

    return {
        "answer": response.strip(),
        "time": elapsed,
        "fragments": [
            {"id": i + 1, "content": doc.page_content}
            for i, doc in enumerate(docs)
        ],
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}
