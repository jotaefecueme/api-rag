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
    "Responde a las preguntas usando únicamente la información proporcionada.\n"
    "Tu respuesta debe ser clara, precisa y en máximo 30 palabras.\n\n"
    "Instrucciones:\n"
    "- Intenta siempre ayudar de la mejor forma posible con la información disponible.\n"
    "- No menciones la fuente ni frases como 'según...', 'documentación', etc.\n"
    "- No especules ni inventes información.\n"
    "- Si la pregunta está completamente fuera de lugar o contexto según tu información, responde exactamente: 'No hay información disponible para responder a esta pregunta.'\n\n"
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
