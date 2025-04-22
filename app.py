# api_query.py

from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
from langchain_cohere import CohereEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
import time
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Cargar modelo de embeddings
embeddings = CohereEmbeddings(
    model="embed-multilingual-v3.0",
    cohere_api_key=os.getenv("COHERE_API_KEY"),
    user_agent="lekta-rag/0.1"
)

# Cargar vectorstore desde disco
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

# Cargar LLM
llm = init_chat_model(
    "meta-llama/llama-4-scout-17b-16e-instruct",
    model_provider="groq"
)

# Prompt base
SYSTEM_PROMPT = (
    "Eres un asistente especializado en responder preguntas utilizando únicamente información proporcionada en la documentación recuperada. "
    "Sé claro, preciso y directo.\n"
    "- No inventes ni especules.\n"
    "- Resume sin perder precisión.\n"
    "- Máximo 3 frases.\n\n"
    "Pregunta: {question}\nDocumentación: {context}\nRespuesta:"
)

# Request model
class QueryRequest(BaseModel):
    question: str
    k: int = 5

@app.post("/query")
def query(request: QueryRequest):
    start_time = time.time()

    # Buscar documentos relevantes
    docs: List[Document] = vector_store.similarity_search(request.question, k=request.k)

    # Crear el contexto para el prompt
    context = "\n\n".join(d.page_content for d in docs)
    prompt = SYSTEM_PROMPT.format(question=request.question, context=context)

    # Obtener respuesta del LLM
    response = llm.invoke(prompt).content

    elapsed = round(time.time() - start_time, 3)
    return {"answer": response, "time": elapsed}
