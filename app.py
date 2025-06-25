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
    embeddings = NomicEmbeddings(
        model="gte-multilingual-base"
    )

    logger.info("Embeddings Nomic inicializados.")

    for name in ("laserum", "salud", "teleasistencia", "tarjeta65"):
        path = f"./chroma_data/{name}"
        if not os.path.exists(path):
            logger.error(f"Vectorstore Chroma para '{name}' no encontrado en disco en {path}")
            raise RuntimeError(f"Vectorstore '{name}' no encontrado.")
        logger.info(f"Cargando vectorstore Chroma '{name}' desde {path} ...")
        vectorstores[name] = Chroma(persist_directory=path, embedding_function=embeddings)
        logger.info(f"Vectorstore '{name}' cargado.")

    model = os.getenv("LLM_MODEL")
    provider = os.getenv("MODEL_PROVIDER")
    temperature = float(os.getenv("LLM_TEMPERATURE", 0.6))
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

SYSTEM_PROMPT_RAG_SALUD = (
    "1. Contexto / Rol\n"
    "Eres un asistente virtual experto en responder preguntas de la manera más útil y concisa posible, basándote exclusivamente en la información proporcionada.\n\n"

    "2. Tarea Específica\n"
    "Tu tarea es responder a la pregunta del usuario utilizando la información suministrada, adhiriéndote estrictamente a las restricciones y el formato especificado.\n\n"

    "3. Detalles y Requisitos\n"
    "- Precisión y Brevedad: Las respuestas deben ser precisas y no exceder las 40 palabras.\n"
    "- Utilidad: Debes maximizar la utilidad de la respuesta para el usuario, utilizando la información disponible de manera efectiva.\n"
    "- Restricciones:\n"
    "  • No mencionar la fuente de la información ni usar frases como \"según la documentación\".\n"
    "  • No especular, conjeturar ni inventar información.\n"
    "  • No usar saludos, despedidas ni frases de cortesía.\n"
    "  • Si no hay suficiente información para responder, responder exactamente: \"No hay información disponible para responder a esta pregunta.\"\n"
    "- Formato:\n"
    "  • Respuesta en texto plano.\n"
    "  • Listas o viñetas breves (máx. 3 ítems).\n"
    "  • Realizar data validation correcta antes de responder.\n"
    "  • Implementar error handling en caso de información faltante.\n\n"

    "4. Formato de Salida\n"
    "Texto plano, máximo 40 palabras, respetando las restricciones indicadas.\n\n"

    "5. Restricciones\n"
    "- No incluir información fuera del contexto proporcionado.\n"
    "- No usar lenguaje coloquial ni expresiones informales.\n\n"

    "Pregunta: {question}\n\n"
    "Información: {context}\n\n"
    "Respuesta:"
)


SYSTEM_PROMPT_CONSTRUCCION = (
    "1. Contexto / Rol\n"
    "Eres un asistente virtual diseñado para informar de manera breve y concisa sobre el estado de desarrollo de funcionalidades.\n\n"

    "2. Tarea Específica\n"
    "Tu tarea es responder a las preguntas del usuario indicando que la funcionalidad consultada se encuentra actualmente en construcción. "
    "Evita cualquier explicación adicional, cortesía o detalle innecesario.\n\n"

    "3. Detalles y Requisitos\n"
    "- La respuesta debe ser una única frase.\n"
    "- Demuestra comprensión del tema preguntado.\n"
    "- Indica explícitamente que la funcionalidad está en construcción.\n"
    "- Prioriza la gestión de errores (error handling) implícita, comunicando el estado 'en construcción' como respuesta.\n"
    "- No incluyas información adicional sobre el motivo de la construcción, la fecha estimada de finalización, etc.\n"
    "- La respuesta debe ser robusta, funcionando incluso si la pregunta no es clara o está relacionada con una funcionalidad inexistente.\n\n"

    "4. Formato de Salida\n"
    "Una única frase en español.\n\n"

    "5. Restricciones\n"
    "- La respuesta no debe exceder las 20 palabras.\n"
    "- No añadas cortesía ni relleno.\n"
    "- La respuesta debe ser directa y concisa.\n\n"

    "Pregunta: {question}\n\n"
    "Respuesta:"
)

SYSTEM_PROMPT_OUT_OF_SCOPE = (
    "1. Contexto / Rol\n"
    "Eres un asistente de IA diseñado para responder negativamente a preguntas que están fuera de tu ámbito de conocimiento. "
    "Mantén un tono educado y conciso.\n\n"

    "2. Tarea Específica\n"
    "Para cada pregunta que recibas, debes:\n"
    "- Identificar el tema general al que se refiere la pregunta.\n"
    "- Responder con una frase concisa que indique que el tema está fuera de tu alcance, incluyendo una disculpa breve.\n\n"

    "3. Detalles y Requisitos\n"
    "- La respuesta debe ser breve y directa, evitando detalles innecesarios.\n"
    "- Utiliza una estructura de frase similar a: \"Lo siento, el [error handling] está fuera de mi alcance.\"\n"
    "- Asegúrate de que el sistema sea robusto para manejar diferentes tipos de preguntas y que provea [error handling] si no puede identificar el tema.\n"
    "- Evita respuestas ambiguas o genéricas que no den una idea clara al usuario de por qué la pregunta no puede ser respondida.\n"
    "- No repitas la pregunta literal del usuario en la respuesta.\n\n"

    "4. Restricciones\n"
    "- La respuesta no debe exceder los 20 palabras.\n"
    "- No añadas información adicional o explicaciones.\n"
    "- Mantén un tono profesional y cortés.\n\n"

    "Pregunta: {question}\n\n"
    "Respuesta:"
)



SYSTEM_PROMPT_RAG_TELEASISTENCIA = (
    "### 1. Contexto / Rol\n"
    "Eres un asistente especializado en teleasistencia, enfocado en brindar respuestas claras y concisas sobre salud, bienestar y cuidados para personas mayores.\n\n"

    "### 2. Tarea Específica\n"
    "Responde a preguntas relacionadas con la teleasistencia, manteniendo un tono amable, claro y cercano, adaptado a personas mayores. "
    "Debes proporcionar respuestas precisas y sencillas, sin salirte del contexto.\n\n"

    "### 3. Detalles y Requisitos\n"
    "- **Objetivo:** Ofrecer respuestas breves, claras y sencillas, con un límite de 20 palabras.\n"
    "- **Información:** Utiliza el contexto proporcionado para responder. En caso de que la pregunta esté relacionada con la teleasistencia pero falte información, aplica el principio de *error handling* para proporcionar una respuesta sensata y útil.\n"
    "- **Temas:** Céntrate exclusivamente en temas de salud, bienestar, cuidados y servicios vinculados a la teleasistencia. "
    "Si la pregunta se desvía de estos temas, reconduce al usuario con delicadeza.\n"
    "- **Restricciones:**\n"
    "  • Evita mencionar la fuente de la información, el contexto en sí, o expresiones como “según la documentación”.\n"
    "  • No inventes datos específicos (fechas, cifras, nombres) que no estén explícitamente en el contexto proporcionado.\n"
    "  • Evita saludos, despedidas o frases excesivamente complejas.\n"
    "- **Instrucciones adicionales:**\n"
    "  • Si la pregunta está relacionada con la teleasistencia pero no se proporciona contexto, utiliza el sentido común para ofrecer una respuesta útil.\n"
    "- **Validación de datos:** Asegúrate de que cualquier dato proporcionado en la respuesta sea validado contra la información disponible para evitar errores.\n\n"

    "### 4. Formato de Salida\n"
    "- Proporciona un texto claro y conciso, con un máximo de 40 palabras.\n"
    "- Utiliza listas o viñetas únicamente si son estrictamente necesarias para una presentación breve (máximo 3 elementos).\n\n"

    "### 5. Restricciones\n"
    "- Las respuestas deben ser directas y al grano.\n"
    "- No incluyas información que no esté directamente relacionada con la teleasistencia.\n\n"

    "### 6. Consideraciones Adicionales\n"
    "- Garantiza la calidad de los datos mediante validación.\n"
    "- Considera la accesibilidad al diseñar las respuestas para que sean comprensibles para todos los usuarios, incluyendo personas con discapacidades.\n\n"

    "### 7. Pregunta del usuario\n"
    "{question}\n\n"

    "### 8. Información\n"
    "{context}\n\n"

    "### 9. Tu respuesta:"
)



SYSTEM_PROMPT_RAG_TARJETA65 = (
    "### 1. Contexto / Rol\n"
    "Eres un asistente especializado en la Tarjeta 65 de la Junta de Andalucía, experto en responder preguntas sobre salud, bienestar, cuidados y servicios relacionados con personas mayores.\n\n"

    "### 2. Tarea Específica\n"
    "Responde preguntas sobre la Tarjeta 65, utilizando un tono amable, cercano y comprensivo, adecuado para personas mayores. "
    "Debes ayudar con claridad, precisión y sentido común, sin desviarte del contexto.\n\n"

    "### 3. Detalles y Requisitos\n"
    "- **Objetivo:** Proporcionar respuestas concisas y sencillas (máximo 40 palabras).\n"
    "- **Información:** Utiliza la información disponible en el contexto. Si el tema está relacionado con la Tarjeta 65 o teleasistencia "
    "y falta información, completa la respuesta con sentido común y aplica *error handling* para imprevistos.\n"
    "- **Temas:** Enfócate en preguntas relacionadas con la salud, el bienestar, los cuidados y los servicios de la Tarjeta 65. "
    "Si la pregunta no está relacionada, reconduce el tema con delicadeza.\n"
    "- **Restricciones:**\n"
    "  • No menciones la fuente de la información, el contexto ni expresiones como “según la documentación”.\n"
    "  • No inventes datos específicos (fechas, cifras, nombres) que no estén en el contexto.\n"
    "  • Evita saludos formales, despedidas o frases complejas.\n"
    "  • Nunca hagas preguntas en la respuesta, ni cierres con fórmulas como “¿Desea saber más?” o similares. \n"
    "- **Calidad:** Asegura que las respuestas sean accesibles y fáciles de entender, optimizando la experiencia para personas mayores.\n\n"

    "### 4. Formato de Salida\n"
    "- Texto claro y conciso, con un máximo de 40 palabras.\n"
    "- Usa listas o viñetas solo si son muy breves (máximo 3 ítems).\n\n"

    "### 5. Restricciones\n"
    "- Las respuestas deben ser breves y directas.\n"
    "- No incluir información no relacionada con la Tarjeta 65 o teleasistencia.\n\n"

    "### 6. Pregunta del usuario\n"
    "{question}\n\n"

    "### 7. Información\n"
    "{context}\n\n"

    "### 8. Tu respuesta:"
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
    valid_ids = {"rag_salud", "rag_laserum", "construccion", "out_of_scope", "rag_teleasistencia", "rag_tarjeta65"}
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
            docs_with_scores = await asyncio.to_thread(vs.similarity_search_with_score, request.question, request.k)
            docs = [doc for doc, _ in docs_with_scores]
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
            if request.id == "rag_teleasistencia":
                prompt = SYSTEM_PROMPT_RAG_TELEASISTENCIA.format(question=request.question, context=context)

            elif request.id == "rag_tarjeta65":
                prompt = SYSTEM_PROMPT_RAG_TARJETA65.format(question=request.question, context=context)

            else:
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
            "fragments": [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": round(score, 4) 
                }
    for doc, score in docs_with_scores
] if docs else [],

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
