@app.post("/query")
async def query(request: QueryRequest, raw_request: Request):
    if request.id not in ["rag_salud", "rag_laserum", "construccion", "out_of_scope"]:
        raise HTTPException(status_code=400, detail="ID no permitido")

    total_start = time.time()
    logger.info(f"Consulta recibida: id={request.id} pregunta='{request.question}' (k={request.k})")

    try:
        t0 = time.time()
        if request.id == "rag_salud":
            docs = await asyncio.to_thread(vector_store_salud.similarity_search, request.question, request.k)
            context = truncate_context("\n\n".join(doc.page_content for doc in docs))
            prompt = SYSTEM_PROMPT_RAG_SALUD.format(question=request.question, context=context)
        elif request.id == "rag_laserum":
            docs = await asyncio.to_thread(vector_store_laserum.similarity_search, request.question, request.k)
            context = truncate_context("\n\n".join(doc.page_content for doc in docs))
            prompt = SYSTEM_PROMPT_RAG_SALUD.format(question=request.question, context=context)
        elif request.id == "construccion":
            docs = []
            prompt = SYSTEM_PROMPT_CONSTRUCCION.format(question=request.question)
        elif request.id == "out_of_scope":
            docs = []
            prompt = SYSTEM_PROMPT_OUT_OF_SCOPE.format(question=request.question)
        t1 = time.time()
        logger.info(f"Tiempo en búsqueda + construcción prompt: {t1 - t0:.3f}s")

        if request.id.startswith("rag_") and not docs:
            total_time = time.time() - total_start
            logger.info(f"Sin resultados para {request.id}. Tiempo total: {total_time:.3f}s")
            return {
                "id": request.id,
                "answer": "No hay información disponible para responder a esta pregunta.",
                "time": round(total_time, 3),
                "fragments": []
            }

        t2 = time.time()
        llm_response = await asyncio.to_thread(llm.invoke, prompt)
        t3 = time.time()
        inference_duration = t3 - t2
        total_duration = t3 - total_start

        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024**2
        logger.info(f"Inferencia completada en {inference_duration:.3f}s | RAM usada: {memory_usage:.2f} MB")

        fragments = [{"id": i + 1, "content": doc.page_content} for i, doc in enumerate(docs)] if docs else []

        t4 = time.time()
        await log_query_to_db(
            raw_request,
            request.id,
            request.question,
            llm_response.content.strip(),
            inference_duration,
            total_duration
        )
        t5 = time.time()
        logger.info(f"Tiempo en log DB: {t5 - t4:.3f}s | Tiempo total completo: {time.time() - total_start:.3f}s")

        return {
            "id": request.id,
            "answer": llm_response.content.strip(),
            "time": round(total_duration, 3),
            "fragments": fragments
        }

    except Exception as e:
        logger.exception(f"Error en procesamiento de /query: {e}")
        raise HTTPException(status_code=500, detail="Error interno en procesamiento de la consulta")
