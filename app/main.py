import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from sqlalchemy.orm import Session

from app.cache import get_cached_value, make_cache_key, set_cached_value
from app.config import settings
from app.db import ConversationLog, SessionLocal, init_db
from app.rag import RAGService
from app.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    RetrieveDebugResponse,
)


rag_service: RAGService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_service
    init_db()
    rag_service = RAGService()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        environment=settings.environment,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    global rag_service

    if rag_service is None:
        raise HTTPException(status_code=500, detail="RAG service is not initialized")

    cache_key = make_cache_key(payload.session_id, payload.question)
    cached = get_cached_value(cache_key)
    if cached:
        return ChatResponse(**cached, cached=True)

    start_time = time.time()

    try:
        result = rag_service.answer(payload.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(exc)}")

    latency_ms = round((time.time() - start_time) * 1000, 2)

    db: Session = SessionLocal()
    try:
        log = ConversationLog(
            session_id=payload.session_id,
            question=payload.question,
            answer=result["answer"],
            retrieved_context=result["retrieved_context"],
        )
        db.add(log)
        db.commit()
    finally:
        db.close()

    response_payload = {
        "answer": result["answer"],
        "sources": result["sources"],
        "safety_alert": result["safety_alert"],
        "safety_message": result["safety_message"],
    }

    set_cached_value(cache_key, response_payload)

    print(f"[chat] latency_ms={latency_ms} session_id={payload.session_id}")

    return ChatResponse(**response_payload, cached=False)


@app.post("/debug/retrieve", response_model=RetrieveDebugResponse)
def debug_retrieve(payload: ChatRequest):
    global rag_service

    if rag_service is None:
        raise HTTPException(status_code=500, detail="RAG service is not initialized")

    try:
        results = rag_service.retrieve_debug(payload.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(exc)}")

    return RetrieveDebugResponse(
        question=payload.question,
        top_k=len(results),
        results=results,
    )