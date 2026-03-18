from pydantic import BaseModel
from typing import List


class ChatRequest(BaseModel):
    session_id: str
    question: str


class SourceItem(BaseModel):
    source: str
    focus: str
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    cached: bool = False
    safety_alert: bool = False
    safety_message: str = ""


class HealthResponse(BaseModel):
    status: str
    app: str
    environment: str


class RetrieveResultItem(BaseModel):
    score: float
    source: str
    focus: str
    related_question: str
    content: str


class RetrieveDebugResponse(BaseModel):
    question: str
    top_k: int
    results: List[RetrieveResultItem]