"""FastAPI server exposing the assistant over Server-Sent Events.

Run: ``uvicorn app.server:app --reload --port 8000``
"""

from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import settings
from .engine import RagEngine, sse
from .observability import get_logger

log = get_logger("rag.server")
app = FastAPI(title="Portfolio RAG Assistant", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

_engine: RagEngine | None = None


def get_engine() -> RagEngine:
    global _engine
    if _engine is None:
        _engine = RagEngine()
    return _engine


class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=settings.max_question_chars)
    history: list[Turn] = Field(default_factory=list, max_length=settings.max_history_turns * 2)


# Simple in-memory sliding-window rate limiter keyed by client IP. For a
# multi-instance deployment put this in Redis or at the gateway.
_hits: dict[str, deque] = defaultdict(deque)


def rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= settings.rate_limit_per_minute:
        raise HTTPException(status_code=429, detail="Too many requests. Please slow down.")
    q.append(now)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "model": settings.model, "retriever": settings.retriever_backend}


@app.post("/chat")
def chat(req: ChatRequest, request: Request) -> StreamingResponse:
    rate_limit(request)
    engine = get_engine()
    history = [t.model_dump() for t in req.history]

    def gen():
        for event in engine.chat(req.message, history):
            yield sse(event)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
