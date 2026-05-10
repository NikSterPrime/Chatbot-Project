from uuid import uuid4
from pathlib import Path
from dotenv import load_dotenv

# Load local .env (kept in src/.env) so optional API keys become available to the web API process.
load_dotenv(Path(__file__).resolve().parent / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from .chatbot import ChatSessionMemory, chat_once
except ImportError:
    from chatbot import ChatSessionMemory, chat_once

try:
    from .podcast_recommender import get_filter_options
except ImportError:
    from podcast_recommender import get_filter_options


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    response: str
    intent: str
    confidence: float
    margin: float
    source: str | None = None


app = FastAPI(title="Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: dict[str, ChatSessionMemory] = {}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/podcast-filters")
def podcast_filters():
    return get_filter_options()


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    session_id = payload.session_id or str(uuid4())
    memory = sessions.get(session_id)
    if memory is None:
        memory = ChatSessionMemory()
        sessions[session_id] = memory

    result = chat_once(payload.message, memory)
    return ChatResponse(
        session_id=session_id,
        response=result["response"],
        intent=result["intent"],
        confidence=float(result["confidence"]),
        margin=float(result["margin"]),
        source=result.get("source"),
    )


@app.post("/api/reset/{session_id}")
def reset_session(session_id: str):
    memory = sessions.get(session_id)
    if memory:
        memory.clear()
    return {"ok": True}
