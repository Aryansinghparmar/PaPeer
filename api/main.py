"""Papeer FastAPI backend.

Wraps the existing LangGraph RAG workflow (unchanged) in a REST + SSE API so a
React SPA can replace the Streamlit UI. The compiled async graph is built once in
the app lifespan and shared across requests. Auth is enforced by the platform
(Azure Easy Auth / Static Web Apps); `api.deps.get_current_user` reads the
forwarded identity header.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.chat import router as chat_router
from api.documents import router as documents_router
from api.sessions import router as sessions_router
from backend.config import CHECKPOINT_DB_PATH
from backend.rag_graph import build_graph_async


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.graph = await build_graph_async(CHECKPOINT_DB_PATH)
    yield


app = FastAPI(title="Papeer API", version="1.0.0", lifespan=lifespan)

_origins = [
    o.strip()
    for o in os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://localhost:3000"
    ).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sessions_router)
app.include_router(documents_router)
app.include_router(chat_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
