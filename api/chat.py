"""Chat + /btw endpoints, streamed as Server-Sent Events.

The main chat drives the async graph with
``astream(stream_mode=["messages", "tasks"])``:

- ``tasks`` events mark when each graph node starts; they become ``status``
  events (``stage`` = node name, ``label`` = short human text), so the UI can
  show progress during the non-streaming nodes (routing, retrieval, claim
  verification) instead of a blank "Thinking...".
- ``messages`` events carry LLM token chunks; only ``generate_answer``
  ``AIMessageChunk``s are forwarded as ``token`` events (the router, agent and
  grader LLM calls stay internal, and the final aggregated ``AIMessage`` would
  duplicate the answer).

A final ``done`` event carries the answer, a sanitised graph-state snapshot for
the UI inspector, and measured observability. Heartbeats, the overall timeout
and error mapping live in ``api.streaming``. ``/btw`` bridges the sync
``handle_btw`` generator from a worker thread.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import PurePath
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from langchain_core.callbacks import (
    UsageMetadataCallbackHandler,
    get_usage_metadata_callback,
)
from langchain_core.messages import AIMessageChunk, HumanMessage
from pydantic import BaseModel, Field

from api import session_store
from api.deps import (
    get_graph,
    get_settings,
    get_store,
    owned_session,
    rate_limited,
    within_budget,
)
from api.settings import MAX_BTW_QUERY_CHARS, MAX_MESSAGE_CHARS, Settings
from api.streaming import SSE_HEADERS, Emit, sse_events
from backend.config import OPENAI_CHAT_MODEL
from backend.telemetry import monotonic_seconds, record_run

router = APIRouter(prefix="/api", tags=["chat"])

# Human-readable progress text per graph node (see backend/rag_graph.py).
NODE_LABELS = {
    "router": "Routing question",
    "agent_node": "Planning search",
    "retrieval": "Searching papers",
    "relevancy_check": "Checking relevance",
    "query_rewrite": "Rewriting query",
    "verify_claim": "Searching arXiv and the web",
    "generate_answer": "Writing answer",
}
_SNIPPET_CHARS = 300
_PATH_METADATA_KEYS = ("source", "file_path")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


class BtwRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_BTW_QUERY_CHARS)


def _safe_metadata(metadata: dict) -> dict:
    """Drop server file-system paths (PyMuPDF puts the temp file path in
    ``source``/``file_path``); URLs are kept since the user supplied them."""

    out = dict(metadata or {})
    for key in _PATH_METADATA_KEYS:
        value = out.get(key)
        if isinstance(value, str) and not value.startswith(("http://", "https://")):
            out[key] = PurePath(value.replace("\\", "/")).name
    return out


def _snippet(content: Any) -> str:
    text = content if isinstance(content, str) else repr(content)
    return text[:_SNIPPET_CHARS]


def _serialize_state(values: dict) -> dict:
    out: dict = {}
    for key, value in values.items():
        if key == "messages":
            out[key] = [
                {"type": type(m).__name__, "content": _snippet(getattr(m, "content", m))}
                for m in (value or [])
            ]
        elif key == "retrieved_docs":
            out[key] = [
                {
                    "content": _snippet(getattr(d, "page_content", "")),
                    "metadata": _safe_metadata(getattr(d, "metadata", {})),
                }
                for d in (value or [])
            ]
        else:
            out[key] = value
    return out


def _input_state(sid: str, message: str) -> dict:
    return {
        "messages": [HumanMessage(content=message)],
        "session_id": sid,
        "query": message,
        "route": None,
        "retrieved_docs": [],
        "retrieval_attempts": 0,
        "claim_verdict": None,
        "claim_source": None,
        "superseding_papers": [],
        "answer": None,
        "is_relevant": None,
        "rewrite_count": 0,
        "force_retrieval": False,
    }


@router.post(
    "/sessions/{sid}/chat",
    dependencies=[Depends(rate_limited("chat")), Depends(within_budget)],
)
async def chat(
    sid: str,
    body: ChatRequest,
    session: dict = Depends(owned_session),
    graph=Depends(get_graph),
    settings: Settings = Depends(get_settings),
    store: session_store.SessionStore = Depends(get_store),
):
    async def produce(emit: Emit) -> None:
        usage_callback = UsageMetadataCallbackHandler()
        config = {"configurable": {"thread_id": sid}, "callbacks": [usage_callback]}
        started = monotonic_seconds()
        response_text = ""
        async for mode, payload in graph.astream(
            _input_state(sid, body.message), config, stream_mode=["messages", "tasks"]
        ):
            if mode == "tasks":
                name = payload.get("name")
                # Task-start payloads have "input"; task-result payloads have "result".
                if name in NODE_LABELS and "result" not in payload:
                    emit({"type": "status", "stage": name, "label": NODE_LABELS[name]})
            elif mode == "messages":
                chunk, metadata = payload
                if (
                    metadata.get("langgraph_node") == "generate_answer"
                    and isinstance(chunk, AIMessageChunk)
                    and chunk.content
                ):
                    response_text += chunk.content
                    emit({"type": "token", "content": chunk.content})

        final_values = (await graph.aget_state(config)).values or {}
        answer = final_values.get("answer") or response_text or "No response generated."
        record = await asyncio.to_thread(
            record_run,
            source="api_chat",
            session_id=sid,
            state=final_values,
            latency_seconds=monotonic_seconds() - started,
            model_names=[OPENAI_CHAT_MODEL],
            usage_by_model=usage_callback.usage_metadata,
        )
        emit(
            {
                "type": "done",
                "answer": answer,
                "state": _serialize_state(final_values),
                "observability": {
                    "latency_seconds": record["latency_seconds"],
                    "estimated_cost_usd": record["estimated_cost_usd"],
                    "input_tokens": record["input_tokens"],
                    "output_tokens": record["output_tokens"],
                },
            }
        )
        # Auto-name the session on its first turn. The stream stays open (with
        # heartbeats) so the UI's session-list refresh after close sees the name.
        if not session.get("is_named"):
            await asyncio.to_thread(
                session_store.rename_and_record, store, sid, body.message
            )

    return StreamingResponse(
        sse_events(
            produce,
            heartbeat_seconds=settings.heartbeat_seconds,
            timeout_seconds=settings.request_timeout_seconds,
            context="chat",
        ),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.post("/btw", dependencies=[Depends(rate_limited("btw")), Depends(within_budget)])
async def btw(
    request: Request,
    body: BtwRequest,
    settings: Settings = Depends(get_settings),
):
    handle_btw = request.app.state.btw_handler

    async def produce(emit: Emit) -> None:
        loop = asyncio.get_running_loop()
        stop = threading.Event()

        def work() -> None:
            started = monotonic_seconds()
            # The callback context lives entirely inside this worker thread.
            with get_usage_metadata_callback() as usage:
                for chunk in handle_btw(body.query):
                    if stop.is_set():
                        break
                    loop.call_soon_threadsafe(emit, {"type": "token", "content": chunk})
            record_run(
                source="api_btw",
                session_id="btw",
                state={"route": "btw"},
                latency_seconds=monotonic_seconds() - started,
                model_names=[OPENAI_CHAT_MODEL],
                usage_by_model=usage.usage_metadata,
            )

        try:
            await asyncio.to_thread(work)
        finally:
            stop.set()  # a cancelled request stops the worker at the next chunk
        emit({"type": "done"})

    return StreamingResponse(
        sse_events(
            produce,
            heartbeat_seconds=settings.heartbeat_seconds,
            timeout_seconds=settings.request_timeout_seconds,
            context="btw",
        ),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
