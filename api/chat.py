"""Chat + /btw endpoints, streamed as Server-Sent Events.

The main chat drives the async graph with ``astream(stream_mode="messages")`` and
forwards `generate_answer` tokens as SSE `token` events, then a final `done` event
carrying the answer, a serialized graph-state snapshot (for the UI inspector), and
measured observability. `/btw` bridges the sync `handle_btw` generator to SSE.
Streaming and telemetry mirror the original `app.py` behaviour exactly.
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import AIMessageChunk, HumanMessage
from pydantic import BaseModel

from api import session_store
from api.deps import get_current_user, get_graph
from backend.btw_handler import handle_btw
from backend.config import OPENAI_CHAT_MODEL
from backend.telemetry import monotonic_seconds, record_run

router = APIRouter(prefix="/api", tags=["chat"])

_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


class ChatRequest(BaseModel):
    message: str


class BtwRequest(BaseModel):
    query: str


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _serialize_state(values: dict) -> dict:
    out: dict = {}
    for key, value in values.items():
        if key == "messages":
            out[key] = [
                {
                    "type": type(m).__name__,
                    "content": (
                        m.content[:300]
                        if isinstance(m.content, str)
                        else repr(m.content)[:300]
                    ),
                }
                for m in (value or [])
            ]
        elif key == "retrieved_docs":
            out[key] = [
                {"content": d.page_content[:300], "metadata": d.metadata}
                for d in (value or [])
            ]
        else:
            out[key] = value
    return out


@router.post("/sessions/{sid}/chat")
async def chat(
    sid: str,
    body: ChatRequest,
    graph=Depends(get_graph),
    user: str = Depends(get_current_user),
):
    async def event_stream():
        input_state = {
            "messages": [HumanMessage(content=body.message)],
            "session_id": sid,
            "query": body.message,
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
        usage_callback = UsageMetadataCallbackHandler()
        config = {"configurable": {"thread_id": sid}, "callbacks": [usage_callback]}
        started = monotonic_seconds()
        response_text = ""
        try:
            async for chunk, metadata in graph.astream(
                input_state, config, stream_mode="messages"
            ):
                # Forward only streaming token chunks. stream_mode="messages" also
                # emits the final aggregated AIMessage, which would duplicate the answer.
                if (
                    metadata.get("langgraph_node") == "generate_answer"
                    and isinstance(chunk, AIMessageChunk)
                    and chunk.content
                ):
                    response_text += chunk.content
                    yield _sse({"type": "token", "content": chunk.content})

            final_values = (await graph.aget_state(config)).values
            answer = final_values.get("answer") or response_text or "No response generated."
            record = record_run(
                source="api_chat",
                session_id=sid,
                state=final_values,
                latency_seconds=monotonic_seconds() - started,
                model_names=[OPENAI_CHAT_MODEL],
                usage_by_model=usage_callback.usage_metadata,
            )
            yield _sse(
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
            # Auto-name the session on its first turn (no-op if already named).
            await asyncio.to_thread(session_store.maybe_rename_session, sid, body.message)
        except Exception as exc:  # surface errors to the client instead of hanging
            yield _sse({"type": "error", "detail": str(exc)})

    return StreamingResponse(
        event_stream(), media_type="text/event-stream", headers=_SSE_HEADERS
    )


@router.post("/btw")
async def btw(body: BtwRequest, user: str = Depends(get_current_user)):
    async def event_stream():
        generator = handle_btw(body.query)
        sentinel = object()
        try:
            while True:
                chunk = await asyncio.to_thread(next, generator, sentinel)
                if chunk is sentinel:
                    break
                yield _sse({"type": "token", "content": chunk})
            yield _sse({"type": "done"})
        except Exception as exc:
            yield _sse({"type": "error", "detail": str(exc)})

    return StreamingResponse(
        event_stream(), media_type="text/event-stream", headers=_SSE_HEADERS
    )
