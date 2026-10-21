"""Server-Sent Events plumbing shared by /chat and /btw.

A producer coroutine pushes event dicts onto a queue; ``sse_events`` drains it
and turns it into ``data: {json}\\n\\n`` frames. Draining with a timeout gives
two things a plain ``async for`` cannot:

- a heartbeat comment (``: ping``) whenever nothing was sent for
  ``heartbeat_seconds``, so idle proxies (nginx, Azure ingress) keep the
  connection open during slow nodes such as claim verification;
- an overall deadline: past ``timeout_seconds`` the producer is cancelled and
  the client gets an ``error`` event with code ``timeout``.

Errors are reported with a generic message and an error id; the exception
itself is only logged server-side.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import date, datetime
from pathlib import PurePath
from typing import Any

from backend.logging_config import get_logger

logger = get_logger(__name__)

PING = ": ping\n\n"
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

Emit = Callable[[dict], None]

_ERROR_TEXT = {
    "timeout": "The request took too long and was stopped. Please try again.",
    "loop_limit": "This question needed too many retrieval steps. Try rephrasing it more specifically.",
}


def json_default(value: Any) -> Any:
    """Fallback for values json can't encode (dates, paths, sets, objects)."""

    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, PurePath):
        return value.name
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return str(value)


def sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False, default=json_default)}\n\n"


def _is_timeout(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):  # includes asyncio.TimeoutError on 3.11+
        return True
    try:
        import httpx
        import openai

        return isinstance(exc, (openai.APITimeoutError, httpx.TimeoutException))
    except ImportError:  # pragma: no cover
        return False


def error_event(exc: BaseException, context: str) -> dict:
    """Map an exception to a safe client event; log the real one with an id."""

    from langgraph.errors import GraphRecursionError

    error_id = uuid.uuid4().hex[:12]
    if isinstance(exc, GraphRecursionError):
        code = "loop_limit"
    elif _is_timeout(exc):
        code = "timeout"
    else:
        code = "internal"
    logger.error(
        "stream_error context=%s code=%s error_id=%s", context, code, error_id, exc_info=exc
    )
    detail = _ERROR_TEXT.get(code, "Something went wrong while answering.")
    return {"type": "error", "detail": f"{detail} (error id {error_id})", "code": code}


class _Failure:
    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


_END = object()


async def sse_events(
    produce: Callable[[Emit], Awaitable[None]],
    *,
    heartbeat_seconds: float,
    timeout_seconds: float,
    context: str,
) -> AsyncIterator[str]:
    queue: asyncio.Queue = asyncio.Queue()

    async def runner() -> None:
        try:
            await produce(queue.put_nowait)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            queue.put_nowait(_Failure(exc))
        finally:
            queue.put_nowait(_END)

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    task = asyncio.create_task(runner())
    finished = False  # a terminal done/error event was sent
    try:
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                if not finished:
                    logger.warning("stream_timeout context=%s seconds=%s", context, timeout_seconds)
                    yield sse(
                        {
                            "type": "error",
                            "detail": _ERROR_TEXT["timeout"],
                            "code": "timeout",
                        }
                    )
                break
            try:
                item = await asyncio.wait_for(
                    queue.get(), timeout=min(heartbeat_seconds, remaining)
                )
            except TimeoutError:
                if loop.time() < deadline:
                    yield PING
                continue
            if item is _END:
                break
            if isinstance(item, _Failure):
                if not finished:
                    finished = True
                    yield sse(error_event(item.exc, context))
                continue
            if item.get("type") in ("done", "error"):
                finished = True
            yield sse(item)
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
