"""Shared fakes for the API tests (no tests here).

The app is built with ``create_app`` and a fake graph / fake /btw handler, and
the TestClient is used without its context manager, so the lifespan (which
would build the real graph and read the real ledger) never runs.
"""

from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from api.main import create_app
from api.session_store import SessionStore
from api.settings import Settings


def task_start(name: str) -> tuple:
    return ("tasks", {"id": f"id-{name}", "name": name, "input": {}, "triggers": []})


def task_end(name: str) -> tuple:
    return (
        "tasks",
        {"id": f"id-{name}", "name": name, "error": None, "result": {}, "interrupts": []},
    )


def token(node: str, text: str) -> tuple:
    return ("messages", (AIMessageChunk(content=text), {"langgraph_node": node}))


DEFAULT_EVENTS = [
    task_start("router"),
    ("messages", (AIMessageChunk(content="internal router json"), {"langgraph_node": "router"})),
    task_end("router"),
    task_start("generate_answer"),
    token("generate_answer", "Hello"),
    token("generate_answer", " world"),
    # The aggregated final message must not be forwarded a second time.
    ("messages", (AIMessage(content="Hello world"), {"langgraph_node": "generate_answer"})),
    task_end("generate_answer"),
]


class FakeCheckpointer:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    async def adelete_thread(self, thread_id: str) -> None:
        self.deleted.append(thread_id)


class FakeGraph:
    """Implements the slice of the compiled graph the API uses."""

    def __init__(self, events=None, final_values=None, exc=None, stall_seconds=0.0):
        self.events = DEFAULT_EVENTS if events is None else events
        self.final_values = final_values or {
            "messages": [HumanMessage(content="hi"), AIMessage(content="Hello world")],
            "answer": "Hello world",
            "route": "direct_answer",
            "retrieved_docs": [],
        }
        self.exc = exc
        self.stall_seconds = stall_seconds
        self.checkpointer = FakeCheckpointer()
        self.calls: list[dict] = []

    async def astream(self, input_state, config, stream_mode=None):
        self.calls.append({"input": input_state, "config": config, "stream_mode": stream_mode})
        if self.stall_seconds:
            await asyncio.sleep(self.stall_seconds)
        for event in self.events:
            yield event
        if self.exc is not None:
            raise self.exc

    async def aget_state(self, config):
        return SimpleNamespace(values=self.final_values)


def fake_btw(query: str):
    yield "side "
    yield "answer"


def make_client(tmp_path, *, graph=None, btw_handler=None, namer=None, **overrides):
    values = {
        "sessions_file": str(tmp_path / "sessions.json"),
        "static_dir": str(tmp_path / "no-static"),
        "cors_origins": ["http://localhost:5173"],
    }
    values.update(overrides)
    app = create_app(
        Settings(**values), graph=graph or FakeGraph(), btw_handler=btw_handler or fake_btw
    )
    app.state.session_store = SessionStore(
        values["sessions_file"], namer=namer or (lambda message: "Named session")
    )
    return TestClient(app)


def parse_sse(text: str) -> list:
    """Return data events as dicts and comment frames as the string 'ping'."""

    events: list = []
    for frame in text.split("\n\n"):
        frame = frame.strip("\n")
        if not frame:
            continue
        if frame.startswith(":"):
            events.append("ping")
        elif frame.startswith("data: "):
            events.append(json.loads(frame[len("data: "):]))
        else:  # pragma: no cover - would be a framing bug
            raise AssertionError(f"bad SSE frame: {frame!r}")
    return events


def easyauth_header(payload: dict) -> dict:
    raw = base64.b64encode(json.dumps(payload).encode()).decode()
    return {"X-MS-CLIENT-PRINCIPAL": raw}


@pytest.fixture(autouse=True)
def zero_spend(monkeypatch):
    """Isolate tests from the process-wide daily spend total. Test modules
    import this name so the autouse fixture applies to them."""

    monkeypatch.setattr("api.deps.spent_today_usd", lambda: 0.0)
