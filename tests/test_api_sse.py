"""SSE framing, heartbeat, timeout and error mapping for /chat and /btw."""

from __future__ import annotations

import time
from datetime import datetime

from langchain_core.documents import Document
from langgraph.errors import GraphRecursionError

from test_api_support import FakeGraph, make_client, parse_sse, zero_spend  # noqa: F401


def _new_session(client) -> str:
    response = client.post("/api/sessions")
    assert response.status_code == 200
    return response.json()["id"]


def test_chat_stream_order_status_tokens_done(tmp_path):
    graph = FakeGraph()
    client = make_client(tmp_path, graph=graph)
    sid = _new_session(client)

    response = client.post(f"/api/sessions/{sid}/chat", json={"message": "hi"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [e for e in parse_sse(response.text) if e != "ping"]
    types = [e["type"] for e in events]
    assert types == ["status", "status", "token", "token", "done"]
    assert [e["stage"] for e in events[:2]] == ["router", "generate_answer"]
    assert events[0]["label"] == "Routing question"
    assert events[1]["label"] == "Writing answer"
    # Only generate_answer chunks are tokens; the router chunk and the final
    # aggregated AIMessage are not forwarded.
    assert "".join(e["content"] for e in events if e["type"] == "token") == "Hello world"
    done = events[-1]
    assert done["answer"] == "Hello world"
    assert set(done["observability"]) == {
        "latency_seconds",
        "estimated_cost_usd",
        "input_tokens",
        "output_tokens",
    }
    assert graph.calls[0]["stream_mode"] == ["messages", "tasks"]
    assert graph.calls[0]["config"]["configurable"]["thread_id"] == sid
    # First turn auto-names the session (fake namer).
    assert client.get("/api/sessions").json()[0]["name"] == "Named session"


def test_done_state_hides_server_paths_and_handles_odd_types(tmp_path):
    graph = FakeGraph(
        final_values={
            "answer": "ok",
            "retrieved_docs": [
                Document(
                    page_content="text",
                    metadata={
                        "source": "C:\\Users\\srv\\AppData\\Local\\Temp\\tmpab12.pdf",
                        "file_path": "/tmp/tmpab12.pdf",
                        "url": "https://example.org/a",
                    },
                ),
                Document(page_content="web", metadata={"source": "https://example.org/p"}),
            ],
            "when": datetime(2026, 1, 2, 3, 4, 5),
            "odd": {1, 2},
        }
    )
    client = make_client(tmp_path, graph=graph)
    sid = _new_session(client)

    events = parse_sse(client.post(f"/api/sessions/{sid}/chat", json={"message": "q"}).text)

    done = [e for e in events if e != "ping" and e["type"] == "done"][0]
    first, second = done["state"]["retrieved_docs"]
    assert first["metadata"]["source"] == "tmpab12.pdf"
    assert first["metadata"]["file_path"] == "tmpab12.pdf"
    assert first["metadata"]["url"] == "https://example.org/a"
    assert second["metadata"]["source"] == "https://example.org/p"
    assert done["state"]["when"] == "2026-01-02T03:04:05"
    assert sorted(done["state"]["odd"]) == [1, 2]


def test_heartbeat_while_graph_stalls(tmp_path):
    client = make_client(
        tmp_path, graph=FakeGraph(stall_seconds=0.4), heartbeat_seconds=0.05
    )
    sid = _new_session(client)

    events = parse_sse(client.post(f"/api/sessions/{sid}/chat", json={"message": "q"}).text)

    assert events.index("ping") < next(
        i for i, e in enumerate(events) if e != "ping" and e["type"] == "status"
    )
    assert events.count("ping") >= 2
    assert events[-1]["type"] == "done"


def test_internal_error_is_sanitised(tmp_path):
    secret = "connection to 10.1.2.3 failed with key sk-test"
    client = make_client(tmp_path, graph=FakeGraph(events=[], exc=RuntimeError(secret)))
    sid = _new_session(client)

    events = parse_sse(client.post(f"/api/sessions/{sid}/chat", json={"message": "q"}).text)

    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "internal"
    assert "10.1.2.3" not in events[-1]["detail"] and "sk-test" not in events[-1]["detail"]
    assert "error id" in events[-1]["detail"]


def test_recursion_limit_maps_to_loop_limit(tmp_path):
    client = make_client(
        tmp_path, graph=FakeGraph(events=[], exc=GraphRecursionError("Recursion limit of 9"))
    )
    sid = _new_session(client)

    events = parse_sse(client.post(f"/api/sessions/{sid}/chat", json={"message": "q"}).text)

    assert events[-1] == {**events[-1], "type": "error", "code": "loop_limit"}


def test_overall_timeout(tmp_path):
    client = make_client(
        tmp_path,
        graph=FakeGraph(stall_seconds=5),
        heartbeat_seconds=0.05,
        request_timeout_seconds=0.3,
    )
    sid = _new_session(client)

    started = time.monotonic()
    events = parse_sse(client.post(f"/api/sessions/{sid}/chat", json={"message": "q"}).text)

    assert time.monotonic() - started < 3
    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "timeout"
    assert not any(e != "ping" and e["type"] == "done" for e in events)


def test_btw_stream_tokens_then_done(tmp_path):
    client = make_client(tmp_path)

    events = parse_sse(client.post("/api/btw", json={"query": "weather?"}).text)

    assert events == [
        {"type": "token", "content": "side "},
        {"type": "token", "content": "answer"},
        {"type": "done"},
    ]


def test_btw_error_is_sanitised(tmp_path):
    def broken(query):
        raise RuntimeError("tavily key tvly-secret rejected")
        yield  # pragma: no cover

    client = make_client(tmp_path, btw_handler=broken)

    events = parse_sse(client.post("/api/btw", json={"query": "q"}).text)

    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "internal"
    assert "tvly" not in events[-1]["detail"]


def test_btw_heartbeat(tmp_path):
    def slow(query):
        time.sleep(0.3)
        yield "late"

    client = make_client(tmp_path, btw_handler=slow, heartbeat_seconds=0.05)

    events = parse_sse(client.post("/api/btw", json={"query": "q"}).text)

    assert events[0] == "ping"
    assert events[-2:] == [{"type": "token", "content": "late"}, {"type": "done"}]


def test_input_length_limits(tmp_path):
    client = make_client(tmp_path)
    sid = _new_session(client)

    assert client.post(f"/api/sessions/{sid}/chat", json={"message": "x" * 4001}).status_code == 422
    assert client.post(f"/api/sessions/{sid}/chat", json={"message": ""}).status_code == 422
    assert client.post("/api/btw", json={"query": "x" * 2001}).status_code == 422
    assert client.post(f"/api/sessions/{sid}/chat", json={"message": "x" * 4000}).status_code == 200
