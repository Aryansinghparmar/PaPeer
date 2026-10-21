"""Auth modes, session ownership, rate limiting and the daily budget cap."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from api.ratelimit import SlidingWindowLimiter
from api.settings import Settings
from backend import telemetry
from test_api_support import FakeGraph, easyauth_header, make_client, parse_sse, zero_spend  # noqa: F401

CODE = "correct-horse-battery"


def test_auth_mode_endpoint_needs_no_auth(tmp_path):
    client = make_client(tmp_path, auth_mode="api_key", access_code=CODE)
    assert client.get("/api/auth/mode").json() == {"mode": "api_key"}
    assert make_client(tmp_path).get("/api/auth/mode").json() == {"mode": "none"}


def test_none_mode_uses_local_dev_principal(tmp_path):
    client = make_client(tmp_path)
    record = client.post("/api/sessions").json()
    assert record["owner"] == "local-dev"


def test_api_key_mode(tmp_path):
    client = make_client(tmp_path, auth_mode="api_key", access_code=CODE)

    missing = client.get("/api/sessions")
    assert missing.status_code == 401
    assert missing.json()["code"] == "unauthorized"
    assert client.get("/api/sessions", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/api/sessions", headers={"Authorization": f"Basic {CODE}"}).status_code == 401

    ok = client.post("/api/sessions", headers={"Authorization": f"Bearer {CODE}"})
    assert ok.status_code == 200
    assert ok.json()["owner"] == "access-code"
    # Pre-stream auth failure on the SSE route is plain JSON, not a stream.
    sid = ok.json()["id"]
    denied = client.post(f"/api/sessions/{sid}/chat", json={"message": "hi"})
    assert denied.status_code == 401
    assert denied.json()["code"] == "unauthorized"


def test_api_key_mode_requires_a_code(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "api_key")
    monkeypatch.delenv("PAPEER_ACCESS_CODE", raising=False)
    with pytest.raises(RuntimeError):
        Settings.from_env()
    monkeypatch.setenv("AUTH_MODE", "bogus")
    with pytest.raises(RuntimeError):
        Settings.from_env()


def test_easyauth_mode(tmp_path):
    client = make_client(tmp_path, auth_mode="easyauth")

    assert client.get("/api/sessions").status_code == 401
    bad = client.get("/api/sessions", headers={"X-MS-CLIENT-PRINCIPAL": "%%%not-base64"})
    assert bad.status_code == 401 and bad.json()["code"] == "unauthorized"
    assert client.get("/api/sessions", headers=easyauth_header({"userRoles": []})).status_code == 401

    swa = client.post(
        "/api/sessions",
        headers=easyauth_header({"identityProvider": "aad", "userId": "u-123", "userDetails": "a@b.c"}),
    )
    assert swa.json()["owner"] == "u-123"

    container_apps = client.post(
        "/api/sessions",
        headers=easyauth_header(
            {
                "auth_typ": "aad",
                "claims": [
                    {"typ": "name", "val": "Ada"},
                    {"typ": "http://schemas.microsoft.com/identity/claims/objectidentifier", "val": "oid-9"},
                ],
            }
        ),
    )
    assert container_apps.json()["owner"] == "oid-9"


def test_ownership_isolation_and_404s(tmp_path, monkeypatch):
    deleted_collections: list[str] = []
    monkeypatch.setattr("api.sessions.delete_session_collection", deleted_collections.append)
    monkeypatch.setattr("api.documents.list_papers", lambda sid: [])
    graph = FakeGraph()
    client = make_client(tmp_path, graph=graph, auth_mode="easyauth")
    alice = easyauth_header({"userId": "alice"})
    bob = easyauth_header({"userId": "bob"})

    sid = client.post("/api/sessions", headers=alice).json()["id"]
    client.post("/api/sessions", headers=bob)

    assert [s["id"] for s in client.get("/api/sessions", headers=alice).json()] == [sid]
    assert sid not in [s["id"] for s in client.get("/api/sessions", headers=bob).json()]

    for method, path, body in [
        ("get", f"/api/sessions/{sid}/messages", None),
        ("get", f"/api/sessions/{sid}/documents", None),
        ("post", f"/api/sessions/{sid}/chat", {"message": "hi"}),
        ("post", f"/api/sessions/{sid}/documents/url", {"urls": ["https://example.org"]}),
        ("post", f"/api/sessions/{sid}/documents/arxiv", {"query": "1706.03762"}),
        ("patch", f"/api/sessions/{sid}", {"first_message": "hi"}),
        ("delete", f"/api/sessions/{sid}", None),
    ]:
        response = client.request(method, path, json=body, headers=bob)
        assert response.status_code == 404, (method, path)
        assert response.json() == {"detail": "Session not found", "code": "not_found"}

    # Malformed and unknown ids are 404 too, even for a signed-in user.
    for bad_sid in ("foo", "../etc", "00000000-0000-4000-8000-000000000000"):
        assert client.get(f"/api/sessions/{bad_sid}/messages", headers=alice).status_code == 404
    assert graph.calls == []

    assert client.get(f"/api/sessions/{sid}/messages", headers=alice).status_code == 200
    response = client.delete(f"/api/sessions/{sid}", headers=alice)
    assert response.json() == {"deleted": True}
    assert deleted_collections == [sid]
    assert graph.checkpointer.deleted == [sid]
    assert client.get("/api/sessions", headers=alice).json() == []


def test_sliding_window_limiter():
    now = [0.0]
    limiter = SlidingWindowLimiter(clock=lambda: now[0])
    assert limiter.hit("chat", "k", 2, 60) is None
    assert limiter.hit("chat", "k", 2, 60) is None
    assert limiter.hit("chat", "k", 2, 60) == pytest.approx(60)
    assert limiter.hit("chat", "other", 2, 60) is None  # separate key
    assert limiter.hit("btw", "k", 2, 60) is None  # separate bucket
    now[0] = 60.5
    assert limiter.hit("chat", "k", 2, 60) is None


def test_chat_rate_limit_returns_429_before_streaming(tmp_path):
    client = make_client(tmp_path, rate_limit_chat=2)
    sid = client.post("/api/sessions").json()["id"]

    for _ in range(2):
        assert client.post(f"/api/sessions/{sid}/chat", json={"message": "q"}).status_code == 200
    limited = client.post(f"/api/sessions/{sid}/chat", json={"message": "q"})

    assert limited.status_code == 429
    assert limited.json()["code"] == "rate_limited"
    assert int(limited.headers["Retry-After"]) >= 1
    # Other route groups have their own budget.
    assert client.post("/api/btw", json={"query": "q"}).status_code == 200


def test_budget_cap_blocks_cost_bearing_routes(tmp_path, monkeypatch):
    monkeypatch.setattr("api.deps.spent_today_usd", lambda: 1.25)
    client = make_client(tmp_path, daily_budget_usd=1.0)
    sid = client.post("/api/sessions").json()["id"]

    for path, body in [
        (f"/api/sessions/{sid}/chat", {"message": "q"}),
        ("/api/btw", {"query": "q"}),
        (f"/api/sessions/{sid}/documents/url", {"urls": ["https://example.org"]}),
    ]:
        response = client.post(path, json=body)
        assert response.status_code == 429, path
        assert response.json()["code"] == "budget_exceeded"

    # Reading is still allowed, and 0 disables the cap.
    assert client.get("/api/sessions").status_code == 200
    unlimited = make_client(tmp_path, daily_budget_usd=0.0)
    assert parse_sse(unlimited.post("/api/btw", json={"query": "q"}).text)[-1] == {"type": "done"}


def test_ledger_seeds_today_spend_and_record_run_adds(tmp_path):
    now = datetime.now(timezone.utc)
    ledger = tmp_path / "runs.jsonl"
    rows = [
        {"timestamp": now.isoformat(), "estimated_cost_usd": 0.25},
        {"timestamp": now.isoformat(), "estimated_cost_usd": None},
        {"timestamp": (now - timedelta(days=1)).isoformat(), "estimated_cost_usd": 9.0},
    ]
    ledger.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\nnot json\n", encoding="utf-8"
    )
    try:
        assert telemetry.load_today_spend(ledger) == pytest.approx(0.25)
        telemetry.add_spend(0.5)
        assert telemetry.spent_today_usd() == pytest.approx(0.75)
    finally:
        telemetry.load_today_spend(tmp_path / "missing.jsonl")
    assert telemetry.spent_today_usd() == 0.0
