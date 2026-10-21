"""Upload/URL ingestion limits, duplicate skipping and partial-failure shape."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from backend.paper_loader import UnsafeURLError
from test_api_support import make_client, zero_spend  # noqa: F401


@pytest.fixture
def ingest(monkeypatch):
    """Fake loaders and vector store; records what would be embedded."""

    added: list[tuple[str, list[Document]]] = []
    loaded_paths: list[str] = []

    def fake_load_document(path):
        loaded_paths.append(path)
        return [Document(page_content="chunk", metadata={"source": path, "file_path": path})]

    monkeypatch.setattr("api.documents.list_papers", lambda sid: ["existing"])
    monkeypatch.setattr("api.documents.add_paper", lambda docs, sid: added.append((sid, docs)))
    monkeypatch.setattr("api.documents.load_document", fake_load_document)
    return added, loaded_paths


def _client_and_sid(tmp_path, **overrides):
    client = make_client(tmp_path, **overrides)
    return client, client.post("/api/sessions").json()["id"]


def test_partial_failure_duplicate_and_success(tmp_path, ingest):
    added, loaded_paths = ingest
    client, sid = _client_and_sid(tmp_path)

    response = client.post(
        f"/api/sessions/{sid}/documents",
        files=[
            ("files", ("notes.txt", b"plain text", "text/plain")),
            ("files", ("malware.exe", b"MZ...", "application/octet-stream")),
            ("files", ("existing.md", b"# dup", "text/markdown")),
            ("files", ("fake.pdf", b"hello, not a pdf", "application/pdf")),
            ("files", ("real.pdf", b"%PDF-1.7\n...", "application/pdf")),
        ],
    )

    assert response.status_code == 200
    body = response.json()
    assert body["added"] == ["notes.txt", "real.pdf"]
    assert body["skipped"] == [{"name": "existing.md", "reason": "already loaded in this session"}]
    assert [f["name"] for f in body["failed"]] == ["malware.exe", "fake.pdf"]
    assert "Unsupported file type" in body["failed"][0]["error"]
    assert body["failed"][1]["error"] == "Not a valid PDF file"
    # Stored metadata carries the user's filename, never the server temp path.
    stored = [doc.metadata for _, docs in added for doc in docs]
    assert [m["title"] for m in stored] == ["notes", "real"]
    assert [m["source"] for m in stored] == ["notes.txt", "real.pdf"]
    assert all("file_path" not in m for m in stored)
    assert len(loaded_paths) == 2


def test_same_file_twice_in_one_batch_is_skipped(tmp_path, ingest):
    client, sid = _client_and_sid(tmp_path)
    body = client.post(
        f"/api/sessions/{sid}/documents",
        files=[("files", ("a.txt", b"x", "text/plain")), ("files", ("a.txt", b"x", "text/plain"))],
    ).json()
    assert body["added"] == ["a.txt"]
    assert body["skipped"][0]["name"] == "a.txt"


def test_too_many_files(tmp_path, ingest):
    client, sid = _client_and_sid(tmp_path)
    files = [("files", (f"f{i}.txt", b"x", "text/plain")) for i in range(6)]
    response = client.post(f"/api/sessions/{sid}/documents", files=files)
    assert response.status_code == 422
    assert response.json()["code"] == "too_many_files"


def test_per_file_size_limit(tmp_path, ingest):
    added, _ = ingest
    client, sid = _client_and_sid(tmp_path, max_upload_bytes=1024)
    body = client.post(
        f"/api/sessions/{sid}/documents",
        files=[
            ("files", ("big.txt", b"x" * 5000, "text/plain")),
            ("files", ("small.txt", b"x" * 100, "text/plain")),
        ],
    ).json()
    assert body["added"] == ["small.txt"]
    assert body["failed"][0]["name"] == "big.txt"
    assert "limit" in body["failed"][0]["error"]
    assert len(added) == 1


def test_request_body_cap_returns_413(tmp_path, ingest):
    added, _ = ingest
    client, sid = _client_and_sid(tmp_path, max_upload_bytes=1024, max_files_per_request=1)
    response = client.post(
        f"/api/sessions/{sid}/documents",
        files=[("files", ("huge.txt", b"x" * (300 * 1024), "text/plain"))],
    )
    assert response.status_code == 413
    assert response.json()["code"] == "payload_too_large"
    assert added == []


def test_unexpected_error_is_generic(tmp_path, ingest, monkeypatch):
    def boom(docs, sid):
        raise RuntimeError("qdrant at 10.0.0.5:6333 refused api-key abc")

    monkeypatch.setattr("api.documents.add_paper", boom)
    client, sid = _client_and_sid(tmp_path)
    body = client.post(
        f"/api/sessions/{sid}/documents", files=[("files", ("a.txt", b"x", "text/plain"))]
    ).json()
    assert body["added"] == []
    assert "10.0.0.5" not in body["failed"][0]["error"]
    assert "error id" in body["failed"][0]["error"]


def test_urls_shape_and_limits(tmp_path, ingest, monkeypatch):
    added, _ = ingest

    def fake_load_webpage(url):
        if "internal" in url:
            raise UnsafeURLError("URL points to a private or reserved network address")
        title = "existing" if "dup" in url else f"Page {url[-1]}"
        return [Document(page_content="t", metadata={"source": url, "title": title})]

    monkeypatch.setattr("api.documents.load_webpage", fake_load_webpage)
    client, sid = _client_and_sid(tmp_path)

    body = client.post(
        f"/api/sessions/{sid}/documents/url",
        json={"urls": ["https://a.example/1", "http://internal/", "https://dup.example/x", " "]},
    ).json()

    assert body["loaded"] == ["Page 1"]
    assert body["skipped"] == [
        {"url": "https://dup.example/x", "name": "existing", "reason": "already loaded in this session"}
    ]
    assert body["failed"] == [
        {"url": "http://internal/", "error": "URL points to a private or reserved network address"}
    ]
    assert len(added) == 1

    too_many = [f"https://a.example/{i}" for i in range(6)]
    assert client.post(f"/api/sessions/{sid}/documents/url", json={"urls": too_many}).status_code == 422


def test_arxiv_shape_and_idempotence(tmp_path, ingest, monkeypatch):
    added, _ = ingest
    monkeypatch.setattr(
        "api.documents.load_arxiv",
        lambda q: [Document(page_content="t", metadata={"title": "existing"})],
    )
    client, sid = _client_and_sid(tmp_path)
    response = client.post(f"/api/sessions/{sid}/documents/arxiv", json={"query": "1706.03762"})
    assert response.json() == {"loaded": "existing"}
    assert added == []  # already in the session, not embedded again


def test_body_cap_counts_streamed_chunks_without_content_length():
    """A chunked upload (no Content-Length) is cut off once it crosses the cap."""

    import asyncio

    from api.documents import UploadSizeLimit
    from api.settings import Settings

    reads: list[int] = []

    async def app(scope, receive, send):
        while True:  # consume the body like the multipart parser would
            message = await receive()
            reads.append(len(message.get("body", b"")))
            if not message.get("more_body"):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    middleware = UploadSizeLimit(app, Settings(max_upload_bytes=1024, max_files_per_request=1))
    chunks = [b"x" * (32 * 1024)] * 10  # 320 KB in 32 KB pieces, cap is 65 KB

    async def receive():
        body = chunks.pop(0)
        return {"type": "http.request", "body": body, "more_body": bool(chunks)}

    sent: list[dict] = []

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/sessions/abc/documents",
        "headers": [(b"transfer-encoding", b"chunked")],
    }
    asyncio.run(middleware(scope, receive, send))

    assert sent[0]["status"] == 413
    assert len(reads) < 4  # stopped reading early instead of draining 320 KB
