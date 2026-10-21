"""Document ingestion endpoints (file upload, web URL, ArXiv).

These are sync ``def`` handlers on purpose: FastAPI runs them in a thread pool,
so the blocking ingestion calls (``load_document``, ``add_paper``, network I/O)
don't stall the event loop.

Each batch is error-and-continue: every file or URL lands in exactly one of
``added``/``loaded``, ``skipped`` (already in the session) or ``failed``, so a
bad file never hides that the others were ingested.

Limits: at most ``MAX_FILES_PER_REQUEST`` files of ``MAX_UPLOAD_MB`` each. The
request body as a whole is capped while it streams in (``UploadSizeLimit``
below), so an oversized upload is cut off instead of being spooled to disk,
and each file is copied in chunks with its own size check. PDFs must start
with the ``%PDF-`` signature.
"""

from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path, PurePath
from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile
from pydantic import BaseModel, Field

from api.deps import ApiError, get_settings, owned_session, rate_limited, within_budget
from api.settings import (
    MAX_ARXIV_QUERY_CHARS,
    MAX_URL_CHARS,
    MAX_URLS_PER_REQUEST,
    Settings,
)
from backend.logging_config import get_logger
from backend.paper_loader import UnsafeURLError, load_arxiv, load_document, load_webpage
from backend.vector_store import add_paper, list_papers

logger = get_logger(__name__)

router = APIRouter(prefix="/api/sessions/{sid}/documents", tags=["documents"])

ALLOWED_SUFFIXES = (".pdf", ".txt", ".md", ".markdown")
_COPY_CHUNK = 1024 * 1024
_PDF_MAGIC = b"%PDF-"
# Multipart framing overhead allowed on top of the file bytes themselves.
_MULTIPART_SLACK_BYTES = 64 * 1024
_ALREADY_LOADED = "already loaded in this session"

_ingest_deps = [Depends(owned_session), Depends(rate_limited("documents")), Depends(within_budget)]


class UrlRequest(BaseModel):
    urls: list[Annotated[str, Field(max_length=MAX_URL_CHARS)]] = Field(
        min_length=1, max_length=MAX_URLS_PER_REQUEST
    )


class ArxivRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_ARXIV_QUERY_CHARS)


class _TooLarge(Exception):
    pass


def _user_error(exc: Exception, context: str) -> str:
    """Message safe to show the uploader. Loader validation errors are ours and
    descriptive; anything else (Qdrant, OpenAI, parser internals) is logged
    with an id and replaced by a generic message."""

    if isinstance(exc, (ValueError, UnsafeURLError, TimeoutError)):
        return str(exc)[:300]
    error_id = uuid.uuid4().hex[:12]
    logger.error("ingest_failed context=%s error_id=%s", context, error_id, exc_info=exc)
    return f"Could not load this document (error id {error_id})"


def _copy_upload(upload: UploadFile, dest, max_bytes: int, is_pdf: bool) -> int:
    size = 0
    first = True
    while chunk := upload.file.read(_COPY_CHUNK):
        if first and is_pdf and _PDF_MAGIC not in chunk[:1024]:
            raise ValueError("Not a valid PDF file")
        first = False
        size += len(chunk)
        if size > max_bytes:
            raise ValueError(f"File exceeds the {max_bytes // (1024 * 1024)} MB limit")
        dest.write(chunk)
    if size == 0:
        raise ValueError("File is empty")
    return size


@router.get("", dependencies=[Depends(owned_session)])
def list_docs(sid: str) -> dict:
    return {"documents": list_papers(sid)}


@router.post("", dependencies=_ingest_deps)
def upload(
    sid: str,
    files: list[UploadFile] = File(...),
    settings: Settings = Depends(get_settings),
) -> dict:
    if len(files) > settings.max_files_per_request:
        raise ApiError(
            422,
            "too_many_files",
            f"Upload at most {settings.max_files_per_request} files at a time",
        )
    existing = set(list_papers(sid))
    added: list[str] = []
    skipped: list[dict] = []
    failed: list[dict] = []
    for upload_file in files:
        # The browser-supplied name is only used for display, the title and the
        # suffix; the bytes go to a server-named temp file.
        name = PurePath((upload_file.filename or "document").replace("\\", "/")).name or "document"
        title = Path(name).stem or "document"
        suffix = Path(name).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            failed.append({"name": name, "error": f"Unsupported file type: {suffix or 'none'}"})
            continue
        if title in existing:
            skipped.append({"name": name, "reason": _ALREADY_LOADED})
            continue
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp_path = tmp.name
                _copy_upload(upload_file, tmp, settings.max_upload_bytes, suffix == ".pdf")
            docs = load_document(tmp_path)
            if not docs:
                raise ValueError("No text could be extracted")
            for doc in docs:
                doc.metadata["title"] = title
                doc.metadata["source"] = name
                doc.metadata.pop("file_path", None)
            add_paper(docs, sid)
            added.append(name)
            existing.add(title)
        except Exception as exc:
            failed.append({"name": name, "error": _user_error(exc, "upload")})
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)
    return {"added": added, "skipped": skipped, "failed": failed}


@router.post("/url", dependencies=_ingest_deps)
def load_urls(sid: str, body: UrlRequest) -> dict:
    existing = set(list_papers(sid))
    loaded: list[str] = []
    skipped: list[dict] = []
    failed: list[dict] = []
    for url in dict.fromkeys(u.strip() for u in body.urls if u.strip()):
        try:
            docs = load_webpage(url)
            if not docs:
                raise ValueError("No text could be extracted")
            title = docs[0].metadata.get("title") or url
            if title in existing:
                skipped.append({"url": url, "name": title, "reason": _ALREADY_LOADED})
                continue
            add_paper(docs, sid)
            loaded.append(title)
            existing.add(title)
        except Exception as exc:
            failed.append({"url": url, "error": _user_error(exc, "url")})
    return {"loaded": loaded, "skipped": skipped, "failed": failed}


@router.post("/arxiv", dependencies=_ingest_deps)
def load_arxiv_paper(sid: str, body: ArxivRequest) -> dict:
    try:
        docs = load_arxiv(body.query.strip())
        if not docs:
            raise ValueError(f"No ArXiv paper found for: {body.query.strip()}")
        title = docs[0].metadata.get("title") or body.query
        # Idempotent: a paper already in the session is not embedded twice.
        if title not in set(list_papers(sid)):
            add_paper(docs, sid)
        return {"loaded": title}
    except Exception as exc:
        raise ApiError(400, "load_failed", _user_error(exc, "arxiv"))


# ── Streaming body-size cap for uploads ──────────────────────────────────────

class UploadSizeLimit:
    """Pure ASGI middleware: cap the multipart body of document uploads.

    Starlette spools file parts to disk as they arrive, which bounds memory but
    not disk use. This rejects a declared ``Content-Length`` over the cap up
    front and counts bytes as they stream in (chunked uploads), answering 413
    as soon as the cap is crossed.
    """

    def __init__(self, app, settings: Settings) -> None:
        self.app = app
        self.limit = (
            settings.max_files_per_request * settings.max_upload_bytes + _MULTIPART_SLACK_BYTES
        )

    @staticmethod
    def _applies(scope) -> bool:
        path = scope.get("path", "")
        return (
            scope["type"] == "http"
            and scope.get("method") == "POST"
            and path.startswith("/api/sessions/")
            and path.rstrip("/").endswith("/documents")
        )

    async def _reject(self, send) -> None:
        body = json.dumps(
            {"detail": "Upload is too large", "code": "payload_too_large"}
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"connection", b"close"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send) -> None:
        if not self._applies(scope):
            await self.app(scope, receive, send)
            return
        declared = dict(scope.get("headers") or []).get(b"content-length", b"")
        if declared.isdigit() and int(declared) > self.limit:
            await self._reject(send)
            return

        received = 0
        exceeded = False
        started = False

        async def limited_receive():
            nonlocal received, exceeded
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.limit:
                    exceeded = True
                    raise _TooLarge()
            return message

        async def guarded_send(message):
            nonlocal started
            if exceeded:
                return  # drop the app's own error response; a 413 follows
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except Exception:
            if not exceeded:
                raise
        if exceeded and not started:
            await self._reject(send)
