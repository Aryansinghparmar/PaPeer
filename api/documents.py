"""Document ingestion endpoints (file upload, web URL, ArXiv).

These are sync `def` handlers on purpose: FastAPI runs them in a thread pool, so
the blocking ingestion calls (`load_document`, `add_paper`, network I/O) don't
stall the event loop. They reuse the existing loaders and vector store unchanged.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from api.deps import get_current_user
from backend.paper_loader import load_arxiv, load_document, load_webpage
from backend.vector_store import add_paper, list_papers

router = APIRouter(prefix="/api/sessions/{sid}/documents", tags=["documents"])


class UrlRequest(BaseModel):
    urls: list[str]


class ArxivRequest(BaseModel):
    query: str


@router.get("")
def list_docs(sid: str, user: str = Depends(get_current_user)) -> dict:
    return {"documents": list_papers(sid)}


@router.post("")
def upload(
    sid: str,
    files: list[UploadFile] = File(...),
    user: str = Depends(get_current_user),
) -> dict:
    added: list[str] = []
    for f in files:
        suffix = Path(f.filename or "").suffix
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(f.file.read())
                tmp_path = tmp.name
            docs = load_document(tmp_path)
            for doc in docs:
                doc.metadata["title"] = Path(f.filename or "document").stem
            add_paper(docs, sid)
            added.append(f.filename or "document")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed: {f.filename} — {exc}")
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)
    return {"added": added}


@router.post("/url")
def load_urls(sid: str, body: UrlRequest, user: str = Depends(get_current_user)) -> dict:
    loaded: list[str] = []
    for url in [u.strip() for u in body.urls if u.strip()]:
        try:
            add_paper(load_webpage(url), sid)
            loaded.append(url)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed: {url} — {exc}")
    return {"loaded": loaded}


@router.post("/arxiv")
def load_arxiv_paper(sid: str, body: ArxivRequest, user: str = Depends(get_current_user)) -> dict:
    try:
        docs = load_arxiv(body.query.strip())
        add_paper(docs, sid)
        title = docs[0].metadata.get("title") if docs else body.query
        return {"loaded": title}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed: {exc}")
