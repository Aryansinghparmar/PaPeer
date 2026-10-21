"""Thin wrapper over the ``arxiv`` package for the claim-verification engine.

One module-level client is shared so the library's 3-second delay between
requests (arXiv API etiquette) holds across calls, and a lock keeps concurrent
requests from different sessions from bypassing it.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from datetime import date

import arxiv
import requests
from requests.adapters import HTTPAdapter

from backend.config import env_int
from backend.logging_config import get_logger

logger = get_logger(__name__)

ARXIV_HTTP_TIMEOUT_SECONDS = env_int("ARXIV_HTTP_TIMEOUT_SECONDS", 10, 2, 60)
ARXIV_NUM_RETRIES = env_int("ARXIV_NUM_RETRIES", 1, 0, 3)
ARXIV_PAGE_SIZE = env_int("ARXIV_PAGE_SIZE", 10, 1, 50)
# arXiv asks clients for no more than one request every three seconds.
ARXIV_DELAY_SECONDS = 3.0
SUMMARY_MAX_CHARS = 600

_VERSION_SUFFIX = re.compile(r"v\d+$")


@dataclass(frozen=True)
class ArxivHit:
    arxiv_id: str
    title: str
    url: str
    summary: str
    published: date | None
    updated: date | None
    primary_category: str
    authors: tuple[str, ...]


class _TimeoutHTTPAdapter(HTTPAdapter):
    """Applies a default timeout to every request sent through the session."""

    def __init__(self, timeout: float, **kwargs) -> None:
        self.timeout = timeout
        super().__init__(**kwargs)

    def send(self, request, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = self.timeout
        return super().send(request, **kwargs)


def _build_client() -> arxiv.Client:
    client = arxiv.Client(
        page_size=ARXIV_PAGE_SIZE,
        delay_seconds=ARXIV_DELAY_SECONDS,
        num_retries=ARXIV_NUM_RETRIES,
    )
    # arxiv 3.0.0 calls session.get() without a timeout, so a stalled socket
    # would block forever. The client exposes no option, so mount an adapter
    # on its (private) session that injects one.
    session = getattr(client, "_session", None)
    if isinstance(session, requests.Session):
        adapter = _TimeoutHTTPAdapter(ARXIV_HTTP_TIMEOUT_SECONDS)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
    else:
        logger.warning("arxiv_client_timeout_not_applied")
    return client


_client = _build_client()
_lock = threading.Lock()
# Worst case one search waits for every retry plus the delays between them.
_LOCK_WAIT_SECONDS = (ARXIV_HTTP_TIMEOUT_SECONDS + ARXIV_DELAY_SECONDS) * (ARXIV_NUM_RETRIES + 1)


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())


def _to_date(value) -> date | None:
    return value.date() if hasattr(value, "date") else None


def _to_hit(result) -> ArxivHit | None:
    entry_id = str(getattr(result, "entry_id", "") or "")
    if "/abs/" not in entry_id:
        return None
    arxiv_id = _VERSION_SUFFIX.sub("", entry_id.split("/abs/")[-1].strip())
    if not arxiv_id:
        return None
    summary = _clean(getattr(result, "summary", ""))
    if len(summary) > SUMMARY_MAX_CHARS:
        summary = summary[:SUMMARY_MAX_CHARS].rsplit(" ", 1)[0] + "..."
    return ArxivHit(
        arxiv_id=arxiv_id,
        title=_clean(getattr(result, "title", "")),
        url=f"https://arxiv.org/abs/{arxiv_id}",
        summary=summary,
        published=_to_date(getattr(result, "published", None)),
        updated=_to_date(getattr(result, "updated", None)),
        primary_category=str(getattr(result, "primary_category", "") or ""),
        authors=tuple(str(getattr(a, "name", a)) for a in (getattr(result, "authors", None) or [])[:3]),
    )


def search_arxiv(query: str, max_results: int) -> list[ArxivHit]:
    """Run one arXiv API search; raises on transport/API errors so callers can record them."""
    query = query.strip()
    if not query:
        return []
    search = arxiv.Search(query=query, max_results=max(1, min(max_results, ARXIV_PAGE_SIZE)))
    if not _lock.acquire(timeout=_LOCK_WAIT_SECONDS):
        raise TimeoutError("arxiv client busy")
    try:
        # results() is a lazy generator; consume it while holding the lock so
        # every HTTP request it makes is covered by the shared rate limit.
        results = list(_client.results(search))
    finally:
        _lock.release()
    hits = [hit for hit in (_to_hit(r) for r in results) if hit is not None]
    logger.info("arxiv_search results=%d parsed=%d", len(results), len(hits))
    return hits
