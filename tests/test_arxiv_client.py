"""Offline tests for the arXiv client wrapper. The arxiv.Client is faked; no network."""

from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from backend import arxiv_client


def _result(entry_id: str, **overrides) -> SimpleNamespace:
    fields = {
        "entry_id": entry_id,
        "title": "Attention Is\n  All You Need",
        "summary": "We propose   the Transformer.\n" * 3,
        "published": datetime(2017, 6, 12, tzinfo=timezone.utc),
        "updated": datetime(2023, 8, 2, tzinfo=timezone.utc),
        "primary_category": "cs.CL",
        "authors": [SimpleNamespace(name=n) for n in ("Vaswani", "Shazeer", "Parmar", "Uszkoreit")],
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


class _FakeClient:
    def __init__(self, results=None, error: Exception | None = None):
        self.results_list = results or []
        self.error = error
        self.searches = []

    def results(self, search):
        self.searches.append(search)
        if self.error:
            raise self.error
        return iter(self.results_list)


def test_search_parses_results_into_hits(monkeypatch) -> None:
    fake = _FakeClient([_result("http://arxiv.org/abs/1706.03762v7")])
    monkeypatch.setattr(arxiv_client, "_client", fake)

    hits = arxiv_client.search_arxiv('all:"attention" AND all:transformer', max_results=3)

    assert len(hits) == 1
    hit = hits[0]
    assert hit.arxiv_id == "1706.03762"
    assert hit.url == "https://arxiv.org/abs/1706.03762"
    assert hit.title == "Attention Is All You Need"
    assert "  " not in hit.summary and "\n" not in hit.summary
    assert hit.published == date(2017, 6, 12)
    assert hit.updated == date(2023, 8, 2)
    assert hit.primary_category == "cs.CL"
    assert hit.authors == ("Vaswani", "Shazeer", "Parmar")
    assert fake.searches[0].query == 'all:"attention" AND all:transformer'
    assert fake.searches[0].max_results == 3


def test_search_handles_old_style_ids_and_skips_malformed(monkeypatch) -> None:
    fake = _FakeClient([
        _result("http://arxiv.org/abs/hep-th/9901001v2"),
        _result("not-an-arxiv-url"),
    ])
    monkeypatch.setattr(arxiv_client, "_client", fake)

    hits = arxiv_client.search_arxiv("all:string", max_results=5)

    assert [h.arxiv_id for h in hits] == ["hep-th/9901001"]


def test_summary_is_trimmed(monkeypatch) -> None:
    fake = _FakeClient([_result("http://arxiv.org/abs/2401.00001v1", summary="word " * 1000)])
    monkeypatch.setattr(arxiv_client, "_client", fake)

    hit = arxiv_client.search_arxiv("all:word", max_results=1)[0]

    assert len(hit.summary) <= arxiv_client.SUMMARY_MAX_CHARS + 3


def test_max_results_capped_and_empty_query_skips_call(monkeypatch) -> None:
    fake = _FakeClient([])
    monkeypatch.setattr(arxiv_client, "_client", fake)

    assert arxiv_client.search_arxiv("   ", max_results=5) == []
    assert fake.searches == []

    arxiv_client.search_arxiv("all:x", max_results=10_000)
    assert fake.searches[0].max_results == arxiv_client.ARXIV_PAGE_SIZE


def test_errors_propagate_and_release_lock(monkeypatch) -> None:
    monkeypatch.setattr(arxiv_client, "_client", _FakeClient(error=ConnectionError("down")))

    with pytest.raises(ConnectionError):
        arxiv_client.search_arxiv("all:x", max_results=1)
    assert not arxiv_client._lock.locked()


def test_real_client_has_etiquette_settings_and_transport_timeout() -> None:
    client = arxiv_client._build_client()

    assert client.delay_seconds >= 3.0
    assert client.num_retries == arxiv_client.ARXIV_NUM_RETRIES
    adapter = client._session.get_adapter("https://export.arxiv.org/api/query")
    assert isinstance(adapter, arxiv_client._TimeoutHTTPAdapter)
    assert adapter.timeout == arxiv_client.ARXIV_HTTP_TIMEOUT_SECONDS


def test_timeout_adapter_injects_default_timeout(monkeypatch) -> None:
    seen = {}

    def fake_send(self, request, **kwargs):
        seen.update(kwargs)
        return "response"

    monkeypatch.setattr(arxiv_client.HTTPAdapter, "send", fake_send)
    adapter = arxiv_client._TimeoutHTTPAdapter(7)

    assert adapter.send(object(), timeout=None) == "response"
    assert seen["timeout"] == 7
