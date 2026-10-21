"""retrieve() honours the agent-chosen k on both paths; Qdrant and the reranker are mocked."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

import backend.vector_store as vector_store


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> dict:
    record: dict = {"search_k": [], "rerank_top_n": []}

    def fake_search(query: str, session_id: str, k: int = 4) -> list[Document]:
        record["search_k"].append(k)
        return [Document(page_content=f"chunk {i}") for i in range(k)]

    def fake_rerank(query: str, docs: list[Document], top_n: int) -> list[Document]:
        record["rerank_top_n"].append(top_n)
        return list(reversed(docs))[:top_n]

    monkeypatch.setattr(vector_store, "search", fake_search)
    monkeypatch.setattr(vector_store, "rerank", fake_rerank)
    return record


@pytest.mark.parametrize("k", [1, 4, 7, 10])
def test_rerank_path_keeps_k(monkeypatch: pytest.MonkeyPatch, calls: dict, k: int) -> None:
    monkeypatch.setattr(vector_store, "RERANK_ENABLED", True)
    monkeypatch.setattr(vector_store, "RETRIEVAL_CANDIDATE_K", 20)

    docs = vector_store.retrieve("q", "s", k=k)

    assert len(docs) == k
    assert calls["search_k"] == [20]
    assert calls["rerank_top_n"] == [k]


def test_rerank_pool_is_at_least_k(monkeypatch: pytest.MonkeyPatch, calls: dict) -> None:
    monkeypatch.setattr(vector_store, "RERANK_ENABLED", True)
    monkeypatch.setattr(vector_store, "RETRIEVAL_CANDIDATE_K", 5)

    assert len(vector_store.retrieve("q", "s", k=8)) == 8
    assert calls["search_k"] == [8]


def test_dense_path_unchanged(monkeypatch: pytest.MonkeyPatch, calls: dict) -> None:
    monkeypatch.setattr(vector_store, "RERANK_ENABLED", False)

    assert len(vector_store.retrieve("q", "s", k=6)) == 6
    assert calls["search_k"] == [6]
    assert calls["rerank_top_n"] == []
