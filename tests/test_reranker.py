"""Offline tests for the local cross-encoder reranker.

These use the real fastembed ONNX model, which runs locally on CPU. They make
NO OpenAI calls and cost nothing. The first run downloads the model (~80 MB).
"""

from langchain_core.documents import Document

from backend.reranker import rerank


def _doc(text: str) -> Document:
    return Document(page_content=text, metadata={"title": "t"})


def test_rerank_promotes_relevant_and_demotes_citation_noise() -> None:
    query = "What memory types does Moltbot use across sessions?"
    docs = [
        _doc("DigitalOcean. (2026, January 27). What is Moltbot? Retrieved from example.com"),
        _doc("Moltbot maintains long-term memory across sessions using local file storage."),
        _doc("Moltbot operates with modest hardware demands and low CPU usage."),
    ]

    result = rerank(query, docs, top_n=2)

    assert len(result) == 2
    # The memory sentence must outrank the bibliography/citation entry.
    assert "long-term memory across sessions" in result[0].page_content
    contents = [d.page_content for d in result]
    assert not any("DigitalOcean" in c for c in contents)


def test_rerank_writes_score_and_preserves_metadata() -> None:
    docs = [_doc("alpha beta"), _doc("gamma delta")]
    result = rerank("alpha", docs, top_n=2)

    assert all("rerank_score" in d.metadata for d in result)
    assert all(d.metadata["title"] == "t" for d in result)
    # Scores must be sorted descending.
    assert result[0].metadata["rerank_score"] >= result[1].metadata["rerank_score"]


def test_rerank_handles_empty_and_top_n_larger_than_pool() -> None:
    assert rerank("q", []) == []
    docs = [_doc("only one chunk here")]
    result = rerank("chunk", docs, top_n=5)
    assert len(result) == 1
