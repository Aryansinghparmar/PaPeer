"""Custom LangSmith spans with secret- and content-safe fields."""

from __future__ import annotations

from typing import Any

from langsmith import traceable
from tavily import TavilyClient


def _safe_inputs(values: dict[str, Any]) -> dict[str, Any]:
    query = values.get("query", values.get("optimized_query", ""))
    session_id = values.get("session_id", "")
    docs = values.get("docs")
    return {
        "query_length": len(str(query)),
        "session_hash": hash(str(session_id)) if session_id else None,
        "k": values.get("k"),
        "max_results": values.get("max_results"),
        "search_depth": values.get("search_depth"),
        "top_n": values.get("top_n"),
        "candidate_count": len(docs) if isinstance(docs, list) else None,
    }


def _safe_outputs(value: Any) -> dict[str, Any]:
    if isinstance(value, list):
        return {"document_count": len(value)}
    if isinstance(value, dict):
        return {"result_count": len(value.get("results", []))}
    return {"result": "completed" if value is not None else "empty"}


@traceable(
    name="qdrant_add_documents",
    run_type="retriever",
    process_inputs=_safe_inputs,
    process_outputs=_safe_outputs,
)
def traced_add_documents(docs: list[Any], session_id: str, vectorstore: Any) -> None:
    vectorstore.add_documents(docs)


@traceable(
    name="qdrant_similarity_search",
    run_type="retriever",
    process_inputs=_safe_inputs,
    process_outputs=_safe_outputs,
)
def traced_similarity_search(
    query: str, session_id: str, k: int, vectorstore: Any
) -> list[Any]:
    return vectorstore.similarity_search(query, k=k)


@traceable(
    name="qdrant_delete_collection",
    run_type="retriever",
    process_inputs=_safe_inputs,
    process_outputs=_safe_outputs,
)
def traced_delete_collection(session_id: str, collection_name: str, client: Any) -> None:
    client.delete_collection(collection_name=collection_name)


@traceable(
    name="cross_encoder_rerank",
    run_type="retriever",
    process_inputs=_safe_inputs,
    process_outputs=_safe_outputs,
)
def traced_rerank(query: str, docs: list[Any], top_n: int, rerank_fn: Any) -> list[Any]:
    """Trace the local cross-encoder rerank step without logging chunk text."""
    return rerank_fn(query, docs, top_n)


@traceable(
    name="tavily_search",
    run_type="retriever",
    process_inputs=_safe_inputs,
    process_outputs=_safe_outputs,
)
def traced_tavily_search(
    api_key: str,
    query: str,
    max_results: int,
    search_depth: str,
) -> dict[str, Any]:
    return TavilyClient(api_key=api_key).search(
        query,
        max_results=max_results,
        search_depth=search_depth,
    )
