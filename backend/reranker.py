"""Local cross-encoder reranking.

A cross-encoder scores each (query, chunk) pair jointly, which is far more
precise than the bi-encoder similarity used for the first-stage vector search.
Its job here is to push the genuinely relevant chunks to the top and demote
noise such as bibliography and citation entries — the exact failure mode the
baseline evaluation exposed (low Contextual Precision / Relevancy).

The model runs locally on CPU via fastembed (ONNX). It adds no OpenAI cost.
Loading is deferred and cached so the model is downloaded and initialised once.
"""

from __future__ import annotations

from langchain_core.documents import Document

from backend.config import RERANK_MODEL, RERANK_TOP_N
from backend.logging_config import get_logger

logger = get_logger(__name__)

_encoder = None


def _get_encoder():
    """Lazily construct and cache the cross-encoder (first call downloads it)."""

    global _encoder
    if _encoder is None:
        # Imported lazily so importing this module stays cheap and side-effect free.
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        logger.info("loading_reranker model=%s", RERANK_MODEL)
        _encoder = TextCrossEncoder(RERANK_MODEL)
    return _encoder


def rerank(
    query: str,
    docs: list[Document],
    top_n: int = RERANK_TOP_N,
) -> list[Document]:
    """Return the ``top_n`` documents most relevant to ``query``.

    The relevance score is written to ``metadata['rerank_score']`` for
    transparency. The input list is not mutated. When there is nothing to
    reorder (0 or 1 docs, or top_n covers everything) the encoder is still used
    only if it adds value; trivial cases short-circuit to avoid loading it.
    """

    if not docs:
        return []
    if top_n < 1:
        return []

    encoder = _get_encoder()
    scores = list(encoder.rerank(query, [doc.page_content for doc in docs]))

    ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
    top: list[Document] = []
    for doc, score in ranked[:top_n]:
        # Copy so the persisted retrieved_docs carry the score without mutating
        # the caller's objects.
        top.append(
            Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "rerank_score": float(score)},
            )
        )
    logger.info(
        "reranked candidates=%d kept=%d top_score=%.4f",
        len(docs),
        len(top),
        top[0].metadata["rerank_score"] if top else float("nan"),
    )
    return top
