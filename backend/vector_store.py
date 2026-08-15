import os

from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Modifier, SparseVectorParams, VectorParams

from backend.config import (
    EMBEDDING_CACHE_DIR,
    EMBEDDING_DIM,
    OPENAI_EMBEDDING_MODEL,
    RERANK_ENABLED,
    RERANK_TOP_N,
    RETRIEVAL_CANDIDATE_K,
    RETRIEVAL_MODE,
    SPARSE_EMBEDDING_MODEL,
)
from backend.logging_config import get_logger
from backend.reranker import rerank
from backend.telemetry import monotonic_seconds, record_ingestion
from backend.tracing import (
    traced_add_documents,
    traced_delete_collection,
    traced_rerank,
    traced_similarity_search,
)

logger = get_logger(__name__)

# langchain-qdrant's default names: unnamed dense vector ("") + this sparse name.
SPARSE_VECTOR_NAME = "langchain-sparse"
HYBRID = RETRIEVAL_MODE == "hybrid"

# ── Config ───────────────────────────────────────────────────────────────────


# ── Singletons ────────────────────────────────────────────────────────────────

base_embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
embedding_file_store = LocalFileStore(str(EMBEDDING_CACHE_DIR))
embeddings = CacheBackedEmbeddings.from_bytes_store(
    base_embeddings,
    embedding_file_store,
    namespace=base_embeddings.model,
    query_embedding_cache=True,
    key_encoder="blake2b",
)

qdrant_client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
    timeout=120,
)

# Sparse BM25 embedding is loaded lazily so dense-only runs never pay for it.
_sparse_embeddings: FastEmbedSparse | None = None


def _get_sparse_embeddings() -> FastEmbedSparse:
    global _sparse_embeddings
    if _sparse_embeddings is None:
        logger.info("loading_sparse_embeddings model=%s", SPARSE_EMBEDDING_MODEL)
        _sparse_embeddings = FastEmbedSparse(model_name=SPARSE_EMBEDDING_MODEL)
    return _sparse_embeddings


# ── Collection ───────────────────────────────────────────────────────────────

def get_collection_name(session_id: str) -> str:
    return f"papeer_{session_id.replace('-', '_')}"


def get_vectorstore(session_id: str) -> QdrantVectorStore:
    """Return the vector store for a session, creating the collection if needed.

    In hybrid mode the collection carries both a dense vector and a BM25 sparse
    vector (IDF modifier), and the store fuses them at query time. In dense mode
    the behaviour is unchanged. Validation is disabled because we own the schema;
    this avoids an extra probe embedding call per construction.
    """

    collection_name = get_collection_name(session_id)
    if not qdrant_client.collection_exists(collection_name):
        create_kwargs = {
            "collection_name": collection_name,
            "vectors_config": VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        }
        if HYBRID:
            create_kwargs["sparse_vectors_config"] = {
                SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF)
            }
        qdrant_client.create_collection(**create_kwargs)

    if HYBRID:
        return QdrantVectorStore(
            client=qdrant_client,
            collection_name=collection_name,
            embedding=embeddings,
            sparse_embedding=_get_sparse_embeddings(),
            retrieval_mode=RetrievalMode.HYBRID,
            sparse_vector_name=SPARSE_VECTOR_NAME,
            validate_embeddings=False,
            validate_collection_config=False,
        )
    return QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embeddings,
        validate_embeddings=False,
        validate_collection_config=False,
    )


# ── Public API ───────────────────────────────────────────────────────────────

def add_paper(docs: list[Document], session_id: str) -> None:
    logger.info("adding_documents session_hash=%s document_count=%d", hash(session_id), len(docs))
    started = monotonic_seconds()
    cache_before = (
        sum(1 for path in EMBEDDING_CACHE_DIR.rglob("*") if path.is_file())
        if EMBEDDING_CACHE_DIR.exists()
        else 0
    )
    traced_add_documents(docs, session_id, get_vectorstore(session_id))
    cache_after = (
        sum(1 for path in EMBEDDING_CACHE_DIR.rglob("*") if path.is_file())
        if EMBEDDING_CACHE_DIR.exists()
        else 0
    )
    record_ingestion(
        source="vector_store.add_paper",
        session_id=session_id,
        docs=docs,
        latency_seconds=monotonic_seconds() - started,
        embedding_model=OPENAI_EMBEDDING_MODEL,
        cache_entries_before=cache_before,
        cache_entries_after=cache_after,
    )


def list_papers(session_id: str) -> list[str]:
    collection_name = get_collection_name(session_id)
    if not qdrant_client.collection_exists(collection_name):
        return []
    seen: set[str] = set()
    titles: list[str] = []
    offset = None
    while True:
        points, offset = qdrant_client.scroll(
            collection_name=collection_name,
            with_payload=True,
            limit=100,
            offset=offset,
        )
        for point in points:
            title = (point.payload or {}).get("metadata", {}).get("title")
            if title and title not in seen:
                seen.add(title)
                titles.append(title)
        if offset is None:
            break
    return titles


def search(query: str, session_id: str, k: int = 4) -> list[Document]:
    """First-stage retrieval: dense or hybrid (dense+BM25) similarity search."""

    docs = traced_similarity_search(query, session_id, k, get_vectorstore(session_id))
    logger.info(
        "vector_search session_hash=%s mode=%s requested_k=%d returned=%d",
        hash(session_id),
        RETRIEVAL_MODE,
        k,
        len(docs),
    )
    return docs


def retrieve(query: str, session_id: str, k: int = 4) -> list[Document]:
    """Two-stage retrieval used by the RAG graph.

    Improved path (RERANK_ENABLED): fetch a larger candidate pool, then a local
    cross-encoder reranks and keeps the top RERANK_TOP_N — this is what lifts
    precision/relevancy by discarding citation and off-topic noise. Baseline
    path (rerank off): the original single-stage dense search honouring ``k``.
    """

    if RERANK_ENABLED:
        pool = search(query, session_id, RETRIEVAL_CANDIDATE_K)
        return traced_rerank(query, pool, RERANK_TOP_N, rerank)
    return search(query, session_id, k)


def delete_session_collection(session_id: str) -> bool:
    """Delete one session collection and return whether it existed."""

    collection_name = get_collection_name(session_id)
    if not qdrant_client.collection_exists(collection_name):
        return False
    traced_delete_collection(session_id, collection_name, qdrant_client)
    logger.info("deleted_collection session_hash=%s", hash(session_id))
    return True
