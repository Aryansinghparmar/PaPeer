"""Central configuration for the application and evaluation runs.

The defaults are selected for a cost-controlled capstone project. Every value
can be changed with an environment variable without changing source code.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def env_str(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    return value or default


def env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return max(minimum, min(value, maximum))


OPENAI_CHAT_MODEL = env_str("OPENAI_CHAT_MODEL", "gpt-5-mini")
OPENAI_RENAME_MODEL = env_str("OPENAI_RENAME_MODEL", OPENAI_CHAT_MODEL)
OPENAI_EVAL_MODEL = env_str("OPENAI_EVAL_MODEL", OPENAI_CHAT_MODEL)
OPENAI_EMBEDDING_MODEL = env_str(
    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
)

# The dimension must match the selected embedding model and the Qdrant
# collection. A changed dimension requires a new collection or re-indexing.
EMBEDDING_DIM = env_int("OPENAI_EMBEDDING_DIM", 1536, 1, 3072)

TAVILY_SEARCH_DEPTH = env_str("TAVILY_SEARCH_DEPTH", "basic")
TAVILY_MAX_RESULTS = env_int("TAVILY_MAX_RESULTS", 3, 1, 10)

MAX_RETRIEVAL_ATTEMPTS = env_int("MAX_RETRIEVAL_ATTEMPTS", 3, 1, 5)
MAX_QUERY_REWRITES = env_int("MAX_QUERY_REWRITES", 1, 0, 3)

# ── Retrieval strategy ────────────────────────────────────────────────────────
# All retrieval upgrades below (hybrid sparse vectors + cross-encoder reranking)
# run locally on CPU via fastembed. They add NO OpenAI cost per query. They are
# toggle-able so the evaluation harness can run a controlled A/B:
#   baseline  -> RETRIEVAL_MODE=dense,  RERANK_ENABLED=false  (original behaviour)
#   improved  -> RETRIEVAL_MODE=hybrid, RERANK_ENABLED=true   (dense+BM25 → rerank)
RETRIEVAL_MODE = env_str("RETRIEVAL_MODE", "hybrid").lower()  # "dense" | "hybrid"
RERANK_ENABLED = env_str("RERANK_ENABLED", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
# Size of the candidate pool fetched before reranking. A larger pool gives the
# cross-encoder more to choose from; the final answer uses only RERANK_TOP_N.
RETRIEVAL_CANDIDATE_K = env_int("RETRIEVAL_CANDIDATE_K", 20, 1, 100)
# Number of chunks kept after reranking (or returned directly when rerank is off).
RERANK_TOP_N = env_int("RERANK_TOP_N", 4, 1, 20)
SPARSE_EMBEDDING_MODEL = env_str("SPARSE_EMBEDDING_MODEL", "Qdrant/bm25")
RERANK_MODEL = env_str("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")

EMBEDDING_CACHE_DIR = Path(env_str("EMBEDDING_CACHE_DIR", "embedding_cache"))
CHECKPOINT_DB_PATH = env_str("CHECKPOINT_DB_PATH", "checkpoints.db")
METRICS_DIR = Path(env_str("METRICS_DIR", "observability"))
LOCAL_METRICS_ENABLED = env_str("LOCAL_METRICS_ENABLED", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
LOG_LEVEL = env_str("LOG_LEVEL", "INFO").upper()


# Current public list prices used only for local estimates. They are not a
# billing guarantee. The estimates are useful for comparing experiments.
MODEL_PRICING_USD_PER_MILLION = {
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "text-embedding-3-small": {"input": 0.02, "output": 0.00},
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = MODEL_PRICING_USD_PER_MILLION.get(model)
    if prices is None:
        for alias, alias_prices in MODEL_PRICING_USD_PER_MILLION.items():
            if model.startswith(f"{alias}-"):
                prices = alias_prices
                break
    if not prices:
        return 0.0
    return (
        input_tokens * prices["input"] + output_tokens * prices["output"]
    ) / 1_000_000
