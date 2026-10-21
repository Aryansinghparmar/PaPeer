"""Local, secret-free run measurements.

LangSmith provides detailed distributed traces when enabled. This module adds
a small local ledger for latency, route, retrieval behaviour, and estimated
token cost. It never stores prompts, answers, documents, or API keys.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
import tiktoken

from backend.config import (
    LOCAL_METRICS_ENABLED,
    METRICS_DIR,
    estimate_cost_usd,
)
from backend.logging_config import get_logger

logger = get_logger(__name__)


def monotonic_seconds() -> float:
    return time.perf_counter()


# ── Daily spend (feeds the API's DAILY_BUDGET_USD cap) ───────────────────────
# A running total of estimated_cost_usd for the current UTC day, kept in
# memory and updated by every record_run / record_ingestion. At startup the API
# seeds it from the ledger so a restart does not reset the day's spend. It is
# per process, like the rest of the API's in-memory state.

_spend_lock = threading.Lock()
_spend_day: date | None = None
_spend_usd = 0.0


def _utc_today() -> date:
    return datetime.now(timezone.utc).date()


def add_spend(cost_usd: float | None) -> None:
    global _spend_day, _spend_usd
    today = _utc_today()
    with _spend_lock:
        if _spend_day != today:
            _spend_day, _spend_usd = today, 0.0
        _spend_usd += float(cost_usd or 0.0)


def spent_today_usd() -> float:
    with _spend_lock:
        return _spend_usd if _spend_day == _utc_today() else 0.0


def load_today_spend(ledger_path: Path | None = None) -> float:
    """Reset the running total to today's (UTC) sum from the JSONL ledger."""

    global _spend_day, _spend_usd
    path = ledger_path or (METRICS_DIR / "runs.jsonl")
    today = _utc_today()
    total = 0.0
    try:
        with path.open(encoding="utf-8") as file:
            for line in file:
                try:
                    record = json.loads(line)
                    stamp = datetime.fromisoformat(record["timestamp"])
                    if stamp.tzinfo is None:
                        stamp = stamp.replace(tzinfo=timezone.utc)
                    if stamp.astimezone(timezone.utc).date() == today:
                        total += float(record.get("estimated_cost_usd") or 0.0)
                except (ValueError, KeyError, TypeError, AttributeError):
                    continue
    except FileNotFoundError:
        pass
    with _spend_lock:
        _spend_day, _spend_usd = today, total
    return total


def _usage_from_message(message: Any) -> tuple[int, int]:
    usage = getattr(message, "usage_metadata", None) or {}
    if usage:
        return int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))

    metadata = getattr(message, "response_metadata", None) or {}
    usage = metadata.get("token_usage") or metadata.get("usage") or {}
    return int(usage.get("prompt_tokens", usage.get("input_tokens", 0))), int(
        usage.get("completion_tokens", usage.get("output_tokens", 0))
    )


def collect_usage(messages: list[Any]) -> dict[str, int | bool]:
    input_tokens = 0
    output_tokens = 0
    usage_available = False
    for message in messages:
        if getattr(message, "usage_metadata", None) or (getattr(message, "response_metadata", None) or {}).get("token_usage"):
            usage_available = True
        current_input, current_output = _usage_from_message(message)
        input_tokens += current_input
        output_tokens += current_output
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "usage_available": usage_available,
    }


def collect_callback_usage(usage_by_model: dict[str, Any]) -> dict[str, int | bool]:
    input_tokens = 0
    output_tokens = 0
    for usage in usage_by_model.values():
        input_tokens += int(usage.get("input_tokens", 0))
        output_tokens += int(usage.get("output_tokens", 0))
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "usage_available": bool(usage_by_model),
    }


def estimate_document_tokens(docs: list[Document], model: str) -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return sum(len(encoding.encode(doc.page_content)) for doc in docs)


def record_ingestion(
    *,
    source: str,
    session_id: str,
    docs: list[Document],
    latency_seconds: float,
    embedding_model: str,
    cache_entries_before: int | None = None,
    cache_entries_after: int | None = None,
) -> dict[str, Any]:
    input_tokens = estimate_document_tokens(docs, embedding_model)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "phase": "ingestion",
        "session_id_hash": str(hash(session_id)),
        "document_count": len(docs),
        "input_tokens_estimated": input_tokens,
        "latency_seconds": round(latency_seconds, 4),
        "embedding_model": embedding_model,
        "cache_entries_before": cache_entries_before,
        "cache_entries_after": cache_entries_after,
        "estimated_cost_usd": round(
            estimate_cost_usd(embedding_model, input_tokens, 0), 8
        ),
    }
    add_spend(record["estimated_cost_usd"])
    if LOCAL_METRICS_ENABLED:
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        with (METRICS_DIR / "runs.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(
        "document_ingestion source=%s session_hash=%s chunks=%d estimated_tokens=%d latency_seconds=%.4f estimated_cost_usd=%.8f",
        source,
        record["session_id_hash"],
        record["document_count"],
        record["input_tokens_estimated"],
        record["latency_seconds"],
        record["estimated_cost_usd"],
    )
    return record


def record_run(
    *,
    source: str,
    session_id: str,
    state: dict[str, Any],
    latency_seconds: float,
    model_names: list[str],
    usage_by_model: dict[str, Any] | None = None,
) -> dict[str, Any]:
    messages = state.get("messages") or []
    usage = collect_callback_usage(usage_by_model) if usage_by_model else collect_usage(messages)
    if usage_by_model:
        model_names = sorted(usage_by_model)
    estimated_cost = None
    if usage_by_model:
        # Price each model's own tokens; pricing the summed tokens once per
        # model would over-count whenever a run uses more than one model.
        estimated_cost = sum(
            estimate_cost_usd(
                model,
                int(model_usage.get("input_tokens", 0)),
                int(model_usage.get("output_tokens", 0)),
            )
            for model, model_usage in usage_by_model.items()
        )
    elif usage["usage_available"]:
        estimated_cost = sum(
            estimate_cost_usd(model, usage["input_tokens"], usage["output_tokens"])
            for model in model_names
        )
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "session_id_hash": str(hash(session_id)),
        "route": state.get("route"),
        "retrieval_attempts": state.get("retrieval_attempts", 0),
        "rewrite_count": state.get("rewrite_count", 0),
        "retrieved_document_count": len(state.get("retrieved_docs") or []),
        "latency_seconds": round(latency_seconds, 4),
        "models": model_names,
        "model_usage": usage_by_model or {},
        **usage,
        "estimated_cost_usd": round(estimated_cost, 8) if estimated_cost is not None else None,
    }
    add_spend(record["estimated_cost_usd"])
    if LOCAL_METRICS_ENABLED:
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        with (METRICS_DIR / "runs.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(
        "rag_run source=%s session_hash=%s route=%s latency_seconds=%.4f documents=%d estimated_cost_usd=%s",
        source,
        record["session_id_hash"],
        record["route"],
        record["latency_seconds"],
        record["retrieved_document_count"],
        record["estimated_cost_usd"],
    )
    return record
