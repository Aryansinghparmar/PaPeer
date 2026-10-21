"""Cost-controlled, reproducible RAG evaluation (DeepEval LLM-as-judge).

Every number this script prints can be regenerated from the report it writes:
the report records the git SHA (and whether the tree was dirty), package
versions, app and judge models (alias and the resolved model id the API
returned), retrieval and loop settings, the goldens file hash, the metric list
and thresholds, and per case the route taken, tool calls by source, retrieved
and unique chunk counts, latency, measured app tokens and cost, and the judge
cost, score and reason for every metric.

    # plan + cost estimate, no API call, no Qdrant connection
    python evaluate.py --goldens all --metrics faithfulness,answer_relevancy --runs 3 --dry-run

    # the real run (hard spend cap; results/<label>_<UTC>.json + .md)
    python evaluate.py --goldens all --metrics faithfulness,answer_relevancy --runs 3 \
        --label rag_core_x3 --budget-usd 1.00

The retrieval strategy is read from config at import time, so run a retrieval
A/B as two separate processes (RETRIEVAL_MODE=dense RERANK_ENABLED=false vs the
hybrid+rerank default) and use ``--compare`` to write the before/after table.

Cost rules: app cost comes from the LangChain usage callback priced with
``backend.config.MODEL_PRICING_USD_PER_MILLION``; judge cost comes from
DeepEval's per-metric ``evaluation_cost`` cross-checked against a per-call
token ledger. A missing (None) or zero cost for a call that used tokens is an
error, never a silent $0. ``--budget-usd`` stops the run *before* a case whose
estimated cost would exceed the cap, and the report records the truncation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import shlex
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

# Only backend.config is imported at module level: it reads env vars and opens
# no client. Everything that talks to OpenAI, Qdrant or Tavily (backend.rag_graph,
# backend.vector_store, deepeval) is imported lazily so --dry-run stays offline.
import backend.config as cfg
from backend.config import (
    MODEL_PRICING_USD_PER_MILLION,
    OPENAI_CHAT_MODEL,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_EVAL_MODEL,
    estimate_cost_usd,
)

REPO_ROOT           = Path(__file__).resolve().parent
PDF_PATH            = "documents/Openclaw_Research_Report.pdf"
GOLDENS_FILE        = REPO_ROOT / "goldens.json"
CURATED_FILE        = REPO_ROOT / "goldens_curated.json"
RESULTS_DIR         = REPO_ROOT / "results"
MAX_CONTEXTS        = 5
GOLDENS_PER_CONTEXT = 2
METRIC_THRESHOLD    = 0.7
DEFAULT_BUDGET_USD  = 1.00
MAX_CONSECUTIVE_FAILURES = 3
REPORT_SCHEMA       = 2

# CLI key -> (DeepEval class name, display name DeepEval reports in results).
METRICS = {
    "faithfulness": ("FaithfulnessMetric", "Faithfulness"),
    "answer_relevancy": ("AnswerRelevancyMetric", "Answer Relevancy"),
    "contextual_precision": ("ContextualPrecisionMetric", "Contextual Precision"),
    "contextual_recall": ("ContextualRecallMetric", "Contextual Recall"),
    "contextual_relevancy": ("ContextualRelevancyMetric", "Contextual Relevancy"),
}
GOLDEN_SETS = {"synthetic": GOLDENS_FILE, "curated": CURATED_FILE}

VECTORSTORE_TOOL = "retrieve_from_vectorstore"
WEB_TOOL = "web_search"

# Capabilities DeepEval 4.0.0 does not know for models missing from its
# OPENAI_MODELS_DATA registry. gpt-6-luna only accepts the default temperature
# (supports_temperature=False makes DeepEval send temperature=1) and accepts
# response_format=json_schema, so the judge gets schema-enforced output.
JUDGE_CAPABILITIES = {
    "gpt-6-luna": {
        "supports_log_probs": False,
        "supports_multimodal": False,
        "supports_structured_outputs": True,
        "supports_json": False,
        "supports_temperature": False,
    },
}
# The judge binds no function tools, so it can reason even though the app's
# tool-calling steps cannot. Judging is multi-step claim checking, and in a
# probe the no-reasoning judge wrote incoherent rationales, so "low" is fixed
# here before any comparison run (recorded in every report).
DEFAULT_JUDGE_REASONING_EFFORT = "low"

# ── Planning-only cost model (dry run and the first case of the budget guard) ──
# Judge cost per case and metric measured with gpt-5-mini as judge on the
# curated set (provenance audit §6). The split of each measured pair (F+AR,
# CP+CR) between its two metrics is approximate; the sums match measurement.
# Other judges are scaled by their price ratio to gpt-5-mini, then multiplied by
# a safety factor for unknown token behaviour. Once a case has been measured,
# the guard uses measured costs only.
REFERENCE_JUDGE_MODEL = "gpt-5-mini"
REFERENCE_JUDGE_COST_PER_CASE = {
    "faithfulness": 0.0085,
    "answer_relevancy": 0.0051,
    "contextual_precision": 0.0087,
    "contextual_recall": 0.0087,
    "contextual_relevancy": 0.0495,
}
PLANNING_APP_TOKENS_PER_CASE = {"input": 4400, "output": 1800}
PLANNING_SAFETY_FACTOR = 2.0

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED      = 12345


class CostAccountingError(RuntimeError):
    """A call used tokens but its cost could not be measured (unpriced model)."""


# ── Provenance helpers ────────────────────────────────────────────────────────

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git_info() -> dict:
    """Commit SHA and whether tracked files differ from it (untracked ignored)."""

    def run(*cmd: str) -> str:
        return subprocess.run(
            ["git", *cmd], capture_output=True, text=True, timeout=15, cwd=REPO_ROOT
        ).stdout.strip()

    try:
        sha = run("rev-parse", "HEAD") or None
        changed = [line[3:] for line in run("status", "--porcelain", "--untracked-files=no").splitlines()]
        branch = run("rev-parse", "--abbrev-ref", "HEAD") or None
    except (OSError, subprocess.SubprocessError):
        return {"sha": None, "dirty": None, "branch": None, "changed_files": []}
    return {"sha": sha, "dirty": bool(changed), "branch": branch, "changed_files": changed}


def package_versions() -> dict:
    from importlib import metadata

    versions = {"python": platform.python_version()}
    for name in ("deepeval", "langchain-openai", "langchain-core", "langgraph", "openai", "qdrant-client", "fastembed"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def repo_relative(path) -> str:
    """Path relative to the repo (forward slashes) so reports never record local folders."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.name


def relative_argv(argv: list[str] | None = None) -> list[str]:
    argv = sys.argv if argv is None else argv
    return [repo_relative(a) if Path(a).is_absolute() else a for a in argv]


def command_line(argv: list[str] | None = None) -> str:
    return "python " + " ".join(shlex.quote(a) for a in relative_argv(argv))


def price_for(model: str) -> dict | None:
    """USD per 1M tokens, resolving dated snapshots (gpt-x-YYYY-MM-DD) to their alias."""
    prices = MODEL_PRICING_USD_PER_MILLION.get(model)
    if prices is None:
        for alias, alias_prices in MODEL_PRICING_USD_PER_MILLION.items():
            if model.startswith(f"{alias}-"):
                return alias_prices
    return prices


def settings_snapshot() -> dict:
    """Retrieval and loop settings that change what the pipeline does."""
    names = (
        "RETRIEVAL_MODE", "RERANK_ENABLED", "RETRIEVAL_CANDIDATE_K", "RERANK_TOP_N",
        "SPARSE_EMBEDDING_MODEL", "RERANK_MODEL", "MAX_RETRIEVAL_ATTEMPTS",
        "MAX_QUERY_REWRITES", "MAX_HISTORY_TURNS", "GRAPH_RECURSION_LIMIT",
        "OPENAI_REASONING_EFFORT", "ANSWER_REASONING_EFFORT", "LLM_MAX_OUTPUT_TOKENS",
        "LLM_TIMEOUT_SECONDS", "LLM_MAX_RETRIES", "TAVILY_MAX_RESULTS", "TAVILY_SEARCH_DEPTH",
    )
    snapshot = {name.lower(): getattr(cfg, name, None) for name in names}
    # Where the vector store ran (never the URL itself, which identifies the cluster).
    host = (urlparse(os.environ.get("QDRANT_URL", "")).hostname or "").lower()
    snapshot["qdrant_target"] = "local" if host in {"localhost", "127.0.0.1", "::1"} else "cloud"
    return snapshot


# ── Output files ──────────────────────────────────────────────────────────────

def default_output_path(label: str, stamp: str | None = None) -> Path:
    return RESULTS_DIR / f"{label}_{stamp or utc_stamp()}.json"


def ensure_writable(path: Path, overwrite: bool) -> None:
    """Refuse to clobber an existing result unless explicitly allowed."""
    for target in (Path(path), Path(path).with_suffix(".md")):
        if target.exists() and not overwrite:
            raise FileExistsError(
                f"{target} already exists; pass --overwrite or choose another --output/--label."
            )


def write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


# ── Goldens ───────────────────────────────────────────────────────────────────

def generate_goldens() -> list[dict]:
    """Paid: regenerate goldens.json with DeepEval's Synthesizer (explicit flag only)."""
    from deepeval.synthesizer import Synthesizer
    from deepeval.synthesizer.config import ContextConstructionConfig

    synthesizer = Synthesizer()
    goldens = synthesizer.generate_goldens_from_docs(
        document_paths=[str(REPO_ROOT / PDF_PATH)],
        include_expected_output=True,
        max_goldens_per_context=GOLDENS_PER_CONTEXT,
        context_construction_config=ContextConstructionConfig(
            max_contexts_per_document=MAX_CONTEXTS,
        ),
    )
    pairs = [
        {"input": g.input, "expected_output": g.expected_output}
        for g in goldens
        if g.input and g.expected_output
    ]
    GOLDENS_FILE.write_text(json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8")
    return pairs


def load_goldens(source: str) -> tuple[list[dict], list[dict]]:
    """Load cases from ``curated``, ``synthetic``, ``all`` (both) or a JSON path.

    Returns (cases, sources). Each case carries a stable ``case_id`` and the
    ``golden_set`` it came from; each source records path, SHA-256 and n.
    """
    if source == "all":
        names = ["synthetic", "curated"]
        files = [(name, GOLDEN_SETS[name]) for name in names]
    elif source in GOLDEN_SETS:
        files = [(source, GOLDEN_SETS[source])]
    else:
        path = Path(source)
        if not path.is_absolute() and not path.exists():
            path = REPO_ROOT / path
        files = [(path.stem, path)]

    cases: list[dict] = []
    sources: list[dict] = []
    for set_name, path in files:
        if not path.exists():
            raise FileNotFoundError(
                f"Goldens file {path} not found (use --generate-goldens to synthesise goldens.json; paid)."
            )
        pairs = json.loads(path.read_text(encoding="utf-8"))
        for index, pair in enumerate(pairs):
            if not pair.get("input") or not pair.get("expected_output"):
                raise ValueError(f"{path} entry {index} needs input and expected_output")
            cases.append({
                "case_id": f"{set_name}:{index}",
                "golden_set": set_name,
                "category": pair.get("category"),
                "input": pair["input"],
                "expected_output": pair["expected_output"],
            })
        try:
            shown = str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
        except ValueError:
            shown = str(path)
        sources.append({"set": set_name, "path": shown, "sha256": file_sha256(path), "n": len(pairs)})
    return cases, sources


def select_range(cases: list[dict], start: int, limit: int | None) -> list[dict]:
    if start < 0:
        raise ValueError("--start cannot be negative")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1")
        cases = cases[start : start + limit]
    else:
        cases = cases[start:]
    if not cases:
        raise ValueError("The selected evaluation range is empty")
    return cases


def parse_metrics(value: str) -> list[str]:
    names = [part.strip().lower() for part in value.split(",") if part.strip()]
    unknown = [name for name in names if name not in METRICS]
    if unknown or not names:
        raise argparse.ArgumentTypeError(
            f"unknown metric(s) {unknown}; choose from {', '.join(METRICS)}"
        )
    return list(dict.fromkeys(names))


# ── Cost planning and the budget guard ────────────────────────────────────────

def estimate_case_cost(metric_names: list[str], app_model: str, judge_model: str) -> dict:
    """Planning estimate for one case (app + judge), with the safety factor applied."""
    app_prices = price_for(app_model)
    app = None
    if app_prices:
        app = PLANNING_SAFETY_FACTOR * (
            PLANNING_APP_TOKENS_PER_CASE["input"] * app_prices["input"]
            + PLANNING_APP_TOKENS_PER_CASE["output"] * app_prices["output"]
        ) / 1_000_000

    judge_prices = price_for(judge_model)
    reference = price_for(REFERENCE_JUDGE_MODEL)
    judge = None
    scale = None
    if judge_prices and reference:
        scale = max(
            judge_prices["input"] / reference["input"],
            judge_prices["output"] / reference["output"],
        )
        judge = sum(
            REFERENCE_JUDGE_COST_PER_CASE[name] * scale * PLANNING_SAFETY_FACTOR
            for name in metric_names
        )
    total = None if app is None or judge is None else app + judge
    return {
        "app_usd": None if app is None else round(app, 6),
        "judge_usd": None if judge is None else round(judge, 6),
        "total_usd": None if total is None else round(total, 6),
        "judge_price_scale_vs_reference": None if scale is None else round(scale, 4),
        "safety_factor": PLANNING_SAFETY_FACTOR,
        "reference_judge_model": REFERENCE_JUDGE_MODEL,
    }


def estimate_plan(
    goldens: str = "all",
    metric_names: list[str] | None = None,
    runs: int = 1,
    start: int = 0,
    limit: int | None = None,
    app_model: str = OPENAI_CHAT_MODEL,
    judge_model: str = OPENAI_EVAL_MODEL,
) -> dict:
    """Offline plan for a RAG evaluation: case count and estimated spend."""
    metric_names = metric_names or list(METRICS)
    cases, sources = load_goldens(goldens)
    cases = select_range(cases, start, limit)
    per_case = estimate_case_cost(metric_names, app_model, judge_model)
    case_runs = len(cases) * runs
    total = None if per_case["total_usd"] is None else round(per_case["total_usd"] * case_runs, 4)
    return {
        "goldens": goldens,
        "sources": sources,
        "n_cases": len(cases),
        "runs": runs,
        "case_runs": case_runs,
        "metrics": metric_names,
        "app_model": app_model,
        "judge_model": judge_model,
        "per_case_estimate": per_case,
        "estimated_total_usd": total,
    }


class BudgetGuard:
    """Cumulative spend tracker that refuses work that would cross the cap.

    The next case is estimated from the most expensive case measured so far
    (the planning prior until the first case completes); the guard stops
    *before* starting a case whose estimate would exceed the budget, and checks
    again between the app pass and the judge pass of each case.
    """

    def __init__(self, budget_usd: float, prior_case_usd: float, prior_judge_usd: float):
        if budget_usd is None or budget_usd < 0:
            raise ValueError("--budget-usd must be a non-negative number")
        self.budget_usd = float(budget_usd)
        self.prior_case_usd = float(prior_case_usd)
        self.prior_judge_usd = float(prior_judge_usd)
        self.spent_usd = 0.0
        self.max_case_usd = 0.0
        self.max_judge_usd = 0.0
        self.cases_measured = 0
        self.truncated = False
        self.stop_reason: str | None = None

    def next_case_estimate(self) -> float:
        return self.max_case_usd if self.cases_measured else self.prior_case_usd

    def next_judge_estimate(self) -> float:
        return self.max_judge_usd if self.cases_measured else self.prior_judge_usd

    def can_start_case(self) -> bool:
        return self._check(self.next_case_estimate(), "next case")

    def can_start_judge(self) -> bool:
        return self._check(self.next_judge_estimate(), "judge pass")

    def _check(self, estimate: float, what: str) -> bool:
        if self.spent_usd + estimate <= self.budget_usd:
            return True
        self.truncated = True
        self.stop_reason = (
            f"stopped before {what}: spent ${self.spent_usd:.4f} + estimated ${estimate:.4f} "
            f"> budget ${self.budget_usd:.4f}"
        )
        return False

    def charge(self, amount_usd: float) -> None:
        self.spent_usd += float(amount_usd or 0.0)

    def record_case(self, case_usd: float, judge_usd: float) -> None:
        self.cases_measured += 1
        self.max_case_usd = max(self.max_case_usd, float(case_usd))
        self.max_judge_usd = max(self.max_judge_usd, float(judge_usd))

    def to_dict(self) -> dict:
        return {
            "budget_usd": self.budget_usd,
            "spent_usd": round(self.spent_usd, 6),
            "remaining_usd": round(self.budget_usd - self.spent_usd, 6),
            "prior_case_estimate_usd": round(self.prior_case_usd, 6),
            "max_measured_case_usd": round(self.max_case_usd, 6),
            "cases_measured": self.cases_measured,
            "truncated": self.truncated,
            "stop_reason": self.stop_reason,
        }


# ── Judge wiring (DeepEval) ───────────────────────────────────────────────────

class JudgeLedger:
    """Per-metric tally of every judge completion: tokens, cost, resolved model ids."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.by_metric: dict[str, dict] = {}

    def add(self, key: str, completion: Any, input_price: float, output_price: float) -> None:
        usage = getattr(completion, "usage", None)
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        output = int(getattr(usage, "completion_tokens", 0) or 0)
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        output_details = getattr(usage, "completion_tokens_details", None)
        cached = int(getattr(prompt_details, "cached_tokens", 0) or 0)
        reasoning = int(getattr(output_details, "reasoning_tokens", 0) or 0)
        with self._lock:
            slot = self.by_metric.setdefault(key, {
                "calls": 0, "input_tokens": 0, "cached_input_tokens": 0,
                "output_tokens": 0, "reasoning_tokens": 0, "cost_usd": 0.0, "model_ids": [],
            })
            slot["calls"] += 1
            slot["input_tokens"] += prompt
            slot["cached_input_tokens"] += cached
            slot["output_tokens"] += output
            slot["reasoning_tokens"] += reasoning
            slot["cost_usd"] += prompt * (input_price or 0.0) + output * (output_price or 0.0)
            model_id = getattr(completion, "model", None)
            if model_id and model_id not in slot["model_ids"]:
                slot["model_ids"].append(model_id)

    def total_cost(self) -> float:
        with self._lock:
            return sum(slot["cost_usd"] for slot in self.by_metric.values())

    def total_tokens(self) -> dict:
        with self._lock:
            keys = ("calls", "input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens")
            return {k: sum(slot[k] for slot in self.by_metric.values()) for k in keys}

    def model_ids(self) -> list[str]:
        with self._lock:
            return sorted({m for slot in self.by_metric.values() for m in slot["model_ids"]})

    def snapshot(self) -> dict:
        with self._lock:
            return {
                key: {**slot, "cost_usd": round(slot["cost_usd"], 6), "model_ids": list(slot["model_ids"])}
                for key, slot in self.by_metric.items()
            }


def register_judge_model(name: str) -> dict:
    """Make DeepEval price and call ``name`` correctly; fail loudly if it cannot.

    DeepEval 4.0.0 looks every OpenAI model up in OPENAI_MODELS_DATA
    (deepeval/models/llms/constants.py:18-44); an unknown name silently gets no
    prices (cost None) and supports_temperature=True (temperature=0.0 is sent,
    which gpt-6-luna rejects). Registering the model fixes both.
    """
    from deepeval.models.llms.constants import OPENAI_MODELS_DATA, make_model_data

    if dict.__contains__(OPENAI_MODELS_DATA, name):
        data = OPENAI_MODELS_DATA[name]
        if data.input_price and data.output_price:
            return {"model": name, "registry": "deepeval-builtin",
                    "input_price_per_token": data.input_price, "output_price_per_token": data.output_price,
                    "supports_temperature": data.supports_temperature,
                    "supports_structured_outputs": data.supports_structured_outputs}

    prices = price_for(name)
    family = next((alias for alias in JUDGE_CAPABILITIES if name == alias or name.startswith(f"{alias}-")), None)
    if not prices or family is None:
        raise CostAccountingError(
            f"Judge model {name!r} is not priced/registered: add it to "
            "backend.config.MODEL_PRICING_USD_PER_MILLION and evaluate.JUDGE_CAPABILITIES."
        )
    capabilities = JUDGE_CAPABILITIES[family]
    OPENAI_MODELS_DATA[name] = make_model_data(
        **capabilities,
        input_price=prices["input"] / 1_000_000,
        output_price=prices["output"] / 1_000_000,
    )
    return {"model": name, "registry": "registered-by-evaluate.py",
            "input_price_per_token": prices["input"] / 1_000_000,
            "output_price_per_token": prices["output"] / 1_000_000,
            **capabilities}


_TRACKED_JUDGE_CLASS = None


def tracked_judge_class():
    """GPTModel subclass that books every completion in a JudgeLedger.

    A GPTModel subclass is still a "native" DeepEval model
    (deepeval/metrics/utils.py:644-660), so DeepEval keeps accruing
    ``evaluation_cost`` per metric; the ledger adds token counts, cached and
    reasoning tokens, and the resolved model id (completion.model), which
    DeepEval does not expose. It hooks _update_llm_span_from_completion, which
    every generate/a_generate path calls (deepeval/models/llms/openai_model.py:177,194,207).
    """
    global _TRACKED_JUDGE_CLASS
    if _TRACKED_JUDGE_CLASS is None:
        from deepeval.models import GPTModel

        class TrackedGPTModel(GPTModel):
            def __init__(self, *args, ledger: JudgeLedger, ledger_key: str, **kwargs):
                self._ledger = ledger
                self._ledger_key = ledger_key
                super().__init__(*args, **kwargs)

            def _update_llm_span_from_completion(self, completion, messages=None):
                self._ledger.add(
                    self._ledger_key, completion,
                    self.model_data.input_price, self.model_data.output_price,
                )
                return super()._update_llm_span_from_completion(completion, messages)

        _TRACKED_JUDGE_CLASS = TrackedGPTModel
    return _TRACKED_JUDGE_CLASS


def build_metrics(metric_names: list[str], judge_model: str, ledger: JudgeLedger,
                  reasoning_effort: str = DEFAULT_JUDGE_REASONING_EFFORT) -> list:
    import deepeval.metrics as deepeval_metrics

    judge_cls = tracked_judge_class()
    generation_kwargs = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
    metrics = []
    for name in metric_names:
        judge = judge_cls(
            model=judge_model, generation_kwargs=generation_kwargs, ledger=ledger, ledger_key=name,
        )
        if not (judge.model_data.input_price and judge.model_data.output_price):
            raise CostAccountingError(f"DeepEval has no price for judge {judge_model!r}; refusing to run.")
        metric_cls = getattr(deepeval_metrics, METRICS[name][0])
        metrics.append(metric_cls(threshold=METRIC_THRESHOLD, model=judge))
    return metrics


def ensure_utf8_stdio() -> None:
    """DeepEval prints rich warnings with non-cp1252 glyphs; a cp1252 Windows
    console would raise UnicodeEncodeError *after* the judge calls were paid."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and (getattr(stream, "encoding", "") or "").lower() != "utf-8":
            stream.reconfigure(encoding="utf-8")


def judge_test_cases(test_cases: list, metrics: list) -> list:
    ensure_utf8_stdio()
    from deepeval import evaluate
    from deepeval.evaluate import AsyncConfig, CacheConfig, DisplayConfig

    results = evaluate(
        test_cases,
        metrics,
        async_config=AsyncConfig(max_concurrent=3, throttle_value=0),
        display_config=DisplayConfig(print_results=False, show_indicator=False),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
    )
    return results.test_results


def map_results_to_cases(test_results: list, cases: list[dict]) -> list[tuple[dict, Any]]:
    """Pair DeepEval results with their cases by input text.

    DeepEval returns results in completion order, not input order, so zipping
    them with the case list mislabels categories (the old evaluate.py:206 bug).
    Repeated inputs are matched first-come-first-served.
    """
    by_input: dict[str, list] = {}
    for result in test_results:
        by_input.setdefault(result.input, []).append(result)
    paired = []
    for case in cases:
        bucket = by_input.get(case["input"])
        if not bucket:
            raise KeyError(f"No DeepEval result for case {case.get('case_id')}: {case['input'][:80]!r}")
        paired.append((case, bucket.pop(0)))
    leftover = sum(len(bucket) for bucket in by_input.values())
    if leftover:
        raise KeyError(f"{leftover} DeepEval result(s) did not match any case input")
    return paired


def summarise_metrics_data(metrics_data: list) -> list[dict]:
    rows = []
    for m in metrics_data or []:
        if m.evaluation_cost is None and not m.error:
            raise CostAccountingError(
                f"DeepEval reported evaluation_cost=None for {m.name}; judge spend is unmeasured."
            )
        rows.append({
            "name": m.name,
            "score": m.score,
            "passed": m.success,
            "threshold": m.threshold,
            "evaluation_cost": m.evaluation_cost,
            "evaluation_model": m.evaluation_model,
            "reason": m.reason,
            "error": m.error,
        })
    return rows


# ── App pass ──────────────────────────────────────────────────────────────────

def run_rag_query(
    graph, query: str, session_id: str, usage_callback=None, paper_only: bool = True
) -> tuple[str, list[str], dict]:
    from langchain_core.messages import HumanMessage

    config = {"configurable": {"thread_id": str(session_id)}}
    if usage_callback is not None:
        config["callbacks"] = [usage_callback]
    final_state = graph.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "session_id": session_id,
            "query": query,
            "retrieved_docs": [],
            "retrieval_attempts": 0,
            "rewrite_count": 0,
            "force_retrieval": True,
            # Paper-QA eval: the retrieve agent must not answer from web_search.
            "paper_only": paper_only,
        },
        config=config,
    )
    answer = final_state.get("answer") or ""
    retrieval_context = [doc.page_content for doc in (final_state.get("retrieved_docs") or [])]
    return answer, retrieval_context, final_state


def trace_case(final_state: dict) -> dict:
    """Route, tool calls by source and retrieved-chunk provenance for one case."""
    tool_calls = {"vectorstore": 0, "web": 0, "other": 0}
    for message in final_state.get("messages") or []:
        for call in getattr(message, "tool_calls", None) or []:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            key = "vectorstore" if name == VECTORSTORE_TOOL else "web" if name == WEB_TOOL else "other"
            tool_calls[key] += 1
    docs = final_state.get("retrieved_docs") or []
    sources = {"paper": 0, "web": 0}
    for doc in docs:
        sources["web" if (doc.metadata or {}).get("url") else "paper"] += 1
    contents = [doc.page_content for doc in docs]
    return {
        "route": final_state.get("route"),
        "tool_calls": tool_calls,
        "retrieved_doc_sources": sources,
        "retrieved_count": len(docs),
        "unique_chunk_count": len(set(contents)),
        "retrieval_attempts": final_state.get("retrieval_attempts", 0),
        "rewrite_count": final_state.get("rewrite_count", 0),
        "retrieval_context_sha256": hashlib.sha256("\x1e".join(contents).encode("utf-8")).hexdigest(),
    }


def app_usage(usage_by_model: dict) -> dict:
    """Tokens and cost per resolved model id; unpriced usage is an error."""
    per_model = {}
    totals = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
    cost = 0.0
    for model, usage in (usage_by_model or {}).items():
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        cached = int((usage.get("input_token_details") or {}).get("cache_read", 0) or 0)
        reasoning = int((usage.get("output_token_details") or {}).get("reasoning", 0) or 0)
        model_cost = estimate_cost_usd(model, input_tokens, output_tokens)
        if (input_tokens or output_tokens) and not model_cost:
            raise CostAccountingError(
                f"App model {model!r} used tokens but has no price in "
                "backend.config.MODEL_PRICING_USD_PER_MILLION; refusing to report $0."
            )
        per_model[model] = {
            "input_tokens": input_tokens, "cached_input_tokens": cached,
            "output_tokens": output_tokens, "reasoning_tokens": reasoning,
            "cost_usd": round(model_cost, 8),
        }
        totals["input_tokens"] += input_tokens
        totals["cached_input_tokens"] += cached
        totals["output_tokens"] += output_tokens
        totals["reasoning_tokens"] += reasoning
        cost += model_cost
    return {"resolved_model_ids": sorted(per_model), "per_model": per_model, **totals, "cost_usd": round(cost, 8)}


# ── Aggregation ───────────────────────────────────────────────────────────────

def bootstrap_ci(values: list[float], n_resamples: int = BOOTSTRAP_RESAMPLES,
                 seed: int = BOOTSTRAP_SEED, confidence: float = 0.95) -> list[float] | None:
    """Percentile bootstrap CI of the mean (stdlib, deterministic for a seed)."""
    if not values:
        return None
    if len(values) == 1:
        return [round(values[0], 4), round(values[0], 4)]
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_resamples))
    tail = (1 - confidence) / 2
    low = means[int(tail * n_resamples)]
    high = means[min(n_resamples - 1, int((1 - tail) * n_resamples) - 1)]
    return [round(low, 4), round(high, 4)]


def aggregate(tests: list[dict], n_resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> dict:
    """Per-metric mean, per-run means (+ std), pass rate, n and case-level bootstrap CI.

    The CI resamples *cases* (each case's score averaged over its runs), so it
    reflects case sampling only, not judge or pipeline run-to-run variance; the
    std of per-run means is reported separately for that. Works on the old
    report format too (no run_index / case_id).
    """
    slots: dict[str, dict] = {}
    judge_cost = 0.0
    for test in tests:
        run = test.get("run_index", 0)
        case_key = test.get("case_id") or test.get("input")
        for m in test.get("metrics") or []:
            slot = slots.setdefault(m["name"], {"scores": [], "by_run": {}, "by_case": {}, "passed": 0, "n": 0, "unscored": 0})
            slot["n"] += 1
            slot["passed"] += 1 if m.get("passed") else 0
            judge_cost += m.get("evaluation_cost") or 0.0
            if m.get("score") is None:
                slot["unscored"] += 1
                continue
            slot["scores"].append(m["score"])
            slot["by_run"].setdefault(run, []).append(m["score"])
            slot["by_case"].setdefault(case_key, []).append(m["score"])

    out = {}
    for name, slot in slots.items():
        scores = slot["scores"]
        run_means = [round(statistics.fmean(slot["by_run"][r]), 4) for r in sorted(slot["by_run"])]
        case_means = [statistics.fmean(v) for v in slot["by_case"].values()]
        mean = statistics.fmean(scores) if scores else None
        out[name] = {
            "mean": round(mean, 4) if mean is not None else None,
            "avg_score": round(mean, 4) if mean is not None else None,
            "per_run_means": run_means,
            "run_mean_std": round(statistics.stdev(run_means), 4) if len(run_means) > 1 else None,
            "pass_rate": round(slot["passed"] / slot["n"], 4) if slot["n"] else None,
            "passed": slot["passed"],
            "n": slot["n"],
            "n_cases": len(slot["by_case"]),
            "runs": len(slot["by_run"]),
            "unscored": slot["unscored"],
            "ci95": bootstrap_ci(case_means, n_resamples, seed),
            "ci95_method": f"case-level percentile bootstrap, {n_resamples} resamples, seed {seed}",
        }
    return {"per_metric": out, "judge_cost_usd": round(judge_cost, 6)}


# ── Reports ───────────────────────────────────────────────────────────────────

def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_report_markdown(report: dict, md_path: Path) -> None:
    exp, meta = report["experiment"], report["metadata"]
    git = meta["git"]
    lines = [
        f"# RAG evaluation: {exp['label']}",
        "",
        f"- **Status:** {report['status']}"
        + (f" (TRUNCATED: {report['budget']['stop_reason']})" if report["budget"]["truncated"] else ""),
        f"- **Goldens:** {exp['goldens_source']} ({exp['case_count']} cases) x {meta['runs_requested']} run(s); "
        f"{exp['case_runs_completed']} case-runs judged",
        f"- **App model:** `{exp['application_model']}` (resolved: {', '.join(meta['app_resolved_model_ids']) or 'n/a'})",
        f"- **Judge model:** `{exp['judge_model']}` (resolved: {', '.join(meta['judge_resolved_model_ids']) or 'n/a'}; "
        f"reasoning_effort={meta['judge']['reasoning_effort'] or 'default'})",
        f"- **Retrieval:** mode=`{exp['retrieval_mode']}`, rerank={exp['rerank_enabled']}, "
        f"candidate_k={exp['retrieval_candidate_k']}, top_n={exp['rerank_top_n']}, paper_only={meta['paper_only']}",
        f"- **Git:** `{git['sha']}`{' (dirty)' if git['dirty'] else ''}",
        f"- **Measured cost:** app ${exp['app_cost_usd']:.4f} + judge ${exp['judge_cost_usd']:.4f} "
        f"= **${exp['total_cost_usd']:.4f}** (budget ${report['budget']['budget_usd']:.2f})",
        f"- **Command:** `{meta['command']}`",
        "",
        f"| Metric | Mean | 95% CI (cases) | Per-run means | Std of run means | Pass (>= {METRIC_THRESHOLD}) | n (cases x runs) |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    for name, stats in report["aggregate"].items():
        ci = stats.get("ci95")
        lines.append(
            f"| {name} | {_fmt(stats['mean'])} | {'[' + ', '.join(_fmt(v) for v in ci) + ']' if ci else '-'} | "
            f"{', '.join(_fmt(v) for v in stats['per_run_means']) or '-'} | {_fmt(stats['run_mean_std'])} | "
            f"{stats['passed']}/{stats['n']} | {stats['n_cases']} x {stats['runs']} |"
        )
    errors = [t for t in report["tests"] if t.get("error")]
    lines += [
        "",
        f"Avg retrieved chunks {exp['avg_retrieved_chunks']} (unique {exp['avg_unique_chunks']}), "
        f"avg latency {exp['avg_latency_seconds']} s, web tool calls {exp['web_tool_calls']}, "
        f"case errors {len(errors)}.",
        "",
        "> CI covers case sampling only; judge and pipeline variance show up in the per-run means. "
        "DeepEval Faithfulness counts claims the context does not contradict as faithful, so it is a "
        "grounding signal, not answer correctness.",
        "",
    ]
    Path(md_path).write_text("\n".join(lines), encoding="utf-8")


def build_report(*, args, status, cases, sources, metric_names, tests, guard, ledger,
                 judge_info, started_utc, output_path, git: dict) -> dict:
    agg = aggregate(tests)
    completed = [t for t in tests if t.get("metrics")]
    app_cost = round(sum(t["app"]["cost_usd"] for t in tests if t.get("app")), 6)
    judge_cost = round(sum(t.get("judge", {}).get("cost_usd", 0.0) for t in tests), 6)
    traced = [t for t in tests if t.get("trace")]
    retrieved = [t["trace"]["retrieved_count"] for t in traced]
    unique = [t["trace"]["unique_chunk_count"] for t in traced]
    latencies = [t["latency_seconds"] for t in tests if t.get("latency_seconds") is not None]
    app_ids = sorted({m for t in tests if t.get("app") for m in t["app"]["resolved_model_ids"]})
    settings = settings_snapshot()
    return {
        "schema_version": REPORT_SCHEMA,
        "status": status,
        "experiment": {
            "label": args.label,
            "goldens_source": args.goldens,
            "judge_model": args.model,
            "application_model": OPENAI_CHAT_MODEL,
            "embedding_model": OPENAI_EMBEDDING_MODEL,
            "retrieval_mode": settings["retrieval_mode"],
            "rerank_enabled": settings["rerank_enabled"],
            "retrieval_candidate_k": settings["retrieval_candidate_k"],
            "rerank_top_n": settings["rerank_top_n"],
            "case_count": len(cases),
            "case_start": args.start,
            "case_runs_completed": len(completed),
            "metric_threshold": METRIC_THRESHOLD,
            "judge_cost_usd": judge_cost,
            "app_cost_usd": app_cost,
            "total_cost_usd": round(judge_cost + app_cost, 6),
            "avg_retrieved_chunks": round(statistics.fmean(retrieved), 2) if retrieved else 0,
            "avg_unique_chunks": round(statistics.fmean(unique), 2) if unique else 0,
            "avg_latency_seconds": round(statistics.fmean(latencies), 2) if latencies else 0,
            "web_tool_calls": sum(t["trace"]["tool_calls"]["web"] for t in traced),
        },
        "metadata": {
            "git": git,
            "started_utc": started_utc,
            "ended_utc": utc_now(),
            "versions": package_versions(),
            "command": command_line(),
            "argv": relative_argv()[1:],
            "output": repo_relative(output_path),
            "app_model_alias": OPENAI_CHAT_MODEL,
            "app_resolved_model_ids": app_ids,
            "judge": judge_info,
            "judge_resolved_model_ids": ledger.model_ids(),
            "embedding_model": OPENAI_EMBEDDING_MODEL,
            "settings": settings,
            "paper_only": not args.allow_web,
            "goldens": {"source": args.goldens, "files": sources, "n_selected": len(cases)},
            "metrics": metric_names,
            "metric_thresholds": {METRICS[name][1]: METRIC_THRESHOLD for name in metric_names},
            "runs_requested": args.runs,
            "document": PDF_PATH,
        },
        "budget": guard.to_dict(),
        "judge_usage": {"totals": ledger.total_tokens(), "by_metric": ledger.snapshot(),
                        "ledger_cost_usd": round(ledger.total_cost(), 6)},
        "aggregate": agg["per_metric"],
        "runs": [
            {"run_index": r,
             "case_runs_judged": sum(1 for t in completed if t["run_index"] == r),
             "per_metric_mean": {name: stats["mean"] for name, stats in
                                 aggregate([t for t in completed if t["run_index"] == r], n_resamples=1)["per_metric"].items()}}
            for r in sorted({t["run_index"] for t in completed})
        ],
        "tests": tests,
    }


# ── Main evaluation loop ──────────────────────────────────────────────────────

def print_plan(plan: dict, args, output_path: Path) -> None:
    per = plan["per_case_estimate"]
    files = ", ".join(f"{s['set']}:{s['n']} sha256={s['sha256'][:12]}" for s in plan["sources"])
    print(f"[dry-run] {args.label}: goldens={plan['goldens']} -> {plan['n_cases']} cases ({files})")
    print(f"  runs={plan['runs']} case-runs={plan['case_runs']} metrics={','.join(plan['metrics'])}")
    print(f"  app={plan['app_model']} judge={plan['judge_model']} paper_only={not args.allow_web} "
          f"retrieval={getattr(cfg, 'RETRIEVAL_MODE', None)} rerank={getattr(cfg, 'RERANK_ENABLED', None)}")
    print(f"  per-case estimate: app ${_fmt(per['app_usd'], 5)} + judge ${_fmt(per['judge_usd'], 5)} "
          f"(judge price x{per['judge_price_scale_vs_reference']} of {per['reference_judge_model']}, "
          f"safety x{per['safety_factor']})")
    total = plan["estimated_total_usd"]
    fits = "fits" if total is not None and total <= args.budget_usd else "EXCEEDS (run would truncate)"
    print(f"  estimated total ${_fmt(total, 4)} vs budget ${args.budget_usd:.2f}: {fits}")
    print(f"  would write {output_path} (+ .md)")


def run_evaluation(args) -> dict:
    metric_names = args.metrics
    cases, sources = load_goldens(args.goldens)
    cases = select_range(cases, args.start, args.limit)
    output_path = Path(args.output) if args.output else default_output_path(args.label)
    ensure_writable(output_path, args.overwrite)

    plan = estimate_plan(args.goldens, metric_names, args.runs, args.start, args.limit,
                         OPENAI_CHAT_MODEL, args.model)
    if args.dry_run:
        print_plan(plan, args, output_path)
        return {"dry_run": True, "plan": plan, "output": str(output_path)}

    per_case = plan["per_case_estimate"]
    if per_case["total_usd"] is None:
        raise CostAccountingError(
            f"Cannot price app model {OPENAI_CHAT_MODEL!r} or judge {args.model!r}; refusing to run."
        )
    judge_info = register_judge_model(args.model)
    judge_info["reasoning_effort"] = args.judge_reasoning_effort
    ledger = JudgeLedger()
    metrics = build_metrics(metric_names, args.model, ledger, args.judge_reasoning_effort)
    guard = BudgetGuard(args.budget_usd, per_case["total_usd"], per_case["judge_usd"])

    from langchain_core.callbacks import UsageMetadataCallbackHandler
    from deepeval.test_case import LLMTestCase

    from backend.paper_loader import load_document
    from backend.rag_graph import build_graph
    from backend.telemetry import record_run
    from backend.vector_store import add_paper, delete_session_collection

    docs = load_document(str(REPO_ROOT / PDF_PATH))
    graph = build_graph(db_path=str(REPO_ROOT / "eval_checkpoints.db"))
    started_utc = utc_now()
    # Captured once at start: the code that ran, not whatever HEAD is when a report is saved.
    run_git = git_info()
    tests: list[dict] = []
    status = "running"
    consecutive_failures = 0

    def save(current_status: str) -> dict:
        report = build_report(
            args=args, status=current_status, cases=cases, sources=sources,
            metric_names=metric_names, tests=tests, guard=guard, ledger=ledger,
            judge_info=judge_info, started_utc=started_utc, output_path=output_path, git=run_git,
        )
        write_json(output_path, report)
        return report

    try:
        for run_index in range(args.runs):
            for case in cases:
                if not guard.can_start_case():
                    break
                record = {"run_index": run_index, **case, "session_id": f"evaluation_session_{uuid4()}"}
                tests.append(record)
                session_id = record["session_id"]
                try:
                    add_paper(docs, session_id)
                    usage_callback = UsageMetadataCallbackHandler()
                    started = time.perf_counter()
                    try:
                        answer, retrieval_context, final_state = run_rag_query(
                            graph, case["input"], session_id, usage_callback, paper_only=not args.allow_web
                        )
                    finally:
                        record["latency_seconds"] = round(time.perf_counter() - started, 3)
                        record["app"] = app_usage(usage_callback.usage_metadata)
                        guard.charge(record["app"]["cost_usd"])
                    ledger_row = record_run(
                        source="deepeval_app_run",
                        session_id=session_id,
                        state=final_state,
                        latency_seconds=record["latency_seconds"],
                        model_names=[OPENAI_CHAT_MODEL],
                        usage_by_model=usage_callback.usage_metadata,
                    )
                    record["app"]["record_run_estimated_cost_usd"] = ledger_row.get("estimated_cost_usd")
                    record["actual_output"] = answer
                    record["trace"] = trace_case(final_state)

                    if not retrieval_context:
                        record["error"] = "empty retrieval_context; judge skipped"
                        continue
                    if not guard.can_start_judge():
                        record["error"] = "budget: judge skipped"
                        break

                    before = ledger.total_cost()
                    test_case = LLMTestCase(
                        input=case["input"],
                        actual_output=answer,
                        expected_output=case["expected_output"],
                        retrieval_context=retrieval_context,
                    )
                    try:
                        results = judge_test_cases([test_case], metrics)
                    finally:
                        ledger_cost = ledger.total_cost() - before
                        guard.charge(ledger_cost)
                        record["judge"] = {"ledger_cost_usd": round(ledger_cost, 8), "cost_usd": round(ledger_cost, 8)}
                    (_, result), = map_results_to_cases(results, [case])
                    record["success"] = result.success
                    record["metrics"] = summarise_metrics_data(result.metrics_data)
                    metric_cost = sum(m["evaluation_cost"] or 0.0 for m in record["metrics"])
                    if metric_cost <= 0 or ledger_cost <= 0:
                        raise CostAccountingError(
                            f"Judge cost not measured for {case['case_id']} "
                            f"(metrics ${metric_cost}, ledger ${ledger_cost})."
                        )
                    judge_cost = max(metric_cost, ledger_cost)
                    guard.charge(judge_cost - ledger_cost)
                    record["judge"].update({
                        "metric_cost_usd": round(metric_cost, 8),
                        "cost_usd": round(judge_cost, 8),
                        "by_metric": {m["name"]: m["evaluation_cost"] for m in record["metrics"]},
                    })
                    guard.record_case(record["app"]["cost_usd"] + judge_cost, judge_cost)
                    consecutive_failures = 0
                except CostAccountingError:
                    raise
                except Exception as exc:  # one failed case must not lose the run's record
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    print(f"Case {case['case_id']} (run {run_index}) failed: {record['error']}", file=sys.stderr)
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        raise RuntimeError(
                            f"{consecutive_failures} consecutive case failures; aborting so a broken "
                            "judge or pipeline does not keep spending."
                        ) from exc
                finally:
                    if not args.keep_collections:
                        try:
                            delete_session_collection(session_id)
                        except Exception as exc:
                            print(f"Warning: could not delete {session_id}: {exc}", file=sys.stderr)
                    save("running")
                    scores = ", ".join(f"{m['name']}={_fmt(m['score'])}" for m in record.get("metrics") or [])
                    print(f"[run {run_index} {case['case_id']}] {scores or record.get('error')} "
                          f"| spent ${guard.spent_usd:.4f}/{guard.budget_usd:.2f}")
            if guard.truncated:
                break
        status = "truncated" if guard.truncated else "complete"
    except BaseException:
        status = "aborted"
        raise
    finally:
        report = save(status)
        write_report_markdown(report, output_path.with_suffix(".md"))

    exp = report["experiment"]
    print(f"\n[{exp['label']}] {status} | {exp['case_runs_completed']} case-runs | mode={exp['retrieval_mode']} "
          f"rerank={exp['rerank_enabled']} | judge=${exp['judge_cost_usd']} app=${exp['app_cost_usd']} "
          f"total=${exp['total_cost_usd']}")
    for name, stats in report["aggregate"].items():
        print(f"  {name:24s} mean={stats['mean']} ci95={stats['ci95']} pass={stats['passed']}/{stats['n']}")
    print(f"Results saved to {output_path} and {output_path.with_suffix('.md')}.")
    return report


def write_comparison(baseline_path: str, improved_path: str, out_path: str) -> None:
    """Before/after markdown from two reports (old flat format or schema 2)."""
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    improved = json.loads(Path(improved_path).read_text(encoding="utf-8"))
    b, i = baseline.get("experiment", {}), improved.get("experiment", {})
    ba = baseline.get("aggregate") or aggregate(baseline.get("tests", []))["per_metric"]
    ia = improved.get("aggregate") or aggregate(improved.get("tests", []))["per_metric"]

    def ci(stats: dict) -> str:
        bounds = stats.get("ci95")
        return f"[{bounds[0]}, {bounds[1]}]" if bounds else "-"

    lines = [
        "# Retrieval A/B Evaluation — Baseline vs. Hybrid+Rerank",
        "",
        f"- **Document:** `{PDF_PATH}`",
        f"- **Question set:** {b.get('goldens_source')} ({b.get('case_count')} cases), "
        f"judge `{b.get('judge_model')}`, app `{b.get('application_model')}`, threshold {b.get('metric_threshold')}",
        f"- **Baseline retrieval:** mode=`{b.get('retrieval_mode')}`, rerank={b.get('rerank_enabled')}",
        f"- **Improved retrieval:** mode=`{i.get('retrieval_mode')}`, rerank={i.get('rerank_enabled')}, "
        f"candidate_k={i.get('retrieval_candidate_k')} → top_n={i.get('rerank_top_n')}",
        f"- **Git:** baseline `{baseline.get('metadata', {}).get('git', {}).get('sha', 'n/a')}`, "
        f"improved `{improved.get('metadata', {}).get('git', {}).get('sha', 'n/a')}`",
        "",
        "## Metric scores (average, threshold 0.7)",
        "",
        "| Metric | Baseline | 95% CI | Improved | 95% CI | Δ |",
        "|---|---:|---|---:|---|---:|",
    ]
    for name in ba:
        bv = ba[name].get("avg_score")
        iv = ia.get(name, {}).get("avg_score")
        delta = round(iv - bv, 4) if (bv is not None and iv is not None) else None
        arrow = ""
        if delta is not None:
            arrow = " 🔼" if delta > 0 else (" 🔽" if delta < 0 else " ▪")
        lines.append(f"| {name} | {bv} | {ci(ba[name])} | {iv} | {ci(ia.get(name, {}))} | {delta}{arrow} |")

    lines += [
        "",
        "## Pass rates (cases meeting the 0.7 bar)",
        "",
        "| Metric | Baseline | Improved |",
        "|---|---:|---:|",
    ]
    for name in ba:
        bp = f"{ba[name].get('passed')}/{ba[name].get('n')}"
        ip = f"{ia.get(name, {}).get('passed', '-')}/{ia.get(name, {}).get('n', '-')}"
        lines.append(f"| {name} | {bp} | {ip} |")

    lines += [
        "",
        "## Efficiency & measured cost",
        "",
        "| Measure | Baseline | Improved |",
        "|---|---:|---:|",
        f"| Avg retrieved chunks / query | {b.get('avg_retrieved_chunks')} | {i.get('avg_retrieved_chunks')} |",
        f"| Avg end-to-end latency (s) | {b.get('avg_latency_seconds')} | {i.get('avg_latency_seconds')} |",
        f"| Application cost (USD) | {b.get('app_cost_usd')} | {i.get('app_cost_usd')} |",
        f"| DeepEval judge cost (USD) | {b.get('judge_cost_usd')} | {i.get('judge_cost_usd')} |",
        f"| **Total measured cost (USD)** | **{b.get('total_cost_usd')}** | **{i.get('total_cost_usd')}** |",
        "",
        "> Costs are measured from OpenAI usage callbacks (application) and DeepEval's "
        "per-metric `evaluation_cost` (judges). Reranking and BM25 sparse retrieval run "
        "locally on CPU and add no OpenAI cost.",
        "",
    ]
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Comparison written to {out_path}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a cost-capped, reproducible DeepEval RAG evaluation.")
    parser.add_argument("--goldens", default="curated",
                        help="curated | synthetic | all (both) | path to a goldens JSON file.")
    parser.add_argument("--metrics", type=parse_metrics, default=list(METRICS),
                        help=f"Comma list from: {', '.join(METRICS)} (default: all five).")
    parser.add_argument("--runs", type=int, default=1, help="Repeat the full app+judge pass N times.")
    parser.add_argument("--budget-usd", type=float, default=DEFAULT_BUDGET_USD,
                        help="Hard cap on measured app+judge spend; the run stops before exceeding it.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan and a cost estimate; no API call, no Qdrant connection.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate at most N cases.")
    parser.add_argument("--start", type=int, default=0, help="Zero-based starting case index.")
    parser.add_argument("--model", default=OPENAI_EVAL_MODEL, help="DeepEval judge model.")
    parser.add_argument("--judge-reasoning-effort", default=DEFAULT_JUDGE_REASONING_EFFORT,
                        help="reasoning_effort sent on judge calls ('' sends none).")
    parser.add_argument("--allow-web", action="store_true",
                        help="Let the retrieve agent use web_search (default: paper_only=True).")
    parser.add_argument("--label", default="rag", help="Human label; also names the output file.")
    parser.add_argument("--output", default=None, help="Output JSON path (default results/<label>_<UTC>.json).")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output file.")
    parser.add_argument("--keep-collections", action="store_true", help="Keep eval collections.")
    parser.add_argument("--generate-goldens", action="store_true",
                        help="Paid: regenerate goldens.json with DeepEval's Synthesizer, then exit.")
    parser.add_argument(
        "--compare",
        nargs=3,
        metavar=("BASELINE", "IMPROVED", "OUT_MD"),
        help="Write a before/after markdown from two report files (no API calls).",
    )
    return parser


def main(argv: list[str] | None = None):
    ensure_utf8_stdio()
    args = build_parser().parse_args(argv)
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")

    if args.compare:
        write_comparison(*args.compare)
        return None
    if args.generate_goldens:
        if GOLDENS_FILE.exists() and not args.overwrite:
            raise SystemExit(f"{GOLDENS_FILE} exists; pass --overwrite to regenerate it (paid).")
        generate_goldens()
        return None
    return run_evaluation(args)


if __name__ == "__main__":
    main()
