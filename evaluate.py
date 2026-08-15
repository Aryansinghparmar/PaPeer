"""Cost-controlled RAG evaluation with a controlled retrieval A/B.

Two passes are run as separate processes so the retrieval strategy (read from
config at import time) is unambiguous and reproducible:

    baseline  RETRIEVAL_MODE=dense  RERANK_ENABLED=false  -> eval_baseline.json
    improved  RETRIEVAL_MODE=hybrid RERANK_ENABLED=true    -> eval_improved.json

Each report records the retrieval settings and the MEASURED judge + application
cost so the spend is honest and auditable. A --compare mode then writes a
before/after markdown summary. DeepEval judge calls are the dominant cost, so
always run a 1-case probe first and size the run from the measured cost.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from uuid import uuid4

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import HumanMessage

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig
from deepeval.test_case import LLMTestCase

from backend.config import (
    OPENAI_CHAT_MODEL,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_EVAL_MODEL,
    RERANK_ENABLED,
    RERANK_TOP_N,
    RETRIEVAL_CANDIDATE_K,
    RETRIEVAL_MODE,
)
from backend.paper_loader import load_document
from backend.rag_graph import build_graph
from backend.telemetry import record_run
from backend.vector_store import add_paper, delete_session_collection

load_dotenv()

PDF_PATH            = "documents/Openclaw_Research_Report.pdf"
GOLDENS_FILE        = Path("goldens.json")
CURATED_FILE        = Path("goldens_curated.json")
MAX_CONTEXTS        = 5
GOLDENS_PER_CONTEXT = 2
METRIC_THRESHOLD    = 0.7


def generate_goldens() -> list[dict]:
    synthesizer = Synthesizer()
    goldens = synthesizer.generate_goldens_from_docs(
        document_paths=[PDF_PATH],
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


def load_goldens(source: str) -> list[dict]:
    """Load either the hand-curated set (default) or the synthetic set."""
    if source == "curated":
        return json.loads(CURATED_FILE.read_text(encoding="utf-8"))
    if GOLDENS_FILE.exists():
        return json.loads(GOLDENS_FILE.read_text(encoding="utf-8"))
    return generate_goldens()


def run_rag_query(
    graph, query: str, session_id: str, usage_callback=None
) -> tuple[str, list[str], dict]:
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
        },
        config=config,
    )
    answer = final_state.get("answer") or ""
    retrieval_context = [doc.page_content for doc in (final_state.get("retrieved_docs") or [])]
    return answer, retrieval_context, final_state


def aggregate(summary: list[dict]) -> dict:
    """Per-metric average score, pass rate, and total judge cost across cases."""
    metrics: dict[str, dict] = {}
    judge_cost = 0.0
    for case in summary:
        for m in case["metrics"]:
            slot = metrics.setdefault(m["name"], {"scores": [], "passed": 0, "n": 0})
            if m["score"] is not None:
                slot["scores"].append(m["score"])
            slot["passed"] += 1 if m["passed"] else 0
            slot["n"] += 1
            judge_cost += m.get("evaluation_cost") or 0.0
    out = {}
    for name, slot in metrics.items():
        scores = slot["scores"]
        out[name] = {
            "avg_score": round(sum(scores) / len(scores), 4) if scores else None,
            "pass_rate": round(slot["passed"] / slot["n"], 4) if slot["n"] else None,
            "passed": slot["passed"],
            "n": slot["n"],
        }
    return {"per_metric": out, "judge_cost_usd": round(judge_cost, 6)}


def run_evaluation(args) -> None:
    pairs = load_goldens(args.goldens)
    if args.start < 0:
        raise ValueError("--start cannot be negative")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        pairs = pairs[args.start : args.start + args.limit]
    else:
        pairs = pairs[args.start :]
    if not pairs:
        raise ValueError("The selected evaluation range is empty")

    docs = load_document(PDF_PATH)
    graph = build_graph(db_path="eval_checkpoints.db")

    metrics = [
        ContextualPrecisionMetric(threshold=METRIC_THRESHOLD, model=args.model),
        ContextualRecallMetric(threshold=METRIC_THRESHOLD, model=args.model),
        ContextualRelevancyMetric(threshold=METRIC_THRESHOLD, model=args.model),
        AnswerRelevancyMetric(threshold=METRIC_THRESHOLD, model=args.model),
        FaithfulnessMetric(threshold=METRIC_THRESHOLD, model=args.model),
    ]

    test_cases = []
    run_records = []
    evaluation_session_ids = []
    categories = []
    for pair in pairs:
        session_id = f"evaluation_session_{uuid4()}"
        evaluation_session_ids.append(session_id)
        categories.append(pair.get("category"))
        add_paper(docs, session_id)

        started = time.perf_counter()
        usage_callback = UsageMetadataCallbackHandler()
        answer, retrieval_context, final_state = run_rag_query(
            graph, pair["input"], session_id, usage_callback
        )
        run_records.append(
            record_run(
                source="deepeval_app_run",
                session_id=session_id,
                state=final_state,
                latency_seconds=time.perf_counter() - started,
                model_names=[OPENAI_CHAT_MODEL],
                usage_by_model=usage_callback.usage_metadata,
            )
        )
        test_cases.append(
            LLMTestCase(
                input=pair["input"],
                actual_output=answer,
                expected_output=pair["expected_output"],
                retrieval_context=retrieval_context,
            )
        )

    results = evaluate(
        test_cases,
        metrics,
        async_config=AsyncConfig(max_concurrent=3, throttle_value=5),
    )

    summary = []
    for test_result, category in zip(results.test_results, categories):
        summary.append({
            "input": test_result.input,
            "category": category,
            "actual_output": test_result.actual_output,
            "success": test_result.success,
            "metrics": [
                {
                    "name": m.name,
                    "score": m.score,
                    "passed": m.success,
                    "evaluation_cost": getattr(m, "evaluation_cost", None),
                    "reason": m.reason,
                }
                for m in test_result.metrics_data
            ],
        })

    agg = aggregate(summary)
    app_cost = round(
        sum(r.get("estimated_cost_usd") or 0.0 for r in run_records), 6
    )
    retrieved_counts = [r.get("retrieved_document_count", 0) for r in run_records]
    latencies = [r.get("latency_seconds", 0.0) for r in run_records]

    report = {
        "experiment": {
            "label": args.label,
            "goldens_source": args.goldens,
            "judge_model": args.model,
            "application_model": OPENAI_CHAT_MODEL,
            "embedding_model": OPENAI_EMBEDDING_MODEL,
            "retrieval_mode": RETRIEVAL_MODE,
            "rerank_enabled": RERANK_ENABLED,
            "retrieval_candidate_k": RETRIEVAL_CANDIDATE_K,
            "rerank_top_n": RERANK_TOP_N,
            "case_count": len(test_cases),
            "case_start": args.start,
            "metric_threshold": METRIC_THRESHOLD,
            "judge_cost_usd": agg["judge_cost_usd"],
            "app_cost_usd": app_cost,
            "total_cost_usd": round(agg["judge_cost_usd"] + app_cost, 6),
            "avg_retrieved_chunks": round(sum(retrieved_counts) / len(retrieved_counts), 2) if retrieved_counts else 0,
            "avg_latency_seconds": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        },
        "aggregate": agg["per_metric"],
        "tests": summary,
        "run_metrics": run_records,
    }
    Path(args.output).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if not args.keep_collections:
        for session_id in evaluation_session_ids:
            try:
                delete_session_collection(session_id)
            except Exception as exc:
                print(f"Warning: could not delete {session_id}: {exc}", file=sys.stderr)

    exp = report["experiment"]
    print(f"\n[{exp['label']}] {exp['case_count']} cases | mode={exp['retrieval_mode']} "
          f"rerank={exp['rerank_enabled']} | judge=${exp['judge_cost_usd']} "
          f"app=${exp['app_cost_usd']} total=${exp['total_cost_usd']}")
    for name, stats in report["aggregate"].items():
        print(f"  {name:24s} avg={stats['avg_score']}  pass={stats['passed']}/{stats['n']}")
    print(f"Results saved to {args.output}.")


def write_comparison(baseline_path: str, improved_path: str, out_path: str) -> None:
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    improved = json.loads(Path(improved_path).read_text(encoding="utf-8"))
    b, i = baseline["experiment"], improved["experiment"]
    ba, ia = baseline["aggregate"], improved["aggregate"]

    lines = [
        "# Retrieval A/B Evaluation — Baseline vs. Hybrid+Rerank",
        "",
        f"- **Document:** `{PDF_PATH}`",
        f"- **Question set:** {b['goldens_source']} ({b['case_count']} cases), "
        f"judge `{b['judge_model']}`, app `{b['application_model']}`, threshold {b['metric_threshold']}",
        f"- **Baseline retrieval:** mode=`{b['retrieval_mode']}`, rerank={b['rerank_enabled']}",
        f"- **Improved retrieval:** mode=`{i['retrieval_mode']}`, rerank={i['rerank_enabled']}, "
        f"candidate_k={i['retrieval_candidate_k']} → top_n={i['rerank_top_n']}",
        "",
        "## Metric scores (average, threshold 0.7)",
        "",
        "| Metric | Baseline | Improved | Δ |",
        "|---|---:|---:|---:|",
    ]
    for name in ba:
        bv = ba[name]["avg_score"]
        iv = ia.get(name, {}).get("avg_score")
        delta = round(iv - bv, 4) if (bv is not None and iv is not None) else None
        arrow = ""
        if delta is not None:
            arrow = " 🔼" if delta > 0 else (" 🔽" if delta < 0 else " ▪")
        lines.append(f"| {name} | {bv} | {iv} | {delta}{arrow} |")

    lines += [
        "",
        "## Pass rates (cases meeting the 0.7 bar)",
        "",
        "| Metric | Baseline | Improved |",
        "|---|---:|---:|",
    ]
    for name in ba:
        bp = f"{ba[name]['passed']}/{ba[name]['n']}"
        ip = f"{ia.get(name, {}).get('passed', '-')}/{ia.get(name, {}).get('n', '-')}"
        lines.append(f"| {name} | {bp} | {ip} |")

    lines += [
        "",
        "## Efficiency & measured cost",
        "",
        "| Measure | Baseline | Improved |",
        "|---|---:|---:|",
        f"| Avg retrieved chunks / query | {b['avg_retrieved_chunks']} | {i['avg_retrieved_chunks']} |",
        f"| Avg end-to-end latency (s) | {b['avg_latency_seconds']} | {i['avg_latency_seconds']} |",
        f"| Application cost (USD) | {b['app_cost_usd']} | {i['app_cost_usd']} |",
        f"| DeepEval judge cost (USD) | {b['judge_cost_usd']} | {i['judge_cost_usd']} |",
        f"| **Total measured cost (USD)** | **{b['total_cost_usd']}** | **{i['total_cost_usd']}** |",
        "",
        "> Costs are measured from OpenAI usage callbacks (application) and DeepEval's "
        "per-metric `evaluation_cost` (judges). Reranking and BM25 sparse retrieval run "
        "locally on CPU and add no OpenAI cost.",
        "",
    ]
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Comparison written to {out_path}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a cost-controlled DeepEval A/B experiment.")
    parser.add_argument("--goldens", choices=["curated", "synthetic"], default="curated")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate at most N cases.")
    parser.add_argument("--start", type=int, default=0, help="Zero-based starting case index.")
    parser.add_argument("--model", default=OPENAI_EVAL_MODEL, help="DeepEval judge model.")
    parser.add_argument("--label", default="run", help="Human label stored in the report.")
    parser.add_argument("--output", default="eval_results.json", help="Output JSON path.")
    parser.add_argument("--keep-collections", action="store_true", help="Keep eval collections.")
    parser.add_argument(
        "--compare",
        nargs=3,
        metavar=("BASELINE", "IMPROVED", "OUT_MD"),
        help="Write a before/after markdown from two report files (no API calls).",
    )
    args = parser.parse_args()

    if args.compare:
        write_comparison(*args.compare)
        return
    run_evaluation(args)


if __name__ == "__main__":
    main()
