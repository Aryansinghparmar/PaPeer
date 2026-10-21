"""Router accuracy eval: labelled queries through ``backend.rag_graph.router_chain`` only.

No retrieval, no Qdrant search, no Tavily: one small structured-output call per
case, so it is cheap. Labels live in ``evals/router_cases.json``; ambiguous
cases carry a ``note`` and ``acceptable_routes`` (reported as a separate
lenient accuracy; the headline number is strict accuracy).

    python eval_router.py --dry-run
    python eval_router.py --runs 3 --budget-usd 0.10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

# evaluate.py imports only backend.config at module level, so this stays offline
# until a real run imports backend.rag_graph.
import evaluate as ev
from backend.config import OPENAI_CHAT_MODEL

ROUTER_CASES_FILE = ev.REPO_ROOT / "evals" / "router_cases.json"
ROUTES = ("retrieve", "verify_claim", "direct_answer")
DEFAULT_BUDGET_USD = 0.10
# Planning-only tokens for one router call (system prompt ~350 tokens + query;
# the structured output is a single enum field). Measured costs replace this
# after the first call.
PLANNING_ROUTER_TOKENS = {"input": 450, "output": 20}


def load_cases(path: Path = ROUTER_CASES_FILE) -> list[dict]:
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    seen = set()
    for case in cases:
        if case["expected_route"] not in ROUTES:
            raise ValueError(f"{case['id']}: unknown expected_route {case['expected_route']!r}")
        if case["id"] in seen:
            raise ValueError(f"duplicate case id {case['id']}")
        seen.add(case["id"])
    return cases


def estimate_call_cost(model: str = OPENAI_CHAT_MODEL) -> float | None:
    prices = ev.price_for(model)
    if not prices:
        return None
    return ev.PLANNING_SAFETY_FACTOR * (
        PLANNING_ROUTER_TOKENS["input"] * prices["input"]
        + PLANNING_ROUTER_TOKENS["output"] * prices["output"]
    ) / 1_000_000


def estimate_plan(runs: int = 1, path: Path = ROUTER_CASES_FILE, model: str = OPENAI_CHAT_MODEL) -> dict:
    cases = load_cases(path)
    per_call = estimate_call_cost(model)
    calls = len(cases) * runs
    return {
        "n_cases": len(cases),
        "runs": runs,
        "calls": calls,
        "model": model,
        "per_call_estimate_usd": None if per_call is None else round(per_call, 7),
        "estimated_total_usd": None if per_call is None else round(per_call * calls, 4),
        "cases_sha256": ev.file_sha256(path),
    }


def score_predictions(cases: list[dict], predictions: list[dict]) -> dict:
    """Accuracy, per-route precision/recall, confusion matrix and misroutes.

    ``predictions`` holds one row per (run, case): ``run_index``, ``case_id``,
    ``predicted`` (None when the call failed; counted as wrong, shown as "error").
    """
    by_id = {case["id"]: case for case in cases}
    labels = list(ROUTES) + ["error"]
    confusion = {expected: {predicted: 0 for predicted in labels} for expected in ROUTES}
    per_run: dict[int, list[int]] = {}
    per_case: dict[str, list[int]] = {}
    per_case_predictions: dict[str, list] = {}
    misroutes = []
    lenient_hits = 0
    for row in predictions:
        case = by_id[row["case_id"]]
        expected, predicted = case["expected_route"], row["predicted"]
        hit = int(predicted == expected)
        acceptable = case.get("acceptable_routes") or [expected]
        lenient_hits += int(predicted in acceptable)
        confusion[expected][predicted if predicted in ROUTES else "error"] += 1
        per_run.setdefault(row["run_index"], []).append(hit)
        per_case.setdefault(case["id"], []).append(hit)
        per_case_predictions.setdefault(case["id"], []).append(predicted)
        if not hit:
            misroutes.append({
                "run_index": row["run_index"], "case_id": case["id"], "intent": case.get("intent"),
                "query": case["query"], "expected": expected, "predicted": predicted,
                "acceptable": predicted in acceptable, "note": case.get("note"),
                "error": row.get("error"),
            })

    n = len(predictions)
    per_route = {}
    for route in ROUTES:
        tp = confusion[route][route]
        predicted_as = sum(confusion[e][route] for e in ROUTES)
        support = sum(confusion[route].values())
        precision = tp / predicted_as if predicted_as else None
        recall = tp / support if support else None
        f1 = (2 * precision * recall / (precision + recall)) if precision and recall else (0.0 if support else None)
        per_route[route] = {
            "precision": None if precision is None else round(precision, 4),
            "recall": None if recall is None else round(recall, 4),
            "f1": None if f1 is None else round(f1, 4),
            "support": support,
            "predicted": predicted_as,
        }
    run_accuracies = [round(statistics.fmean(per_run[r]), 4) for r in sorted(per_run)]
    case_means = [statistics.fmean(v) for v in per_case.values()]
    unstable = sorted(cid for cid, preds in per_case_predictions.items() if len(set(preds)) > 1)
    return {
        "accuracy": round(sum(sum(v) for v in per_run.values()) / n, 4) if n else None,
        "lenient_accuracy": round(lenient_hits / n, 4) if n else None,
        "per_run_accuracy": run_accuracies,
        "run_accuracy_std": round(statistics.stdev(run_accuracies), 4) if len(run_accuracies) > 1 else None,
        "ci95": ev.bootstrap_ci(case_means),
        "ci95_method": f"case-level percentile bootstrap, {ev.BOOTSTRAP_RESAMPLES} resamples, seed {ev.BOOTSTRAP_SEED}",
        "n": n,
        "n_cases": len(per_case),
        "runs": len(per_run),
        "errors": sum(1 for row in predictions if row["predicted"] is None),
        "per_route": per_route,
        "confusion_matrix": confusion,
        "confusion_labels": {"rows": "expected", "columns": "predicted"},
        "misroutes": misroutes,
        "unstable_case_ids": unstable,
    }


def router_prompt_sha256() -> str | None:
    try:
        from backend import rag_graph

        text = "\n".join(str(getattr(m, "prompt", m)) for m in rag_graph.ROUTER_PROMPT.messages)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    except Exception:
        return None


def write_markdown(report: dict, md_path: Path) -> None:
    s, meta = report["scores"], report["metadata"]
    lines = [
        f"# Router accuracy: {report['label']}",
        "",
        f"- **Status:** {report['status']}"
        + (f" (TRUNCATED: {report['budget']['stop_reason']})" if report["budget"]["truncated"] else ""),
        f"- **Cases:** {s['n_cases']} x {s['runs']} run(s) = {s['n']} calls; errors {s['errors']}",
        f"- **Model:** `{meta['model_alias']}` (resolved: {', '.join(meta['resolved_model_ids']) or 'n/a'}; "
        f"reasoning_effort={meta['settings'].get('openai_reasoning_effort') or 'default'})",
        f"- **Git:** `{meta['git']['sha']}`{' (dirty)' if meta['git']['dirty'] else ''}; "
        f"cases sha256 `{meta['cases_sha256'][:12]}`",
        f"- **Measured cost:** ${report['cost_usd']:.5f} (budget ${report['budget']['budget_usd']:.2f})",
        f"- **Command:** `{meta['command']}`",
        "",
        f"**Strict accuracy {ev._fmt(s['accuracy'])}** (95% CI {s['ci95']}), lenient "
        f"{ev._fmt(s['lenient_accuracy'])}; per-run {s['per_run_accuracy']} (std {ev._fmt(s['run_accuracy_std'])}).",
        "",
        "| Route | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for route, stats in s["per_route"].items():
        lines.append(f"| {route} | {ev._fmt(stats['precision'])} | {ev._fmt(stats['recall'])} | "
                     f"{ev._fmt(stats['f1'])} | {stats['support']} |")
    columns = list(ROUTES) + ["error"]
    lines += ["", "Confusion matrix (rows expected, columns predicted):", "",
              "| expected \\ predicted | " + " | ".join(columns) + " |",
              "|---|" + "---:|" * len(columns)]
    for expected, row in s["confusion_matrix"].items():
        lines.append(f"| {expected} | " + " | ".join(str(row[c]) for c in columns) + " |")
    if s["misroutes"]:
        lines += ["", "Misroutes:", ""]
        for m in s["misroutes"]:
            lines.append(f"- run {m['run_index']} `{m['case_id']}` expected {m['expected']}, got {m['predicted']}"
                         f"{' (acceptable)' if m['acceptable'] else ''}: {m['query']}")
    lines.append("")
    Path(md_path).write_text("\n".join(lines), encoding="utf-8")


def run_router_eval(args, chain=None) -> dict:
    cases = load_cases(Path(args.cases))
    output_path = Path(args.output) if args.output else ev.default_output_path(args.label)
    ev.ensure_writable(output_path, args.overwrite)
    plan = estimate_plan(args.runs, Path(args.cases))

    if args.dry_run:
        fits = (plan["estimated_total_usd"] is not None and plan["estimated_total_usd"] <= args.budget_usd)
        print(f"[dry-run] router: {plan['n_cases']} cases x {plan['runs']} runs = {plan['calls']} calls, "
              f"model={plan['model']}, ~${ev._fmt(plan['per_call_estimate_usd'], 6)}/call "
              f"(safety x{ev.PLANNING_SAFETY_FACTOR})")
        print(f"  estimated total ${ev._fmt(plan['estimated_total_usd'], 4)} vs budget ${args.budget_usd:.2f}: "
              f"{'fits' if fits else 'EXCEEDS (run would truncate)'}")
        print(f"  would write {output_path} (+ .md)")
        return {"dry_run": True, "plan": plan, "output": str(output_path)}

    if plan["per_call_estimate_usd"] is None:
        raise ev.CostAccountingError(f"No price for router model {OPENAI_CHAT_MODEL!r}; refusing to run.")
    from langchain_core.callbacks import UsageMetadataCallbackHandler

    prompt_sha = None
    if chain is None:
        from backend.rag_graph import router_chain as chain  # noqa: N813 (lazy: opens API clients)

        prompt_sha = router_prompt_sha256()

    guard = ev.BudgetGuard(args.budget_usd, plan["per_call_estimate_usd"], 0.0)
    started_utc = ev.utc_now()
    run_git = ev.git_info()  # the code that ran, captured at start
    predictions: list[dict] = []
    resolved: set[str] = set()
    cost = 0.0
    tokens = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
    for run_index in range(args.runs):
        for case in cases:
            if not guard.can_start_case():
                break
            callback = UsageMetadataCallbackHandler()
            row = {"run_index": run_index, "case_id": case["id"], "predicted": None}
            started = time.perf_counter()
            try:
                decision = chain.invoke({"query": case["query"]}, config={"callbacks": [callback]})
                row["predicted"] = getattr(decision, "route", None)
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            row["latency_seconds"] = round(time.perf_counter() - started, 3)
            usage = ev.app_usage(callback.usage_metadata)
            if not callback.usage_metadata and "error" not in row:
                raise ev.CostAccountingError("Router call reported no token usage; cost would be unmeasured.")
            row["cost_usd"] = usage["cost_usd"]
            row["resolved_model_ids"] = usage["resolved_model_ids"]
            for key in tokens:
                tokens[key] += usage[key]
            resolved.update(usage["resolved_model_ids"])
            cost += usage["cost_usd"]
            guard.charge(usage["cost_usd"])
            guard.record_case(usage["cost_usd"], 0.0)
            predictions.append(row)
        if guard.truncated:
            break

    report = {
        "schema_version": ev.REPORT_SCHEMA,
        "kind": "router_accuracy",
        "label": args.label,
        "status": "truncated" if guard.truncated else "complete",
        "metadata": {
            "git": run_git,
            "started_utc": started_utc,
            "ended_utc": ev.utc_now(),
            "versions": ev.package_versions(),
            "command": ev.command_line(),
            "argv": ev.relative_argv()[1:],
            "model_alias": OPENAI_CHAT_MODEL,
            "resolved_model_ids": sorted(resolved),
            "settings": ev.settings_snapshot(),
            "router_prompt_sha256": prompt_sha,
            "cases_file": ev.repo_relative(args.cases),
            "cases_sha256": ev.file_sha256(Path(args.cases)),
            "runs_requested": args.runs,
        },
        "budget": guard.to_dict(),
        "cost_usd": round(cost, 8),
        "tokens": tokens,
        "scores": score_predictions(cases, predictions),
        "predictions": predictions,
    }
    ev.write_json(output_path, report)
    write_markdown(report, output_path.with_suffix(".md"))
    s = report["scores"]
    print(f"[router] {report['status']} | accuracy={s['accuracy']} ci95={s['ci95']} lenient={s['lenient_accuracy']} "
          f"n={s['n']} | cost ${report['cost_usd']:.5f}")
    print(f"Results saved to {output_path} and {output_path.with_suffix('.md')}.")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure router accuracy on labelled queries.")
    parser.add_argument("--cases", default=str(ROUTER_CASES_FILE), help="Labelled router cases JSON.")
    parser.add_argument("--runs", type=int, default=1, help="Repeat every case N times.")
    parser.add_argument("--budget-usd", type=float, default=DEFAULT_BUDGET_USD, help="Hard spend cap.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan; no API call.")
    parser.add_argument("--label", default="router", help="Names the output file.")
    parser.add_argument("--output", default=None, help="Output JSON (default results/<label>_<UTC>.json).")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output.")
    return parser


def main(argv: list[str] | None = None, chain=None):
    ev.ensure_utf8_stdio()
    args = build_parser().parse_args(argv)
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")
    return run_router_eval(args, chain=chain)


if __name__ == "__main__":
    main()
