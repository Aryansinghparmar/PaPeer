"""Regenerate every reported evaluation number with one command, under one spend cap.

Steps (in order; each gets whatever budget the previous steps left):
    router  eval_router.py, 3 runs over evals/router_cases.json
    rag     evaluate.py --goldens all --metrics faithfulness,answer_relevancy --runs 3
    full    evaluate.py --goldens all (all five metrics) --runs 1

Then writes results/SUMMARY.md (plus a timestamped copy) listing every headline
number with n, runs, CI, models, git SHA, measured cost and the exact command.

    python scripts/reproduce_eval.py --dry-run          # plan + estimate, offline
    python scripts/reproduce_eval.py --budget-usd 2.00  # the real thing
    python scripts/reproduce_eval.py --only router      # one step (comma list ok)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import eval_router  # noqa: E402  (offline at import: backend.config only)
import evaluate as ev  # noqa: E402

STEP_ORDER = ("router", "rag", "full")
DEFAULT_BUDGET_USD = 2.00
FA_METRICS = ["faithfulness", "answer_relevancy"]


def step_specs(stamp: str) -> dict[str, dict]:
    results = ev.RESULTS_DIR
    return {
        "router": {
            "title": "Router accuracy (3 runs)",
            "script": "eval_router.py",
            "output": results / f"router_x3_{stamp}.json",
            "args": ["--runs", "3", "--label", "router_x3"],
        },
        "rag": {
            "title": "Faithfulness + Answer Relevancy, goldens all, 3 runs",
            "script": "evaluate.py",
            "output": results / f"rag_core_x3_{stamp}.json",
            "args": ["--goldens", "all", "--metrics", ",".join(FA_METRICS), "--runs", "3", "--label", "rag_core_x3"],
        },
        "full": {
            "title": "All five DeepEval metrics, goldens all, 1 run",
            "script": "evaluate.py",
            "output": results / f"rag_all_metrics_x1_{stamp}.json",
            "args": ["--goldens", "all", "--metrics", ",".join(ev.METRICS), "--runs", "1", "--label", "rag_all_metrics_x1"],
        },
    }


def estimate_step(name: str) -> float | None:
    if name == "router":
        return eval_router.estimate_plan(runs=3)["estimated_total_usd"]
    if name == "rag":
        return ev.estimate_plan("all", FA_METRICS, 3)["estimated_total_usd"]
    return ev.estimate_plan("all", list(ev.METRICS), 1)["estimated_total_usd"]


def min_case_estimate(name: str) -> float:
    """Cheapest unit of work for a step; below this the step cannot start."""
    if name == "router":
        return eval_router.estimate_call_cost() or 0.0
    metrics = FA_METRICS if name == "rag" else list(ev.METRICS)
    return ev.estimate_case_cost(metrics, ev.OPENAI_CHAT_MODEL, ev.OPENAI_EVAL_MODEL)["total_usd"] or 0.0


def parse_only(value: str | None) -> list[str]:
    if not value:
        return list(STEP_ORDER)
    names = [part.strip() for part in value.split(",") if part.strip()]
    bad = [n for n in names if n not in STEP_ORDER]
    if bad:
        raise argparse.ArgumentTypeError(f"unknown step(s) {bad}; choose from {', '.join(STEP_ORDER)}")
    return [n for n in STEP_ORDER if n in names]


def step_command(spec: dict, budget: float, dry_run: bool) -> list[str]:
    cmd = [sys.executable, str(REPO_ROOT / spec["script"]), *spec["args"],
           "--budget-usd", f"{budget:.4f}", "--output", str(spec["output"])]
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def display_command(cmd: list[str]) -> str:
    shown = ["python"] + [Path(c).relative_to(REPO_ROOT).as_posix() if Path(c).is_absolute()
                          and REPO_ROOT in Path(c).parents else c for c in cmd[1:]]
    return " ".join(shown)


def spent_from(report_path: Path) -> float:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return float(report.get("budget", {}).get("spent_usd") or 0.0)


def summary_lines(steps_done: list[dict], budget: float, stamp: str) -> list[str]:
    git = ev.git_info()
    lines = [
        "# Evaluation summary",
        "",
        f"Generated {ev.utc_now()} by `python scripts/reproduce_eval.py` at git `{git['sha']}`"
        f"{' (dirty: ' + ', '.join(git['changed_files']) + ')' if git['dirty'] else ''}. "
        f"Overall budget ${budget:.2f}.",
        "",
        "Every number below is copied from the JSON report next to it; nothing is typed by hand. "
        "CIs are case-level bootstrap 95% intervals (case sampling only); run-to-run spread is the "
        "std of per-run means. DeepEval Faithfulness counts claims the context does not contradict as "
        "faithful, so it is a grounding signal, not correctness. App and judge are the same model "
        "(self-preference risk).",
        "",
    ]
    total = 0.0
    for step in steps_done:
        lines += [f"## {step['title']}", ""]
        if step.get("skipped"):
            lines += [f"Skipped: {step['skipped']}", ""]
            continue
        report = json.loads(Path(step["output"]).read_text(encoding="utf-8"))
        cost = float(report["budget"]["spent_usd"])
        total += cost
        rel = Path(step["output"]).relative_to(REPO_ROOT).as_posix()
        if report.get("kind") == "router_accuracy":
            s, meta = report["scores"], report["metadata"]
            lines += [
                "| Number | Value | 95% CI | n (cases x runs) | Per-run |",
                "|---|---:|---|---:|---|",
                f"| Router strict accuracy | {ev._fmt(s['accuracy'])} | {s['ci95']} | {s['n_cases']} x {s['runs']} | "
                f"{s['per_run_accuracy']} |",
                f"| Router lenient accuracy (ambiguous cases) | {ev._fmt(s['lenient_accuracy'])} | - | "
                f"{s['n_cases']} x {s['runs']} | - |",
                "",
                f"Model `{meta['model_alias']}` (resolved {', '.join(meta['resolved_model_ids']) or 'n/a'}); "
                f"status {report['status']}; cost ${cost:.5f}; report `{rel}`.",
            ]
        else:
            meta, exp = report["metadata"], report["experiment"]
            lines += [
                "| Metric | Mean | 95% CI | n (cases x runs) | Per-run means | Std of run means | Pass rate |",
                "|---|---:|---|---:|---|---:|---:|",
            ]
            for name, stats in report["aggregate"].items():
                lines.append(
                    f"| {name} | {ev._fmt(stats['mean'])} | {stats['ci95']} | {stats['n_cases']} x {stats['runs']} | "
                    f"{stats['per_run_means']} | {ev._fmt(stats['run_mean_std'])} | {ev._fmt(stats['pass_rate'])} |"
                )
            lines += [
                "",
                f"Goldens {exp['goldens_source']} ({', '.join(f['set'] + ' n=' + str(f['n']) + ' sha256 ' + f['sha256'][:12] for f in meta['goldens']['files'])}); "
                f"app `{exp['application_model']}` (resolved {', '.join(meta['app_resolved_model_ids']) or 'n/a'}), "
                f"judge `{exp['judge_model']}` (resolved {', '.join(meta['judge_resolved_model_ids']) or 'n/a'}); "
                f"retrieval {exp['retrieval_mode']}, rerank {exp['rerank_enabled']}, paper_only {meta['paper_only']}; "
                f"status {report['status']}; cost ${cost:.4f} (app ${exp['app_cost_usd']:.4f}, judge ${exp['judge_cost_usd']:.4f}); "
                f"report `{rel}`.",
            ]
        lines += ["", f"Command: `{step['command']}`", ""]
    lines += [f"**Total measured spend: ${total:.4f}** of ${budget:.2f}.", "",
              "Reproduce: `python scripts/reproduce_eval.py --budget-usd "
              f"{budget:.2f}`", ""]
    return lines


def main(argv: list[str] | None = None) -> int:
    ev.ensure_utf8_stdio()
    parser = argparse.ArgumentParser(description="Regenerate every reported evaluation number.")
    parser.add_argument("--budget-usd", type=float, default=DEFAULT_BUDGET_USD, help="Overall hard spend cap.")
    parser.add_argument("--dry-run", action="store_true", help="Print plans and estimates; no API calls.")
    parser.add_argument("--only", type=parse_only, default=list(STEP_ORDER),
                        help="Comma list of steps to run: router, rag, full (default: all).")
    args = parser.parse_args(argv)

    stamp = ev.utc_stamp()
    specs = step_specs(stamp)
    remaining = args.budget_usd
    steps_done: list[dict] = []
    estimate_total = 0.0

    for name in args.only:
        spec = specs[name]
        estimate = estimate_step(name)
        estimate_total += estimate or 0.0
        budget_for_step = max(0.0, remaining)
        cmd = step_command(spec, budget_for_step, args.dry_run)
        record = {"name": name, "title": spec["title"], "output": spec["output"], "command": display_command(cmd),
                  "estimate_usd": estimate}
        print(f"\n== {name}: {spec['title']} (estimate ${ev._fmt(estimate, 4)}, budget ${budget_for_step:.4f})")
        print(f"   $ {record['command']}")
        if not args.dry_run and remaining < min_case_estimate(name):
            record["skipped"] = f"remaining budget ${remaining:.4f} is below one case's estimate"
            print(f"   skipped: {record['skipped']}")
            steps_done.append(record)
            continue
        sys.stdout.flush()
        completed = subprocess.run(cmd, cwd=REPO_ROOT)
        if args.dry_run:
            remaining -= estimate or 0.0  # later steps see the budget left after this estimate
            if completed.returncode != 0:
                print(f"   dry run of {name} failed (exit {completed.returncode})")
                return completed.returncode
            continue
        if not Path(spec["output"]).exists():
            print(f"   step {name} failed before writing a report (exit {completed.returncode}); stopping.")
            return completed.returncode or 1
        spent = spent_from(Path(spec["output"]))
        remaining -= spent
        print(f"   step {name}: spent ${spent:.4f}, remaining ${remaining:.4f}, exit {completed.returncode}")
        steps_done.append(record)
        if completed.returncode != 0:
            print(f"   step {name} exited with {completed.returncode}; stopping before later steps.")
            break

    if args.dry_run:
        fits = "fits" if estimate_total <= args.budget_usd else "EXCEEDS"
        print(f"\n[dry-run] estimated total ${estimate_total:.4f} vs budget ${args.budget_usd:.2f}: {fits}. "
              f"Nothing was called or written.")
        return 0

    lines = summary_lines(steps_done, args.budget_usd, stamp)
    ev.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamped = ev.RESULTS_DIR / f"SUMMARY_{stamp}.md"
    stamped.write_text("\n".join(lines), encoding="utf-8")
    (ev.RESULTS_DIR / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nResults: {', '.join(str(s['output']) for s in steps_done if not s.get('skipped'))}")
    print(f"Summary: {ev.RESULTS_DIR / 'SUMMARY.md'} (copy: {stamped})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
