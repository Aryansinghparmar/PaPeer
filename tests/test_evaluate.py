"""Offline tests for the evaluation harness (no API, Qdrant or judge calls)."""

import json
import random
import socket
import sys
import types
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

import evaluate as ev


def _result(input_text, score=1.0, cost=0.001, name="Faithfulness"):
    metric = SimpleNamespace(name=name, score=score, success=score >= 0.7, threshold=0.7,
                             evaluation_cost=cost, evaluation_model="gpt-6-luna", reason="r", error=None)
    return SimpleNamespace(input=input_text, success=metric.success, metrics_data=[metric])


def test_results_are_mapped_to_cases_by_input_not_order():
    cases, _ = ev.load_goldens("curated")
    results = [_result(case["input"]) for case in cases]
    random.Random(7).shuffle(results)  # DeepEval returns completion order

    paired = ev.map_results_to_cases(results, cases)

    assert [case["case_id"] for case, _ in paired] == [case["case_id"] for case in cases]
    for case, result in paired:
        assert result.input == case["input"]
    revenue = next(case for case, _ in paired if "revenue" in case["input"])
    assert revenue["category"] == "unanswerable"


def test_mapping_fails_loudly_on_missing_or_extra_results():
    cases, _ = ev.load_goldens("curated")
    with pytest.raises(KeyError):
        ev.map_results_to_cases([_result(c["input"]) for c in cases[1:]], cases)
    with pytest.raises(KeyError):
        ev.map_results_to_cases([_result(c["input"]) for c in cases] + [_result("stray")], cases)


def test_load_goldens_all_combines_both_sets_with_hashes():
    cases, sources = ev.load_goldens("all")
    assert [s["set"] for s in sources] == ["synthetic", "curated"]
    assert len(cases) == sum(s["n"] for s in sources) == 17
    assert all(len(s["sha256"]) == 64 for s in sources)
    assert len({c["case_id"] for c in cases}) == len(cases)


def test_parse_metrics_validates_names():
    assert ev.parse_metrics("faithfulness, answer_relevancy") == ["faithfulness", "answer_relevancy"]
    with pytest.raises(Exception):
        ev.parse_metrics("faithfulness,bogus")


def _tests_for_aggregation():
    scores = {0: [1.0, 0.5, 0.9], 1: [0.8, 0.6, 1.0]}
    tests = []
    for run, run_scores in scores.items():
        for i, score in enumerate(run_scores):
            tests.append({"run_index": run, "case_id": f"c{i}", "input": f"q{i}",
                          "metrics": [{"name": "Faithfulness", "score": score, "passed": score >= 0.7,
                                       "evaluation_cost": 0.01}]})
    return tests


def test_aggregate_reports_mean_runs_std_pass_rate_and_ci():
    agg = ev.aggregate(_tests_for_aggregation())
    f = agg["per_metric"]["Faithfulness"]
    assert f["mean"] == pytest.approx(0.8)
    assert f["per_run_means"] == [0.8, 0.8]
    assert f["run_mean_std"] == 0.0
    assert (f["passed"], f["n"], f["n_cases"], f["runs"]) == (4, 6, 3, 2)
    assert f["pass_rate"] == pytest.approx(4 / 6, abs=1e-4)
    low, high = f["ci95"]
    assert 0.55 <= low <= f["mean"] <= high <= 0.95
    assert agg["judge_cost_usd"] == pytest.approx(0.06)


def test_bootstrap_is_deterministic_for_a_seed():
    values = [1, 1, 1, 0.9375, 0.964, 0.944, 1, 1, 0.963, 1]
    first = ev.bootstrap_ci(values)
    assert first == ev.bootstrap_ci(values)
    assert ev.aggregate(_tests_for_aggregation()) == ev.aggregate(_tests_for_aggregation())
    assert first[0] < sum(values) / len(values) < first[1]
    assert ev.bootstrap_ci([0.5]) == [0.5, 0.5]
    assert ev.bootstrap_ci([]) is None


def test_aggregate_tolerates_old_report_format():
    old = [{"input": "q", "metrics": [{"name": "Faithfulness", "score": 0.9, "passed": True}]}]
    f = ev.aggregate(old)["per_metric"]["Faithfulness"]
    assert f["avg_score"] == 0.9 and f["n"] == 1


def test_budget_guard_stops_before_exceeding_the_cap():
    guard = ev.BudgetGuard(0.05, prior_case_usd=0.02, prior_judge_usd=0.015)
    started = 0
    while guard.can_start_case():
        guard.charge(0.016)
        guard.record_case(0.016, 0.012)
        started += 1
        assert guard.spent_usd <= guard.budget_usd
    assert started == 3
    assert guard.truncated and "stopped before next case" in guard.stop_reason
    assert guard.spent_usd + guard.next_case_estimate() > guard.budget_usd


def test_budget_guard_uses_prior_until_first_measurement():
    guard = ev.BudgetGuard(0.01, prior_case_usd=0.02, prior_judge_usd=0.01)
    assert not guard.can_start_case()  # the prior alone would exceed the cap
    assert guard.to_dict()["truncated"] is True


def test_app_usage_fails_loudly_on_unpriced_model():
    usage = {"some-unpriced-model": {"input_tokens": 10, "output_tokens": 5}}
    with pytest.raises(ev.CostAccountingError):
        ev.app_usage(usage)
    priced = ev.app_usage({"gpt-6-luna-2026-10-01": {"input_tokens": 1_000_000, "output_tokens": 0,
                                                     "input_token_details": {"cache_read": 10}}})
    assert priced["cost_usd"] == pytest.approx(0.10)
    assert priced["resolved_model_ids"] == ["gpt-6-luna-2026-10-01"]


def test_trace_case_counts_tool_calls_and_sources():
    state = {
        "route": "retrieve",
        "messages": [AIMessage(content="", tool_calls=[{"name": "retrieve_from_vectorstore", "args": {}, "id": "1"}]),
                     AIMessage(content="", tool_calls=[{"name": "web_search", "args": {}, "id": "2"}])],
        "retrieved_docs": [Document(page_content="a", metadata={"title": "p", "start_index": 0}),
                           Document(page_content="a", metadata={"title": "p", "start_index": 0}),
                           Document(page_content="w", metadata={"url": "https://x", "title": "Web"})],
    }
    trace = ev.trace_case(state)
    assert trace["tool_calls"] == {"vectorstore": 1, "web": 1, "other": 0}
    assert trace["retrieved_doc_sources"] == {"paper": 2, "web": 1}
    assert (trace["retrieved_count"], trace["unique_chunk_count"]) == (3, 2)


def test_refuses_to_overwrite_existing_results(tmp_path):
    out = tmp_path / "report.json"
    out.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        ev.main(["--goldens", "curated", "--dry-run", "--output", str(out)])
    assert out.read_text(encoding="utf-8") == "{}"
    ev.main(["--goldens", "curated", "--dry-run", "--output", str(out), "--overwrite"])


def test_default_output_never_targets_repo_root():
    path = ev.default_output_path("rag", "20260101T000000Z")
    assert path.parent == ev.RESULTS_DIR and path.name == "rag_20260101T000000Z.json"


def _block_network_and_backends(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted during dry run")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    for name in ("backend.vector_store", "backend.rag_graph", "backend.paper_loader",
                 "backend.telemetry", "deepeval"):
        monkeypatch.setitem(sys.modules, name, None)  # any import now raises ImportError


def test_dry_run_makes_no_network_call_and_imports_no_client(monkeypatch, tmp_path, capsys):
    _block_network_and_backends(monkeypatch)
    plan = ev.main(["--goldens", "all", "--metrics", "faithfulness,answer_relevancy", "--runs", "3",
                    "--dry-run", "--output", str(tmp_path / "x.json")])
    assert plan["dry_run"] and plan["plan"]["case_runs"] == 51
    assert plan["plan"]["estimated_total_usd"] > 0
    assert not (tmp_path / "x.json").exists()
    assert "estimated total" in capsys.readouterr().out


def test_reproduce_dry_run_is_offline(monkeypatch, capsys):
    _block_network_and_backends(monkeypatch)
    sys.path.insert(0, str(ev.REPO_ROOT / "scripts"))
    try:
        import reproduce_eval as repro
    finally:
        sys.path.pop(0)
    calls = []
    monkeypatch.setattr(repro.subprocess, "run",
                        lambda cmd, cwd=None: calls.append(cmd) or SimpleNamespace(returncode=0))
    assert repro.main(["--dry-run"]) == 0
    assert len(calls) == 3 and all("--dry-run" in cmd for cmd in calls)
    assert "estimated total" in capsys.readouterr().out


class _FakeGraph:
    def __init__(self):
        self.inputs = []

    def invoke(self, state, config):
        self.inputs.append(state)
        config["callbacks"][0].usage_metadata["gpt-6-luna"] = {"input_tokens": 4000, "output_tokens": 1000,
                                                              "total_tokens": 5000}
        return {
            "answer": "grounded answer",
            "route": "retrieve",
            "retrieval_attempts": 1,
            "messages": [AIMessage(content="", tool_calls=[{"name": "retrieve_from_vectorstore", "args": {}, "id": "1"}])],
            "retrieved_docs": [Document(page_content="chunk", metadata={"title": "paper", "start_index": 0})],
        }


def test_run_stops_before_budget_and_records_truncation(monkeypatch, tmp_path):
    graph = _FakeGraph()
    fakes = {
        "backend.paper_loader": {"load_document": lambda path: ["doc"]},
        "backend.rag_graph": {"build_graph": lambda db_path=None: graph},
        "backend.telemetry": {"record_run": lambda **kw: {"estimated_cost_usd": 0.0009}},
        "backend.vector_store": {"add_paper": lambda docs, sid: None, "delete_session_collection": lambda sid: None},
    }
    for name, attrs in fakes.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    ledgers = []

    class CapturingLedger(ev.JudgeLedger):
        def __init__(self):
            super().__init__()
            ledgers.append(self)

    def fake_judge(test_cases, metrics):
        usage = SimpleNamespace(prompt_tokens=10_000, completion_tokens=10_000,
                                prompt_tokens_details=None, completion_tokens_details=None)
        ledgers[-1].add("faithfulness", SimpleNamespace(usage=usage, model="gpt-6-luna"), 1e-7, 5e-7)
        return [_result(tc.input, score=0.9, cost=0.006) for tc in reversed(test_cases)]

    monkeypatch.setattr(ev, "JudgeLedger", CapturingLedger)
    monkeypatch.setattr(ev, "register_judge_model", lambda name: {"model": name})
    monkeypatch.setattr(ev, "build_metrics", lambda *a, **k: [])
    monkeypatch.setattr(ev, "judge_test_cases", fake_judge)

    out = tmp_path / "run.json"
    report = ev.main(["--goldens", "curated", "--metrics", "faithfulness", "--budget-usd", "0.02",
                      "--model", "gpt-6-luna", "--output", str(out)])

    assert report["status"] == "truncated" and report["budget"]["truncated"]
    assert report["budget"]["spent_usd"] <= 0.02
    assert report["experiment"]["case_runs_completed"] == 2
    assert all(state["paper_only"] is True for state in graph.inputs)
    first = report["tests"][0]
    assert first["category"] == "factual" and first["trace"]["tool_calls"]["vectorstore"] == 1
    assert first["app"]["cost_usd"] == pytest.approx(0.0009)
    assert first["judge"]["cost_usd"] == pytest.approx(0.006)
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["metadata"]["paper_only"] is True and saved["metadata"]["git"]["sha"]
    assert out.with_suffix(".md").exists()


def test_compare_accepts_old_and_new_formats(tmp_path):
    old = {"experiment": {"goldens_source": "curated", "case_count": 1, "judge_model": "j",
                          "application_model": "a", "metric_threshold": 0.7, "retrieval_mode": "dense",
                          "rerank_enabled": False, "avg_retrieved_chunks": 1, "avg_latency_seconds": 1,
                          "app_cost_usd": 0, "judge_cost_usd": 0, "total_cost_usd": 0},
           "aggregate": {"Faithfulness": {"avg_score": 0.9, "passed": 1, "n": 1}}}
    new = {"experiment": dict(old["experiment"], retrieval_mode="hybrid"),
           "aggregate": ev.aggregate(_tests_for_aggregation())["per_metric"],
           "metadata": {"git": {"sha": "abc"}}}
    (tmp_path / "a.json").write_text(json.dumps(old), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps(new), encoding="utf-8")
    ev.write_comparison(str(tmp_path / "a.json"), str(tmp_path / "b.json"), str(tmp_path / "c.md"))
    text = (tmp_path / "c.md").read_text(encoding="utf-8")
    assert "| Faithfulness | 0.9 | - | 0.8 |" in text


def test_run_aborts_after_consecutive_failures_and_saves_report(monkeypatch, tmp_path):
    class BrokenGraph:
        def invoke(self, state, config):
            raise RuntimeError("pipeline down")

    fakes = {
        "backend.paper_loader": {"load_document": lambda path: ["doc"]},
        "backend.rag_graph": {"build_graph": lambda db_path=None: BrokenGraph()},
        "backend.telemetry": {"record_run": lambda **kw: {}},
        "backend.vector_store": {"add_paper": lambda docs, sid: None, "delete_session_collection": lambda sid: None},
    }
    for name, attrs in fakes.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(ev, "register_judge_model", lambda name: {"model": name})
    monkeypatch.setattr(ev, "build_metrics", lambda *a, **k: [])

    out = tmp_path / "broken.json"
    with pytest.raises(RuntimeError, match="consecutive case failures"):
        ev.main(["--goldens", "curated", "--metrics", "faithfulness", "--model", "gpt-6-luna", "--output", str(out)])
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["status"] == "aborted" and len(saved["tests"]) == ev.MAX_CONSECUTIVE_FAILURES
    assert all("pipeline down" in t["error"] for t in saved["tests"])


def test_reports_record_repo_relative_paths_only(tmp_path):
    inside = ev.REPO_ROOT / "results" / "x.json"
    assert ev.repo_relative(inside) == "results/x.json"
    assert ev.repo_relative(tmp_path / "elsewhere.json") == "elsewhere.json"
    argv = ["ev.py", "--output", str(inside), "--runs", "3"]
    assert ev.relative_argv(argv) == ["ev.py", "--output", "results/x.json", "--runs", "3"]
    assert str(ev.REPO_ROOT) not in ev.command_line(argv)
