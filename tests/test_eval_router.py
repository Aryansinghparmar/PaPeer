"""Offline tests for the router accuracy eval (fake chain, no API calls)."""

import json
from types import SimpleNamespace

import pytest

import eval_router as er
import evaluate as ev


class FakeChain:
    """Routes by a lookup table and books fake token usage on the callback."""

    def __init__(self, routes, report_usage=True):
        self.routes = routes
        self.report_usage = report_usage
        self.calls = 0

    def invoke(self, inputs, config):
        self.calls += 1
        if self.report_usage:
            config["callbacks"][0].usage_metadata["gpt-6-luna"] = {
                "input_tokens": 400, "output_tokens": 10, "total_tokens": 410}
        return SimpleNamespace(route=self.routes[inputs["query"]])


def _routes_with_mistakes(cases):
    routes = {c["query"]: c["expected_route"] for c in cases}
    by_id = {c["id"]: c for c in cases}
    routes[by_id["verify-07"]["query"]] = "retrieve"       # verify misrouted to retrieve
    routes[by_id["direct-05"]["query"]] = "retrieve"       # ambiguous: acceptable
    routes[by_id["live-01"]["query"]] = "direct_answer"    # live misrouted to direct
    return routes


def test_router_cases_file_is_well_formed():
    cases = er.load_cases()
    counts = {}
    for case in cases:
        counts[case["expected_route"]] = counts.get(case["expected_route"], 0) + 1
    assert len(cases) == 30
    assert counts == {"retrieve": 15, "verify_claim": 8, "direct_answer": 7}
    assert any("verify" not in c["query"].lower() for c in cases if c["expected_route"] == "verify_claim")
    assert sum(1 for c in cases if c.get("note")) >= 3


def test_score_predictions_confusion_matrix_and_per_route_metrics():
    cases = er.load_cases()
    routes = _routes_with_mistakes(cases)
    predictions = [{"run_index": 0, "case_id": c["id"], "predicted": routes[c["query"]]} for c in cases]
    s = er.score_predictions(cases, predictions)

    assert s["accuracy"] == pytest.approx(27 / 30, abs=1e-4)
    assert s["lenient_accuracy"] == pytest.approx(28 / 30, abs=1e-4)
    cm = s["confusion_matrix"]
    assert cm["verify_claim"]["retrieve"] == 1 and cm["verify_claim"]["verify_claim"] == 7
    assert cm["direct_answer"]["retrieve"] == 1 and cm["retrieve"]["direct_answer"] == 1
    assert sum(sum(row.values()) for row in cm.values()) == 30
    assert s["per_route"]["verify_claim"] == {"precision": 1.0, "recall": 0.875, "f1": pytest.approx(0.9333, abs=1e-4),
                                             "support": 8, "predicted": 7}
    assert s["per_route"]["retrieve"]["precision"] == pytest.approx(14 / 16, abs=1e-4)
    assert {m["case_id"] for m in s["misroutes"]} == {"verify-07", "direct-05", "live-01"}
    assert s["ci95"] == er.score_predictions(cases, predictions)["ci95"]  # deterministic


def test_errors_and_unstable_cases_are_reported():
    cases = er.load_cases()[:2]
    predictions = [
        {"run_index": 0, "case_id": cases[0]["id"], "predicted": "retrieve"},
        {"run_index": 1, "case_id": cases[0]["id"], "predicted": "direct_answer"},
        {"run_index": 0, "case_id": cases[1]["id"], "predicted": None, "error": "boom"},
        {"run_index": 1, "case_id": cases[1]["id"], "predicted": "retrieve"},
    ]
    s = er.score_predictions(cases, predictions)
    assert s["errors"] == 1 and s["confusion_matrix"]["retrieve"]["error"] == 1
    assert s["per_run_accuracy"] == [0.5, 0.5] and s["run_accuracy_std"] == 0.0
    assert set(s["unstable_case_ids"]) == {cases[0]["id"], cases[1]["id"]}


def test_router_eval_end_to_end_with_fake_chain(tmp_path):
    cases = er.load_cases()
    chain = FakeChain(_routes_with_mistakes(cases))
    out = tmp_path / "router.json"
    report = er.main(["--runs", "2", "--output", str(out)], chain=chain)

    assert chain.calls == 60
    assert report["status"] == "complete"
    assert report["scores"]["n"] == 60 and report["scores"]["runs"] == 2
    assert report["cost_usd"] == pytest.approx(60 * (400 * 0.10 + 10 * 0.50) / 1e6)
    assert report["metadata"]["resolved_model_ids"] == ["gpt-6-luna"]
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["scores"]["accuracy"] == report["scores"]["accuracy"]
    md = out.with_suffix(".md").read_text(encoding="utf-8")
    assert "Confusion matrix" in md and "verify-07" in md


def test_router_eval_budget_guard_truncates(tmp_path):
    cases = er.load_cases()
    chain = FakeChain({c["query"]: c["expected_route"] for c in cases})
    per_call = (400 * 0.10 + 10 * 0.50) / 1e6
    report = er.main(["--runs", "3", "--budget-usd", str(per_call * 10.5), "--output", str(tmp_path / "r.json")],
                     chain=chain)
    assert report["status"] == "truncated"
    assert report["budget"]["spent_usd"] <= per_call * 10.5
    assert chain.calls == 10


def test_router_eval_refuses_unmeasured_cost(tmp_path):
    cases = er.load_cases()
    chain = FakeChain({c["query"]: c["expected_route"] for c in cases}, report_usage=False)
    with pytest.raises(ev.CostAccountingError):
        er.main(["--output", str(tmp_path / "r.json")], chain=chain)


def test_router_dry_run_makes_no_call(tmp_path, capsys):
    chain = FakeChain({})
    result = er.main(["--runs", "3", "--dry-run", "--output", str(tmp_path / "r.json")], chain=chain)
    assert result["dry_run"] and chain.calls == 0
    assert result["plan"]["calls"] == 90
    assert not (tmp_path / "r.json").exists()
    assert "estimated total" in capsys.readouterr().out
