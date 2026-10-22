# Evaluation summary

Generated 2026-10-21T18:13:15+00:00 by `python scripts/reproduce_eval.py` at git `0e654d2881d96adb0753d29ff11b9f2926209f7b`. Overall budget $1.00.

Every number below is copied from the JSON report next to it; nothing is typed by hand. CIs are case-level bootstrap 95% intervals (case sampling only); run-to-run spread is the std of per-run means. DeepEval Faithfulness counts claims the context does not contradict as faithful, so it is a grounding signal, not correctness. App and judge are the same model (self-preference risk).

## Router accuracy (3 runs)

| Number | Value | 95% CI | n (cases x runs) | Per-run |
|---|---:|---|---:|---|
| Router strict accuracy | 1.000 | [1.0, 1.0] | 30 x 3 | [1.0, 1.0, 1.0] |
| Router lenient accuracy (ambiguous cases) | 1.000 | - | 30 x 3 | - |

Model `gpt-6-luna` (resolved gpt-6-luna); status complete; cost $0.00313; report `results/router_x3_20261021T173222Z.json`.

Command: `python eval_router.py --runs 3 --label router_x3 --budget-usd 1.0000 --output results/router_x3_20261021T173222Z.json`

## Faithfulness + Answer Relevancy, goldens all, 3 runs

| Metric | Mean | 95% CI | n (cases x runs) | Per-run means | Std of run means | Pass rate |
|---|---:|---|---:|---|---:|---:|
| Faithfulness | 0.992 | [0.9745, 1.0] | 17 x 3 | [0.9941, 0.9853, 0.9951] | 0.005 | 1.000 |
| Answer Relevancy | 0.999 | [0.9961, 1.0] | 17 x 3 | [1.0, 1.0, 0.9961] | 0.002 | 1.000 |

Goldens all (synthetic n=10 sha256 624153833c4a, curated n=7 sha256 6f4fa77f7624); app `gpt-6-luna` (resolved gpt-6-luna), judge `gpt-6-luna` (resolved gpt-6-luna); retrieval hybrid, rerank True, paper_only True; status complete; cost $0.0802 (app $0.0222, judge $0.0580); report `results/rag_core_x3_20261021T173222Z.json`.

Command: `python evaluate.py --goldens all --metrics faithfulness,answer_relevancy --runs 3 --label rag_core_x3 --budget-usd 0.9969 --output results/rag_core_x3_20261021T173222Z.json`

## All five DeepEval metrics, goldens all, 1 run

| Metric | Mean | 95% CI | n (cases x runs) | Per-run means | Std of run means | Pass rate |
|---|---:|---|---:|---|---:|---:|
| Faithfulness | 0.993 | [0.9779, 1.0] | 17 x 1 | [0.9926] | - | 1.000 |
| Answer Relevancy | 1.000 | [1.0, 1.0] | 17 x 1 | [1.0] | - | 1.000 |
| Contextual Precision | 0.964 | [0.9305, 0.9904] | 17 x 1 | [0.964] | - | 1.000 |
| Contextual Recall | 0.937 | [0.8667, 0.991] | 17 x 1 | [0.9374] | - | 0.941 |
| Contextual Relevancy | 0.533 | [0.4111, 0.6501] | 17 x 1 | [0.533] | - | 0.353 |

Goldens all (synthetic n=10 sha256 624153833c4a, curated n=7 sha256 6f4fa77f7624); app `gpt-6-luna` (resolved gpt-6-luna), judge `gpt-6-luna` (resolved gpt-6-luna); retrieval hybrid, rerank True, paper_only True; status complete; cost $0.0739 (app $0.0070, judge $0.0669); report `results/rag_all_metrics_x1_20261021T173222Z.json`.

Command: `python evaluate.py --goldens all --metrics faithfulness,answer_relevancy,contextual_precision,contextual_recall,contextual_relevancy --runs 1 --label rag_all_metrics_x1 --budget-usd 0.9166 --output results/rag_all_metrics_x1_20261021T173222Z.json`

**Total measured spend: $0.1573** of $1.00.

Reproduce: `python scripts/reproduce_eval.py --budget-usd 1.00`
