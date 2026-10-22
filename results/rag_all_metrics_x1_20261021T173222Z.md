# RAG evaluation: rag_all_metrics_x1

- **Status:** complete
- **Goldens:** all (17 cases) x 1 run(s); 17 case-runs judged
- **App model:** `gpt-6-luna` (resolved: gpt-6-luna)
- **Judge model:** `gpt-6-luna` (resolved: gpt-6-luna; reasoning_effort=low)
- **Retrieval:** mode=`hybrid`, rerank=True, candidate_k=20, top_n=4, paper_only=True
- **Git:** `0e654d2881d96adb0753d29ff11b9f2926209f7b`
- **Measured cost:** app $0.0070 + judge $0.0669 = **$0.0739** (budget $0.92)
- **Command:** `python evaluate.py --goldens all --metrics faithfulness,answer_relevancy,contextual_precision,contextual_recall,contextual_relevancy --runs 1 --label rag_all_metrics_x1 --budget-usd 0.9166 --output results/rag_all_metrics_x1_20261021T173222Z.json`

| Metric | Mean | 95% CI (cases) | Per-run means | Std of run means | Pass (>= 0.7) | n (cases x runs) |
|---|---:|---|---|---:|---:|---:|
| Faithfulness | 0.993 | [0.978, 1.000] | 0.993 | - | 17/17 | 17 x 1 |
| Answer Relevancy | 1.000 | [1.000, 1.000] | 1.000 | - | 17/17 | 17 x 1 |
| Contextual Precision | 0.964 | [0.930, 0.990] | 0.964 | - | 17/17 | 17 x 1 |
| Contextual Recall | 0.937 | [0.867, 0.991] | 0.937 | - | 16/17 | 17 x 1 |
| Contextual Relevancy | 0.533 | [0.411, 0.650] | 0.533 | - | 6/17 | 17 x 1 |

Avg retrieved chunks 6.18 (unique 6.18), avg latency 10.69 s, web tool calls 0, case errors 0.

> CI covers case sampling only; judge and pipeline variance show up in the per-run means. DeepEval Faithfulness counts claims the context does not contradict as faithful, so it is a grounding signal, not answer correctness.
