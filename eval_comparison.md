# Retrieval A/B Evaluation — Baseline vs. Hybrid+Rerank

- **Document:** `documents/Openclaw_Research_Report.pdf`
- **Question set:** curated (5 cases), judge `gpt-5-mini`, app `gpt-5-mini`, threshold 0.7
- **Baseline retrieval:** mode=`dense`, rerank=False
- **Improved retrieval:** mode=`hybrid`, rerank=True, candidate_k=20 → top_n=4

## Metric scores (average, threshold 0.7)

| Metric | Baseline | Improved | Δ |
|---|---:|---:|---:|
| Contextual Precision | 0.8492 | 0.6795 | -0.1697 🔽 |
| Contextual Recall | 0.9333 | 0.8667 | -0.0666 🔽 |
| Contextual Relevancy | 0.5054 | 0.5315 | 0.0261 🔼 |
| Answer Relevancy | 0.95 | 1.0 | 0.05 🔼 |
| Faithfulness | 1.0 | 1.0 | 0.0 ▪ |

## Pass rates (cases meeting the 0.7 bar)

| Metric | Baseline | Improved |
|---|---:|---:|
| Contextual Precision | 5/5 | 3/5 |
| Contextual Recall | 4/5 | 3/5 |
| Contextual Relevancy | 2/5 | 1/5 |
| Answer Relevancy | 5/5 | 5/5 |
| Faithfulness | 5/5 | 5/5 |

## Efficiency & measured cost

| Measure | Baseline | Improved |
|---|---:|---:|
| Avg retrieved chunks / query | 13.2 | 11.4 |
| Avg end-to-end latency (s) | 22.85 | 38.19 |
| Application cost (USD) | 0.022051 | 0.023649 |
| DeepEval judge cost (USD) | 0.402667 | 0.357717 |
| **Total measured cost (USD)** | **0.424718** | **0.381366** |

> Costs are measured from OpenAI usage callbacks (application) and DeepEval's per-metric `evaluation_cost` (judges). Reranking and BM25 sparse retrieval run locally on CPU and add no OpenAI cost.

> **Note:** the "Improved" column above is the **post-answer-gating-fix** run. Read it with
> the variance caveat in Finding 2 below — the retrieval-metric deltas vs. baseline are within
> run-to-run noise at this sample size.

## Per-case detail (post-fix improved run)

| Case | Faithfulness | Answer Rel. | Precision | Recall |
|---|---:|---:|---:|---:|
| factual | 1.000 | 1.000 | 0.824 | 1.000 |
| numeric | 1.000 | 1.000 | 0.555 | 1.000 |
| multi-section | 1.000 | 1.000 | 1.000 | 0.667 |
| security | 1.000 | 1.000 | 0.907 | 1.000 |
| unanswerable | 1.000 | 1.000 | 0.111 | 0.667 |

## Findings

### 1. The answer-gating fix is validated (repeatable, causal)
Before the fix, the `security` case retrieved correct context (Precision/Recall 1.0) but the
answer node emitted the canned *"I wasn't able to find relevant information…"*, so Faithfulness
was 0.667. After the fix (generate from `retrieved_docs` whenever non-empty), **Faithfulness is
1.000 on all five cases** and **Answer Relevancy is 1.000 on all five** — the false refusal is
gone. This is the reliable, causal win of this round.

### 2. Retrieval metrics are NOT reliable at n=5 single-run (honest caveat)
The answer-gating fix does not touch retrieval, yet two improved runs of the *same*
hybrid+rerank config disagree sharply on the retrieval metrics:

| Metric | Improved run 1 | Improved run 2 (post-fix) | Baseline |
|---|---:|---:|---:|
| Contextual Precision | 0.922 | 0.680 | 0.849 |
| Contextual Recall | 1.000 | 0.867 | 0.933 |

The improved runs **straddle the baseline**, so at this sample size hybrid+rerank shows **no
statistically distinguishable retrieval-quality change** — the swings are run-to-run variance
from the non-deterministic agent retrieval loop (the LLM chooses `k`, may rewrite the query,
and accumulates a variable chunk set) plus LLM-judge variance. **An earlier "precision/recall
improved" reading was within noise and should not be claimed from a single run.**

### 3. The `unanswerable` case structurally depresses contextual metrics
For "What was Moltbot's revenue?" (not in the document) the retrieved chunks are *correctly*
irrelevant, so Precision (0.111) and Relevancy are low by construction while Faithfulness and
Answer Relevancy stay perfect (the system correctly reports the info isn't available). This
case tests robustness, not retrieval quality, and drags the contextual averages down.

## What is reliably true vs. not

- **Reliable:** the answer-gating fix eliminates false refusals — Faithfulness and Answer
  Relevancy are 1.0 across all cases. Hybrid+rerank modestly reduces retrieved-chunk count
  (13.2 → 11.4 here) but adds latency.
- **Not reliable at n=5:** any fine precision/recall/relevancy delta between dense and
  hybrid+rerank — it is within run-to-run variance.

## Next step for a defensible retrieval claim (deferred — budget)
Run each config multiple times (e.g., 3–5 seeds) and average, and/or make retrieval
deterministic (agent `temperature=0`, fixed `k`, disable query rewrite) to shrink variance;
then the retrieval deltas become interpretable.

