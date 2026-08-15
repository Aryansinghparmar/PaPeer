# Retrieval A/B Evaluation — Baseline vs. Hybrid+Rerank

- **Document:** `documents/Openclaw_Research_Report.pdf`
- **Question set:** curated (5 cases), judge `gpt-5-mini`, app `gpt-5-mini`, threshold 0.7
- **Baseline retrieval:** mode=`dense`, rerank=False
- **Improved retrieval:** mode=`hybrid`, rerank=True, candidate_k=20 → top_n=4

## Metric scores (average, threshold 0.7)

| Metric | Baseline | Improved | Δ |
|---|---:|---:|---:|
| Contextual Precision | 0.8492 | 0.9219 | 0.0727 🔼 |
| Contextual Recall | 0.9333 | 1.0 | 0.0667 🔼 |
| Contextual Relevancy | 0.5054 | 0.5117 | 0.0063 🔼 |
| Answer Relevancy | 0.95 | 1.0 | 0.05 🔼 |
| Faithfulness | 1.0 | 0.9333 | -0.0667 🔽 |

## Pass rates (cases meeting the 0.7 bar)

| Metric | Baseline | Improved |
|---|---:|---:|
| Contextual Precision | 5/5 | 4/5 |
| Contextual Recall | 4/5 | 5/5 |
| Contextual Relevancy | 2/5 | 2/5 |
| Answer Relevancy | 5/5 | 5/5 |
| Faithfulness | 5/5 | 4/5 |

## Efficiency & measured cost

| Measure | Baseline | Improved |
|---|---:|---:|
| Avg retrieved chunks / query | 13.2 | 8.8 |
| Avg end-to-end latency (s) | 22.85 | 35.36 |
| Application cost (USD) | 0.022051 | 0.02476 |
| DeepEval judge cost (USD) | 0.402667 | 0.300821 |
| **Total measured cost (USD)** | **0.424718** | **0.325581** |

> Costs are measured from OpenAI usage callbacks (application) and DeepEval's per-metric `evaluation_cost` (judges). Reranking and BM25 sparse retrieval run locally on CPU and add no OpenAI cost.

## Per-case Contextual Relevancy (the metric we targeted)

| Case | Baseline | Improved | Note |
|---|---:|---:|---|
| factual | 0.403 | 0.370 | ~flat |
| numeric | 0.844 | 0.522 | regressed |
| multi-section synthesis | 0.467 | **0.924** | large gain |
| security | 0.813 | **0.000** | regressed — see below |
| unanswerable | 0.000 | **0.743** | large gain |

## Findings & honest interpretation

**What clearly improved.** Hybrid retrieval + cross-encoder reranking lifted
**Contextual Precision** (0.849 → 0.922) and drove **Contextual Recall to a perfect
1.000 (5/5)** and **Answer Relevancy to 1.000 (5/5)**, while cutting average retrieved
chunks per query from **13.2 → 8.8** (less noise, cheaper to judge). This confirms the
first-stage upgrade puts the right evidence in front of the model more reliably.

**What did not improve, and why (no overclaiming).** Average **Contextual Relevancy
was essentially flat** (0.505 → 0.512). The per-case view shows this is a wash of two
large gains (multi-section, unanswerable) against two regressions (numeric, security),
not a uniform lift. The **security case regressed sharply** (relevancy 0.813 → 0.000,
faithfulness 1.000 → 0.667) for a diagnosable reason: even though retrieval was correct
(that case's Precision and Recall were both 1.000), the answer node emitted the canned
*"I wasn't able to find relevant information in the uploaded papers"* fallback. That
fallback fires whenever the binary relevancy gate returns false after a query rewrite —
**regardless of whether good chunks were actually retrieved** — and the judge correctly
penalised the contradiction. So the reranking exposed a pre-existing weakness in the
**answer-gating / query-rewrite logic**, not a retrieval failure.

**The real conclusion.** The retrieval upgrade is a genuine but *partial* win: precision,
recall, answer-relevancy, and chunk efficiency improved; relevancy is gated by a
downstream logic bug. The clear next step is to stop trusting the binary relevancy gate
over documents that are actually present (generate from retrieved chunks instead of the
canned fallback when `retrieved_docs` is non-empty), then re-run this same A/B.

**Cost of latency.** Average end-to-end latency rose (22.85s → 35.36s) because the
CPU cross-encoder reranks a 20-candidate pool per call. This is a quality/latency
trade-off that can be tuned (smaller candidate pool, batching, or a lighter reranker).

*This comparison is on a single document with 5 hand-audited questions. It is a
controlled, reproducible signal — not a broad benchmark. Broadening to multiple papers
and more questions is deferred to a future, better-funded session.*

