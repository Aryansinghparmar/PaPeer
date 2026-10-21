# A/B: reasoning in the final answer step (gpt-6-luna), 21 Oct 2026

**Question.** gpt-6-luna needs `reasoning_effort="none"` wherever function tools are bound, but the answer step binds none, so it *could* reason. Does reasoning there improve grounding?

**Pre-registered setup (fixed before any comparison run):**
- **Arms:** `ANSWER_REASONING_EFFORT` = `none` vs `low`. Everything else was identical: the agent, router and grader all use `none`, with hybrid + rerank and `paper_only=True`.
- **Judge:** gpt-6-luna, `reasoning_effort=low`, fixed before either arm ran (`DEFAULT_JUDGE_REASONING_EFFORT` in `evaluate.py`).
- **Data:** 17 goldens (10 synthetic + 7 curated, one 34-page PDF) × 3 runs = 51 case-runs per arm. Metrics: Faithfulness and Answer Relevancy. Local Qdrant v1.17.1.
- **Decision rule:** the higher mean Faithfulness wins, unless the 95% CIs overlap, in which case the cheaper setting (`none`) is kept.

**Commands:**
```bash
ANSWER_REASONING_EFFORT=none python evaluate.py --goldens all --metrics faithfulness,answer_relevancy --runs 3 --label ab_answer_none_x3 --budget-usd 0.40
ANSWER_REASONING_EFFORT=low  python evaluate.py --goldens all --metrics faithfulness,answer_relevancy --runs 3 --label ab_answer_low_x3  --budget-usd 0.40
```

## Results

| Arm | Faithfulness mean (95% CI) | Per-run means | Pass ≥ 0.7 | Answer Relevancy | Tool calls / turn | p50 / p90 latency | App + judge cost |
|---|---|---|---|---|---|---|---|
| `none` | **0.998** (0.994–1.000) | 0.994 / 1.000 / 1.000 | 51/51 | 0.997 (0.990–1.000) | 1.43 | 11.2 s / 22.6 s | $0.022 + $0.056 |
| `low` | **0.992** (0.977–1.000) | 0.980 / 1.000 / 0.995 | 50/51 | 1.000 (1.000–1.000) | 1.22 | 10.2 s / 16.6 s | $0.021 + $0.057 |

The paired per-case Faithfulness difference (low − none) over the 17 cases is **−0.006, with a 95% CI of [−0.021, +0.004]**. That interval includes zero.

**Decision:** the CIs overlap, so the default stays `none`. There is no measurable effect at this sample size.

## What the per-case reading showed
- **Unanswerable question** ("What was Moltbot's total revenue in 2025?"): all 6 runs across both arms answered that the context does not contain it, and scored Faithfulness 1.0 with no web text in the context. (Compare the August archive, where this case was answered from unrelated Tavily text.)
- **The one sub-threshold score** (`low`, synthetic:9, run 0, Faithfulness 0.667) was a **judge error**. The answer states that credentials are stored in plaintext with no encryption at rest, but the judge's reason claims the answer "does not reflect" exactly that. So 1 of the 102 Faithfulness judgments was clearly wrong.
- Tool calls per turn differ between the arms (1.43 vs 1.22) even though the agent configuration is identical. That is run-to-run variance: gpt-6-luna only accepts temperature 1.

## Caveats
- The test set is small (17 cases, one document) and the scores sit near the ceiling, so this A/B can only detect large effects.
- The app and the judge are the same model family, so self-preference bias is possible.
- **Git metadata:** both arms ran on the pre-publication development history, which was squashed into the initial public commit before publishing. The pipeline code that ran is identical to that commit; the only later differences under `backend/` and `api/` are a docstring and the removal of an unused helper. The reports' `git` field records this.
