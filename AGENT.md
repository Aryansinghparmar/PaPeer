# Papeer Implementation and Verification Record

This file is the working record for the Papeer capstone project. It contains the
implementation plan, cost controls, evaluation rules, risks, and progress marks.

Progress marks:

- `[x]` Done and verified
- `[~]` In progress
- `[ ]` Not started
- `[!]` Blocked or needs a user decision

## Project objective

Turn the course capstone into a reliable and measurable RAG research assistant
that can be explained honestly in a CV and in technical interviews.

The final project must report measured facts. It must not claim that a feature,
metric, deployment, or observability system exists unless it is implemented and
verified.

## Safety and cost rules

- Never print or commit API keys.
- Never place secrets inside a Docker image.
- Keep `.env` local and ignored by Git.
- Use `text-embedding-3-small` unless a measured experiment proves that another
  embedding model is worth the extra cost.
- Use `gpt-5-mini` as the default application and evaluation model unless a
  measured quality comparison justifies `gpt-5.4-mini`.
- Do not use GPT-5.4, GPT-5.5, Pro, or other expensive models for routine work.
- Run one cost probe before a full DeepEval run.
- Record model name, token usage, latency, and estimated cost for every measured
  experiment.
- Use Tavily basic search by default.
- Keep the first Tavily test stage at or below 50 credits unless a new decision
  is reported first.
- Clean evaluation collections from Qdrant when it is safe to do so.
- Do not create Azure or AWS resources without checking the expected cost and
  receiving a clear deployment decision.

## Current project facts

- UI: Streamlit.
- Workflow: LangGraph.
- Vector database: Qdrant Cloud.
- Embeddings: OpenAI `text-embedding-3-small`, 1,536 dimensions.
- Local embedding cache: `CacheBackedEmbeddings` with `LocalFileStore`.
- Checkpoint store: SQLite.
- Web search: Tavily.
- Current evaluation tool: DeepEval.
- Current source retrieval: dense Qdrant similarity search.
- Current source does not contain hybrid search or a reranker.
- Current source does not contain Prometheus or Grafana.
- Current source contains a `langsmith` dependency but no completed LangSmith
  tracing configuration.

## Existing baseline

The repository contains 10 saved DeepEval cases from one report. The saved
averages are:

| Metric | Average | Test pass count |
|---|---:|---:|
| Contextual Precision | 0.978 | 10/10 |
| Contextual Recall | 1.000 | 10/10 |
| Contextual Relevancy | 0.576 | 5/10 |
| Answer Relevancy | 0.991 | 10/10 |
| Faithfulness | 0.981 | 10/10 |

Older notes mention a different 64-question experiment with hybrid search and a
reranker. That experiment is not part of the current repository. It must not be
reported as a result of this code without reproducing it.

The `eval_results.json` file in the repo holds the ORIGINAL student synthetic run
(10 auto-generated goldens, earlier code). It is kept for reference only and must
not be presented as a result of the current retrieval code.

## Session 2 — Retrieval upgrade + controlled A/B (2026-08-15, verified)

A second engineer took over here. This session added a production-style retrieval
upgrade and measured it against the original retrieval with a controlled A/B on a
hand-curated, human-audited question set (`goldens_curated.json`, 5 cases on the
same document). All numbers below are measured, not estimated.

**What was built (all retrieval compute is local/CPU, no OpenAI cost):**
- `backend/reranker.py`: local cross-encoder rerank (`fastembed`,
  `Xenova/ms-marco-MiniLM-L-6-v2`). Offline-tested (`tests/test_reranker.py`).
- Hybrid retrieval in `backend/vector_store.py`: dense + BM25 sparse
  (`FastEmbedSparse`, `Qdrant/bm25`), then a two-stage `retrieve()` that pulls a
  candidate pool and reranks to the top N. Config-toggled for a clean A/B.
- Config flags in `backend/config.py` (`RETRIEVAL_MODE`, `RERANK_ENABLED`,
  `RETRIEVAL_CANDIDATE_K`, `RERANK_TOP_N`, ...).
- A/B harness in `evaluate.py` with measured judge+app cost and a `--compare` report.
- Content-safe LangSmith span for the rerank step (`backend/tracing.py`).

**Measured A/B result (judge & app `gpt-5-mini`, threshold 0.7). Full write-up in
`eval_comparison.md`:**

| Metric | Baseline (dense) | Improved (hybrid+rerank) | Δ |
|---|---:|---:|---:|
| Contextual Precision | 0.849 | 0.922 | +0.073 |
| Contextual Recall | 0.933 | 1.000 | +0.067 |
| Contextual Relevancy | 0.505 | 0.512 | +0.006 (flat) |
| Answer Relevancy | 0.950 | 1.000 | +0.050 |
| Faithfulness | 1.000 | 0.933 | −0.067 |
| Avg chunks/query | 13.2 | 8.8 | fewer, less noise |
| Avg latency (s) | 22.9 | 35.4 | slower (CPU rerank) |

**Honest reading:** precision, recall, answer-relevancy, and chunk efficiency
improved; Contextual Relevancy was a wash (two gains vs two regressions), and
Faithfulness dipped because the `security` case hit the canned "not found" fallback
even though retrieval was correct (Precision/Recall 1.0 there). Root cause is the
downstream answer-gating/query-rewrite logic, not retrieval. Next step: generate from
`retrieved_docs` when non-empty instead of trusting the binary relevancy gate, then
re-run the same A/B. This is a genuine but PARTIAL win — not a clean sweep.

**Session 2 measured OpenAI spend:** baseline $0.425 + improved $0.326 + 1-case probe
$0.081 ≈ **$0.83 total** (target was $0.75, hard ceiling $1.00 — within ceiling).

## Environment variables

The local `.env` file must contain the existing application variables:

```text
OPENAI_API_KEY=
TAVILY_API_KEY=
QDRANT_URL=
QDRANT_API_KEY=
```

The LangSmith variables are also prepared:

```text
LANGSMITH_API_KEY=
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=papeer
LANGSMITH_ENDPOINT=
LANGSMITH_WORKSPACE_ID=
```

The local configuration check found all these variables present and tracing set
to true. Secret values must never be shown in terminal output, responses, logs,
traces, or this file. A live trace still requires the dependencies and one
controlled API call.

## Phase plan and progress

**Overall status (2026-08-15): Phases 0–3 substantially done and verified; Phase 5
decided (no monitoring stack); Phases 4, 6, 7 only partially prepared and NOT
complete.** This session prioritised retrieval + evaluation (Phase 3) by the user's
choice. The marks below are corrected to reflect what is actually verified, not
merely written. Not everything from 0–7 is finished — deployment (6–7) and full
LangSmith verification (4) remain open.

Legend: `[x]` done & verified · `[~]` partially done · `[ ]` not started ·
`[!]` blocked/needs a decision.

### Phase 0 - Control and measurement design

- [x] Create this `AGENT.md` progress record.
- [x] Define model and API cost rules.
- [x] Define Tavily credit rules.
- [x] Define Azure-first deployment policy.
- [x] Define the rule that older metrics cannot be mixed with current metrics.
- [x] Verify the names and presence of local environment variables without
  exposing their values.
- [x] Create a test and measurement ledger.

### Phase 1 - Local setup and smoke test

- [x] Install or synchronize project dependencies. Confirmed installed on
  2026-08-15 (`.venv`); `fastembed` added.
- [x] Parse all Python files and run safe static checks.
- [x] Start the Streamlit application locally. Booted 2026-08-15 (health 200,
  UI rendered in browser: title, chat input, sidebar).
- [x] Test one small document first (hybrid probe with tiny docs, live Qdrant).
- [x] Test PDF ingestion and chunk creation (92 chunks from the report, in eval).
- [x] Test embedding creation and Qdrant insertion (verified in probes + eval).
- [x] Test a paper question (all 5 eval cases via the retrieve route).
- [x] Test a direct-answer question (prior-session smoke run in the ledger).
- [x] Test one controlled Tavily question (live web query routed to web_search,
  returned current 2026 results; `phase1_web` in the ledger).
- [x] Test one claim-verification question (live `verify_claim` route returned a
  correct superseded verdict; `phase1_claim` in the ledger).
- [~] Test session creation and session switching. Default session auto-created
  and rendered on app load; multi-session switching not click-tested in the UI.
- [x] Test history reconstruction and duplicate-message behavior (regression
  tests `tests/test_history.py` pass).
- [x] Record failures, latency, tokens, cost, and external API usage
  (`observability/runs.jsonl`, 58 rows; per-run cost in eval reports).

### Phase 2 - Correctness and reliability fixes

- [x] Fix only confirmed defects.
- [x] Normalize assistant history so tool-call and intermediate messages do not
  appear as final assistant replies.
- [x] Prevent internal query-rewrite messages from appearing as user messages.
- [x] Align query-rewrite limits with a central configuration value.
- [x] Add centralized configuration without exposing secrets.
- [x] Add safe cleanup for evaluation resources.
- [x] Add tests for the fixed behavior. Test execution waits for dependencies.

### Phase 3 - Rigorous RAG evaluation

- [x] Add a controlled evaluation command with case limits, fixed model choice,
  exact user queries, per-run measurements, and cleanup.
- [x] Build a hand-curated question set (factual, numeric, multi-section,
  security, unanswerable) with human-audited expected answers
  (`goldens_curated.json`).
- [x] Run one DeepEval cost probe before the full run.
- [x] Run the evaluation within the agreed cost cap (5 cases each side).
- [x] Keep the model, prompt, threshold, document, and retrieval settings fixed
  for a comparable A/B.
- [x] Report per-test scores, averages, pass rates, and failure reasons
  (`eval_comparison.md`).
- [x] Report latency, token usage, cost, and retrieval count.
- [x] Human-audit the results (per-case reading; security regression diagnosed).
- [x] Keep current results separate from the older synthetic `eval_results.json`.
- [ ] Broaden to multiple papers and a larger question set (deferred, budget).
- [ ] Add web-search and claim-verification eval cases (deferred).

### Phase 4 - LangSmith observability

- [x] Confirm that `LANGSMITH_API_KEY` is present without printing it.
- [x] Confirm that the user has approved sending prompts, retrieved context, and
  outputs to LangSmith (user: "I am ok with that, no problem").
- [x] Enable tracing (`LANGSMITH_TRACING=true`); all runs this session executed
  with it on.
- [x] Assign a clear LangSmith project name (`papeer`).
- [x] LangGraph and LangChain calls are configured for automatic tracing.
- [x] Add custom, content-safe traces for Qdrant add/search/delete, Tavily, and
  the rerank step (`backend/tracing.py`).
- [ ] Verify that DeepEval judge calls are traced. NOT verified this session.
- [x] Verify runs land with token usage and custom spans. Confirmed via the
  LangSmith SDK on 2026-08-15: project `papeer` returned live runs including
  `LangGraph`, `router`, `verify_claim`, `generate_answer`, `ChatOpenAI` (with
  token counts), and the custom `tavily_search` span.
- [ ] Use masking when raw inputs or outputs are not needed. Not implemented (user
  accepted full-content traces).

### Phase 5 - Prometheus and Grafana decision

- [x] Review the current architecture and decide whether system metrics are
  needed in addition to LLM traces. For this Streamlit capstone, the local
  ledger plus LangSmith is sufficient for the first deployment.
- [x] Decide not to add Prometheus and Grafana at this stage. Streamlit does
  not expose a native metrics endpoint, and adding a second service would add
  resource and deployment cost before a measured need exists.
- [ ] Add Prometheus only for useful application and infrastructure metrics.
- [ ] Add Grafana dashboards only if they improve debugging or deployment
  evidence.
- [ ] Avoid adding a monitoring stack that increases cost or complexity without a
  measured benefit.

### Phase 6 - Containerization

- [x] Add a container health check and keep runtime secrets outside the image.
- [ ] Build the Docker image locally.
- [ ] Confirm that `.env`, keys, caches, databases, and local secrets are not in
  the image.
- [ ] Run the container with environment variables at runtime.
- [ ] Verify the Streamlit port and health behavior.
- [ ] Decide which data needs persistent volumes.
- [ ] Verify session, SQLite, cache, and Qdrant behavior after a container
  restart.
- [ ] Tag the image with a reproducible version.
- [ ] Push to Docker Hub only after local verification.

### Phase 7 - Azure-first deployment

- [ ] Confirm the Azure for Students subscription and credit limits.
- [ ] Select a region and resource type with a clear cost estimate.
- [ ] Add shutdown and budget controls.
- [ ] Deploy only after the cost is accepted.
- [ ] Use runtime environment variables or a secret store.
- [ ] Configure persistent data where required.
- [ ] Verify the deployed application, logs, metrics, and shutdown behavior.
- [ ] Use AWS only if a separate check confirms that the deployment is free.

## Evaluation design

The evaluation must distinguish these questions:

1. Did retrieval find useful evidence?
2. Did retrieval include unnecessary evidence?
3. Did the answer use the evidence correctly?
4. Did the answer address the question?
5. Did the system fail safely when evidence was absent?
6. What did the request cost?
7. How long did each stage take?

DeepEval provides RAG quality scores. LangSmith provides runtime traces. Python
logging and deployment metrics provide application and infrastructure evidence.

## Retrieval experiment policy

Hybrid search and reranking are not assumed to be necessary.

After the baseline, compare the same test set with:

1. Dense retrieval.
2. Dense retrieval plus reranking.
3. Hybrid retrieval.
4. Hybrid retrieval plus reranking.

Keep a method only if it improves the selected metrics without unacceptable cost,
latency, or complexity.

## Known risks

- DeepEval scores depend on the judge model and prompt.
- A small or weak test set can produce misleading scores.
- Model aliases can change. Fixed snapshots may be needed for reproducibility.
- Changing embeddings requires re-indexing Qdrant.
- LangSmith traces may contain research-paper text.
- The application creates one Qdrant collection per session.
- Evaluation sessions can leave collections unless cleanup is added.
- Docker restarts can lose local SQLite and cache data without volumes.
- Azure and AWS free offers can change.
- Prometheus and Grafana can add resource and maintenance cost.
- Tavily usage can rise quickly if web and claim tests are repeated.
- The history fix addresses checkpoint-to-UI reconstruction. A live one-query
  smoke test is still required before claiming that no duplicate final answer
  is produced by the graph itself.

## Current implementation status

The following implementation files are now present:

- `backend/config.py`: model, search, path, and local cost-estimate settings.
- `backend/history.py`: visible-history normalization.
- `backend/telemetry.py`: secret-free JSONL run ledger.
- `backend/logging_config.py`: safe application logging helper.
- `tests/test_history.py`: regression tests for the duplicate-message defect.
- `.env.example`: safe configuration template.

Static compilation passed for all Python files. Runtime tests, API smoke tests,
DeepEval, LangSmith trace verification, Docker build, and Azure deployment are
not marked complete until they actually run successfully.

### Change log

- 2026-08-14: Created this record and verified that all required `.env` names
  are present without displaying values.
- 2026-08-14: Selected `gpt-5-mini` and `text-embedding-3-small` as defaults.
- 2026-08-14: Added internal-message filtering and final-answer history
  reconstruction.
- 2026-08-14: Added evaluation limits, exact-query evaluation, collection
  cleanup, local latency/token/cost ledger, and Docker health check.
- 2026-08-14: Added safe module logging. Logs contain counts and hashed session
  identifiers, not prompts, paper text, keys, or retrieved content.
- 2026-08-14: Added an Azure-first preparation record. No cloud resource was
  created because the subscription is not active.
- 2026-08-14: Static compilation passed. Dependency installation remains
  blocked by the package registry connection.
- 2026-08-15: (Takeover) Dependencies confirmed installed. Added `fastembed`
  hybrid BM25 + local cross-encoder reranker, config-driven dense/hybrid
  `retrieve()`, offline reranker tests, and an A/B evaluation harness with
  measured judge+app cost and a `--compare` report. Removed dead `TavilyClient`
  and repo clutter (`index/Untitled.java`, stale batch/probe files, pytest
  caches); synced `requirements.txt` and `.gitignore`; added a rerank trace span.
- 2026-08-15: Ran a controlled 5-case A/B (baseline dense vs hybrid+rerank) on
  the hand-curated set. Precision/Recall/Answer-Relevancy improved and chunk count
  dropped; Contextual Relevancy stayed flat and Faithfulness dipped due to a
  downstream answer-gating fallback (diagnosed, not yet fixed). Measured spend
  this session ≈ $0.83. Results in `eval_baseline.json`, `eval_improved.json`,
  `eval_comparison.md`.
