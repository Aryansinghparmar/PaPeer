# Papeer — Senior-Level Interview Q&A

Twenty questions specific to this repository, each with what the interviewer is probing, a
first-person model answer grounded in the code, likely follow-ups, and red flags to avoid.

---

## 1. Question
What problem does Papeer actually solve, and who is it for? Why build it instead of using a general chatbot?

### What the Interviewer Is Testing
Product judgment — whether I understand the user, the scope, and the reason the project exists.

### Strong Answer
Papeer is for students and researchers who need to *use* a paper quickly — pull a method,
a number, or check whether a claim still holds — without reading the whole PDF. A general
chatbot fails here for two reasons: it isn't grounded in the specific document, so it
hallucinates, and it has no explicit way to check whether a finding has been superseded.
Papeer grounds answers with retrieval over the uploaded paper (Qdrant + embeddings) and
adds an explicit `verify_claim` path that searches the web and ArXiv. It's deliberately
scoped as a single-user local tool — there's no account system — because the goal was a
strong, honest RAG project, not a multi-tenant product. The trade-off is that it assumes a
technical user with their own API keys. If the requirement changed to "many users," the
architecture would need auth and per-user budgets first.

### Likely Follow-Up
Isn't long-context (just pasting the paper) simpler than RAG now?

### Strong Follow-Up Direction
Acknowledge it's viable for one small paper, but note retrieval wins for multiple/large
papers, cost control, and citation grounding; I'd decide based on measured cost and accuracy.

### Red Flags to Avoid
Claiming it's a finished product, inventing a user base, or pretending it has multi-user features.

---

## 2. Question
Why LangGraph for the workflow instead of a simple LangChain chain or a few function calls?

### What the Interviewer Is Testing
Whether I chose orchestration deliberately and understand its cost.

### Strong Answer
The workflow isn't linear — it has branches (retrieve vs verify vs direct) and *loops*
(retrieve → relevancy check → query rewrite → retrieve again, bounded). LangGraph gives me
an explicit state machine with a typed state (`RAGState`) and checkpointing, so I can persist
each session's conversation and resume it. A plain chain would make the retry/rewrite loop and
the "force pending tool calls before finishing" logic awkward and error-prone. The cost is
conceptual overhead — nodes, conditional edges, and a state schema are more to reason about
than a function. For this project the loops and per-session checkpointing justified it; for a
strictly linear pipeline I'd use a simple chain.

### Likely Follow-Up
What specifically breaks if you get the graph edges wrong?

### Strong Follow-Up Direction
Point to the `agent_routing` guard: if I let the agent finish with pending tool calls, the
checkpoint would hold an `AIMessage` with unmatched `tool_call` ids and corrupt history for
every future turn — so edge correctness is a real reliability concern.

### Red Flags to Avoid
Saying LangGraph is "more scalable" (it isn't the point) or being unable to name a downside.

---

## 3. Question
Why RAG with managed embeddings rather than fine-tuning a model on the papers?

### What the Interviewer Is Testing
Understanding of the core AI architecture trade-off.

### Strong Answer
Fine-tuning bakes knowledge into weights; it's expensive, slow to update, and still
hallucinates without grounding. RAG keeps the paper as external data — I chunk it, embed it
with `text-embedding-3-small`, store it in Qdrant, and retrieve at query time. That gives
grounded, citable answers and lets documents change with no retraining. The trade-off is that
retrieval quality becomes the bottleneck — which my evaluation confirmed (low Contextual
Relevancy on the baseline). So I invested in retrieval (hybrid + rerank) rather than the model.
I'd only consider fine-tuning for a fixed domain style or format, not for factual grounding.

### Likely Follow-Up
Your embedding model is fixed at 1536 dims — what's the migration cost if you change it?

### Strong Follow-Up Direction
Changing the embedding model changes vector dimensionality and semantics, so every Qdrant
collection must be re-indexed; I'd version collections and re-embed, ideally behind a flag.

### Red Flags to Avoid
Claiming RAG "eliminates hallucination" (it reduces it) or ignoring the re-index cost.

---

## 4. Question
You create one Qdrant collection per session (`papeer_{session_id}`). Defend that choice and its limits.

### What the Interviewer Is Testing
Data-model judgment and awareness of scale implications.

### Strong Answer
Per-session collections give hard isolation — one chat's papers can never leak into another,
and cleanup is a single `delete_collection`. It's simple to reason about and matched the
single-user design. The cost is collection sprawl: nothing automatically removes user
sessions, so collections accumulate, and at multi-user scale this is wasteful. The standard
alternative is a single collection with a `session_id` payload filter, which scales far better
but makes isolation a query-discipline concern rather than a structural one. At the current
scale the isolation-by-construction was the right call; if this went multi-user I'd switch to
the filtered single-collection model and add lifecycle cleanup.

### Likely Follow-Up
How would you migrate existing per-session collections to the filtered model?

### Strong Follow-Up Direction
Dual-write during a transition, backfill payloads with `session_id`, switch reads to filtered
queries, then drop old collections — with the evaluation harness guarding quality.

### Red Flags to Avoid
Pretending sprawl isn't an issue, or claiming per-collection is a security boundary (it isn't).

---

## 5. Question
Papeer has no REST API — its "interfaces" are LangGraph tools and structured outputs. Walk me through that design.

### What the Interviewer Is Testing
Interface design and typed contracts in an LLM system.

### Strong Answer
The contracts that matter are the tool schemas and the structured LLM outputs. Tools use
Pydantic `args_schema` (`RetrieverInput` with `query`/`k`, `WebSearchInput` with
`optimized_query`/`max_results`), so the model must produce valid arguments. Router,
relevancy, and claim-verification outputs are Pydantic models (`RouterDecision`,
`RelevancyDecision`, `ClaimVerificationResult`) via `with_structured_output`, which turns a
fuzzy LLM response into a validated object I can branch on deterministically. That's the API:
typed boundaries between probabilistic components. The eval CLI (`evaluate.py`) is the other
interface. There's no HTTP API because it's a single Streamlit app; if I exposed it as a
service I'd add a FastAPI layer with request validation and versioning.

### Likely Follow-Up
What happens if the model returns something that doesn't fit the schema?

### Strong Follow-Up Direction
Structured output validation raises/coerces; I'd add explicit fallbacks (retry, default route)
so a schema miss degrades gracefully instead of erroring the turn.

### Red Flags to Avoid
Describing REST endpoints that don't exist, or ignoring validation-failure handling.

---

## 6. Question
There's no authentication. Is that a problem, and what would you do before deploying?

### What the Interviewer Is Testing
Security judgment and honesty about a real gap.

### Strong Answer
For a local single-user prototype it's acceptable — the app assumes one trusted user. But it's
the single biggest blocker to deployment: a public Azure URL would be open to anyone, and
because the app uses *my* OpenAI and Tavily keys, an open endpoint is an open wallet.
Session isolation via per-collection naming is data organization, not access control — there's
no identity. Before deploying I'd put authentication in front (Azure Container Apps'
Easy Auth or an identity provider), add per-user rate limiting, and a hard budget cap so a
single user can't drain the account. I'd treat that as a release gate, not a nice-to-have.

### Likely Follow-Up
Auth aside, what's your worst-case cost-abuse scenario?

### Strong Follow-Up Direction
An open endpoint scripted to upload large docs and spam expensive queries; mitigations are
auth + rate limits + budget alerts + capping document size and `max_results`.

### Red Flags to Avoid
Calling per-session collections a security feature, or dismissing the deployment risk.

---

## 7. Question
Your evaluation showed faithfulness dropping on one case. Walk me through what happened and how you'd fix it.

### What the Interviewer Is Testing
Debugging discipline and honesty about a real defect.

### Strong Answer
In the hybrid+rerank A/B, the average faithfulness dropped and Contextual Relevancy stayed
flat, which looked like the change hurt. But I read the per-case data instead of trusting the
average. On the `security` question, Contextual Precision and Recall were both 1.0 — retrieval
was *correct* — yet the answer said "I wasn't able to find relevant information," so the judge
flagged a contradiction and faithfulness fell to 0.667. The root cause is in
`generate_answer_node`: when the relevancy gate returns false after a query rewrite, it emits a
canned "not found" answer regardless of whether good chunks were actually retrieved. So the
reranker exposed a pre-existing downstream bug, not a retrieval regression. The fix is to
generate from `retrieved_docs` whenever they're non-empty, and only fall back to "not found"
when retrieval is genuinely empty — then re-run the same A/B to confirm.

### Likely Follow-Up
Could the DeepEval judge itself be wrong here?

### Strong Follow-Up Direction
Possible — LLM judges have variance — so I'd human-audit that case (I did), and the judge was
right: the answer text genuinely contradicted the retrieved context.

### Red Flags to Avoid
Blaming the metric to avoid the bug, or claiming a clean win when it was partial.

---

## 8. Question
Where does latency come from, and how would you reduce it without hurting quality?

### What the Interviewer Is Testing
Performance reasoning grounded in measurement.

### Strong Answer
Measured end-to-end latency went from ~23 s (dense baseline) to ~35 s (hybrid+rerank) per eval
query. The contributors are the agentic loop (multiple sequential LLM calls), embedding on
ingest, and now the CPU cross-encoder reranking a 20-candidate pool. Before optimizing I'd use
LangSmith per-node timings to confirm the split. Likely wins: shrink the candidate pool
(`RETRIEVAL_CANDIDATE_K`) or use a lighter/GPU/hosted reranker; reduce unnecessary rewrite
loops; and parallelize independent LLM calls where safe. I'd validate each change against the
evaluation so I don't trade latency for accuracy blindly.

### Likely Follow-Up
Which single change would you try first?

### Strong Follow-Up Direction
Reduce the candidate pool from 20 and measure the reranker time vs. metric delta — cheapest
lever with a clear measurement.

### Red Flags to Avoid
Quoting invented latency numbers or optimizing without measuring.

---

## 9. Question
What breaks first at 10× or 100× users, given it's a Streamlit app?

### What the Interviewer Is Testing
Scalability limits of the current architecture.

### Strong Answer
Streamlit is a single stateful process, so concurrency is the first wall — sessions live in
memory plus SQLite and `sessions.json` on one node, which doesn't scale horizontally.
Per-session Qdrant collections would proliferate. External rate limits (OpenAI/Tavily) would
bite next. To scale I'd separate the UI from a stateless backend service (FastAPI), move
conversation state to a shared store, switch Qdrant to a filtered single collection, add a
queue for heavy ingestion, and put auth + rate limiting in front. That's a real re-architecture,
which is fine — the current design was optimized for a solo demo, not throughput.

### Likely Follow-Up
Why is Streamlit specifically hard to scale here?

### Strong Follow-Up Direction
It couples UI, session state, and compute in one stateful process with websocket sessions;
horizontal replicas would each hold separate in-memory state without a shared backend.

### Red Flags to Avoid
Claiming it "scales fine" or proposing microservices without justification.

---

## 10. Question
The agent can loop (retrieve → rewrite → retrieve). How do you keep that safe and consistent?

### What the Interviewer Is Testing
Concurrency/consistency and agent control.

### Strong Answer
Two guards. First, hard caps: `MAX_RETRIEVAL_ATTEMPTS` and `MAX_QUERY_REWRITES` bound the loop
so ambiguous queries can't burn tokens indefinitely — at the cap the agent switches to a plain
LLM so it can't emit more tool calls. Second, checkpoint integrity: `agent_routing` forces
pending tool calls to execute before the graph can finish, because an `AIMessage` with
tool-calls that isn't matched by `ToolMessage`s would corrupt the persisted history for every
future turn in that session. So the loop is bounded *and* the state stays consistent. The
trade-off is that a hard cap can stop before finding the best evidence; a confidence-based stop
would be smarter but harder to tune.

### Likely Follow-Up
How would you move from a fixed cap to a smarter stop?

### Strong Follow-Up Direction
Use the relevancy score/confidence to decide continuation, with the cap as a backstop, and
validate against the eval so quality doesn't regress.

### Red Flags to Avoid
Ignoring the checkpoint-corruption risk or treating the caps as arbitrary.

---

## 11. Question
What are the main security and privacy risks, including AI-specific ones?

### What the Interviewer Is Testing
Breadth of security thinking for an LLM app.

### Strong Answer
Beyond the missing auth: prompt injection is real — uploaded paper text and web results flow
straight into prompts, so a malicious document or page could try to steer the model. The
loaders fetch arbitrary URLs server-side, which is an SSRF-shaped surface. Secrets are handled
reasonably — `.env` is gitignored, keys are read from the environment and never baked into the
image, and my logs/traces are content-safe (hashed session ids, counts not content). The
biggest privacy note is that LangSmith receives prompts and paper text; that's a deliberate,
user-accepted trade-off, and I support masking if needed. For production I'd add auth, budget
caps, URL allow-listing, and consider injection guardrails. These are recommendations — I'm not
claiming an observed exploit.

### Likely Follow-Up
How would you actually mitigate prompt injection here?

### Strong Follow-Up Direction
Separate instructions from retrieved content, constrain the answer to cite retrieved chunks,
add output checks, and never let document content trigger tool actions without guards.

### Red Flags to Avoid
Claiming it's "secure," or forgetting the injection/SSRF surface unique to RAG.

---

## 12. Question
What's actually tested, what isn't, and is that the right balance?

### What the Interviewer Is Testing
Testing strategy and honesty about coverage.

### Strong Answer
I have two focused unit test files that pass (verified this session): `test_history.py` covers
chat-history reconstruction and the duplicate-message bug, and `test_reranker.py` covers the
cross-encoder ordering, score metadata, and edge cases — offline, no API cost. What's *not*
unit-tested is the graph end-to-end, routing correctness, ingestion, and the UI. I lean instead
on the evaluation harness as the quality signal, which is reasonable for a prototype but leaves
gaps. If I were hardening it, I'd add unit tests around the routing/gating logic (which is where
the real bug lives), a few integration tests with mocked OpenAI/Qdrant/Tavily, and a thin E2E
smoke test — a normal pyramid.

### Likely Follow-Up
Given limited time, what one test would you add first?

### Strong Follow-Up Direction
A unit test for `generate_answer_node`'s gating: non-empty `retrieved_docs` must produce a
grounded answer, not the canned fallback — it would have caught the faithfulness bug.

### Red Flags to Avoid
Saying "it's well tested," or claiming tests pass without having run them.

---

## 13. Question
Walk me through deploying this to Azure and keeping cost under control.

### What the Interviewer Is Testing
Deployment and cost awareness (real constraint).

### Strong Answer
It's containerized — the `Dockerfile` installs pinned deps, pre-caches the BM25 and reranker
models so there's no cold download at request time, exposes 8501, and has a health check. I
wrote a full guide (`azure/DEPLOYMENT.md`) for Azure Container Apps: create the environment,
deploy the image from Docker Hub, and pass the API keys as Container App *secrets* referenced by
env vars — never baked into the image. On cost: Container Apps has a monthly free grant and bills
per vCPU-second/GiB-second. Always-on at 1 vCPU / 2 GiB is roughly $30–75/month, which would burn
a $100 student credit fast, so I default to **scale-to-zero** (near-free, ~30–60 s cold start)
and only flip to always-on during an active demo window. I also document a budget alert and a
one-command teardown. I have not created any billable resource yet.

### Likely Follow-Up
What's the downside of scale-to-zero for this specific app?

### Strong Follow-Up Direction
Cold starts and session resets — Streamlit is stateful, so a wake loses in-memory state; I'd
mount Azure Files or keep min-replicas 1 during demos if persistence matters.

### Red Flags to Avoid
Claiming it's deployed, inventing exact Azure prices, or baking secrets into the image.

---

## 14. Question
How do you observe this system in production, and what did you deliberately not add?

### What the Interviewer Is Testing
Observability judgment and restraint.

### Strong Answer
Two layers. A local, secret-free JSONL ledger (`telemetry.py`) records route, latency, token
counts, and estimated cost per run using OpenAI usage callbacks and a central pricing table —
that's my cost truth. And LangSmith gives distributed traces of the graph, LLM calls, and my
custom content-safe spans for Qdrant, Tavily, and rerank; I verified via the SDK that runs land
in the `papeer` project. I deliberately did *not* add Prometheus/Grafana: Streamlit exposes no
native metrics endpoint, and a second service would add cost and ops overhead before there's a
measured need. That restraint is itself a decision I can defend — add monitoring when a metric
justifies it, not by default.

### Likely Follow-Up
When would Prometheus/Grafana become worth it?

### Strong Follow-Up Direction
Once it's a long-running multi-user service where infra metrics (CPU, memory, request rate,
error rate) drive scaling and alerting decisions.

### Red Flags to Avoid
Adding tools for their own sake, or claiming dashboards exist that don't.

---

## 15. Question
Qdrant, OpenAI, or Tavily goes down or rate-limits you mid-request. What happens today, and what should happen?

### What the Interviewer Is Testing
External-dependency failure handling.

### Strong Answer
Today, most external failures surface as exceptions; ingestion is wrapped in try/except in the
UI so an upload fails cleanly, and empty retrieval returns a "no documents found" path, but there
is no retry/backoff, timeout tuning, or circuit breaker around the LLM/search/vector calls. So a
transient OpenAI blip can fail a turn. What *should* happen: bounded retries with exponential
backoff on transient errors, explicit timeouts, and graceful user-visible degradation (e.g., "web
search is unavailable, answering from the paper only"). For claim verification, which does two
Tavily calls, I'd degrade to one source rather than fail entirely. This is a clear, known
reliability gap I'd close before any real usage.

### Likely Follow-Up
Where would a retry be dangerous?

### Strong Follow-Up Direction
Non-idempotent or expensive steps — I'd only retry idempotent reads (search/embed), cap attempts,
and avoid retrying anything that could double-charge or duplicate state.

### Red Flags to Avoid
Claiming robust resilience that isn't in the code.

---

## 16. Question
What's the single most important trade-off in the project, and why did you accept it?

### What the Interviewer Is Testing
Ability to identify and defend the central trade-off.

### Strong Answer
The retrieval upgrade: hybrid search plus a local cross-encoder reranker buys quality and costs
latency (and image size). I accepted it because the baseline evaluation showed retrieval was the
weak link, and doing the rerank *locally* on CPU means the quality gain comes with zero marginal
OpenAI cost — which matters on a tight budget. The measured result was a genuine but partial win:
precision, recall, and answer relevancy improved and chunk noise dropped, at the cost of ~23 s →
~35 s latency. I accepted that because for a research assistant, answer quality matters more than
a few seconds, and the latency is tunable. If this were a latency-critical product I'd reconsider
the pool size or the reranker.

### Likely Follow-Up
Would an LLM-based reranker have been better?

### Strong Follow-Up Direction
Possibly higher quality but it adds per-query token cost and latency; I chose the local
cross-encoder to keep marginal cost at zero, and I'd A/B it if budget allowed.

### Red Flags to Avoid
Presenting the trade-off as free, or overstating the quality gain.

---

## 17. Question
What technical debt would you pay down first, and why that order?

### What the Interviewer Is Testing
Prioritization.

### Strong Answer
First, the answer-gating bug in `generate_answer_node` — it produces wrong answers and it's
already diagnosed, so it's the highest impact for the least effort; I'd fix it and re-run the
A/B to confirm. Second, initialize git + GitHub with a minimal CI running `pytest`, because right
now there's no version control, which hurts credibility and safety. Third, add retry/backoff and
timeouts around external calls for reliability. I'd defer bigger items — auth, multi-paper eval,
persistence — behind those, because correctness and repeatability come before new capability.

### Likely Follow-Up
Why not do auth first if deployment is the goal?

### Strong Follow-Up Direction
Auth is a deployment *gate*, but the correctness bug affects every user and is cheaper to fix; I'd
sequence auth right before the actual public deploy.

### Red Flags to Avoid
Listing debt without prioritizing, or ignoring the known bug.

---

## 18. Question
If you restarted this project with more time, what would you redesign?

### What the Interviewer Is Testing
Architectural hindsight.

### Strong Answer
I'd split the UI from a stateless backend service from day one, so state lives in a shared store
and the app can scale and be tested independently. I'd use a single filtered Qdrant collection
with session lifecycle management instead of per-session collections. I'd bake evaluation into CI
as a quality gate across multiple documents, not a manual run on one. And I'd design the answer
node to always prefer generating from retrieved evidence, avoiding the gating bug by construction.
I wouldn't change the core RAG-with-managed-embeddings approach or the cost-first discipline —
those were right.

### Likely Follow-Up
What would you keep exactly as is?

### Strong Follow-Up Direction
The bounded agentic graph, the content-safe cost/trace observability, and the local reranker —
they solved real problems cheaply.

### Red Flags to Avoid
Redesigning everything (signals poor judgment) or nothing (signals no reflection).

---

## 19. Question
Where does this go over the next six months to become production-grade?

### What the Interviewer Is Testing
Roadmap realism.

### Strong Answer
Near term: fix the gating bug and re-measure; add git + CI; add retry/backoff. Then the
deployment track: auth + per-user budget caps, deploy to Azure Container Apps with scale-to-zero
and budget alerts, and decide persistence (Azure Files or a managed store). In parallel, broaden
evaluation to multiple papers plus web/claim cases and wire it into CI as a quality gate. Medium
term: move to a stateless backend + filtered Qdrant collection for scale, tune reranker latency,
and add prompt-injection guardrails. Each step is validated by the evaluation and the cost ledger
rather than by feel.

### Likely Follow-Up
How do you avoid this roadmap becoming a wish list?

### Strong Follow-Up Direction
Gate each item on a measurable outcome (a metric moves, a cost drops, coverage rises) and do the
cheapest high-impact item first.

### Red Flags to Avoid
Calendar promises or a laundry list with no sequencing.

---

## 20. Question
How do you know Papeer actually works well? Defend your evaluation methodology.

### What the Interviewer Is Testing
AI evaluation rigor and honesty.

### Strong Answer
I use DeepEval with five RAG metrics — Contextual Precision, Recall, Relevancy, Answer Relevancy,
and Faithfulness — on a small hand-authored, human-audited question set that spans factual,
numeric, multi-section, security, and *unanswerable* cases. It's run as a controlled A/B: the same
questions through baseline dense retrieval vs. hybrid+rerank, same judge model and threshold, with
fresh Qdrant collections and cleanup each run. Crucially I measure cost — application tokens via
usage callbacks and DeepEval's per-metric `evaluation_cost` — so the whole A/B ran for under a
dollar and I ran a 1-case probe first to size it. I'm honest about the limits: it's one document
with a handful of questions, so it's a *signal*, not a benchmark, and the LLM judge has variance,
which is why I human-audited the surprising cases (that's how I caught the answer-gating bug). To
strengthen it I'd broaden to multiple papers and put it in CI.

### Likely Follow-Up
LLM-as-judge is itself unreliable — why trust these numbers?

### Strong Follow-Up Direction
I don't trust them blindly: I fix the judge model/threshold for comparability, keep it to relative
A/B deltas rather than absolute claims, and human-audit outliers.

### Red Flags to Avoid
Presenting the numbers as a definitive benchmark, hiding the single-document limitation, or
claiming the judge is objective truth.
