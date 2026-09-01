# Papeer — Senior-Level Interview Q&A

Twenty-two questions specific to this repository, each with what the interviewer is
probing, a first-person model answer grounded in the code, likely follow-ups, and red
flags to avoid. Questions 21–22 were added after the FastAPI + React migration; the
others were revised in place where the architecture changed (notably Q5, Q6, Q7, Q9, Q13).

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
scoped as a no-real-users portfolio project — there's no account system yet — because the
goal was a strong, honest RAG project with production-shaped architecture, not an actual
multi-tenant product. The trade-off is that it assumes a technical user with their own API
keys today. The API/SPA split now makes adding real multi-user auth a config step rather
than a redesign, if that requirement ever appeared.

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
strictly linear pipeline I'd use a simple chain. It also paid off when I added the FastAPI
layer: the same graph definition compiles with either a sync or an async checkpointer, so the
workflow logic didn't have to change at all for the new client.

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
and cleanup is a single `delete_collection` (now exposed directly as `DELETE
/api/sessions/{sid}`). It's simple to reason about and matched the original single-user
design. The cost is collection sprawl: nothing automatically removes user sessions beyond
that explicit delete, so collections accumulate, and at multi-user scale this is wasteful.
The standard alternative is a single collection with a `session_id` payload filter, which
scales far better but makes isolation a query-discipline concern rather than a structural
one. I deliberately kept per-session collections through the FastAPI/React migration — it
wasn't the bottleneck the evaluation or the architecture review flagged, so I didn't spend
effort there. If this went multi-user I'd switch to the filtered single-collection model
and add lifecycle cleanup.

### Likely Follow-Up
How would you migrate existing per-session collections to the filtered model?

### Strong Follow-Up Direction
Dual-write during a transition, backfill payloads with `session_id`, switch reads to filtered
queries, then drop old collections — with the evaluation harness guarding quality.

### Red Flags to Avoid
Pretending sprawl isn't an issue, or claiming per-collection is a security boundary (it isn't).

---

## 5. Question
Walk me through the API design. What does a client actually talk to?

### What the Interviewer Is Testing
Interface design and typed contracts, now that there's a real HTTP API to defend.

### Strong Answer
There's now a real REST + SSE API (`api/`), built with FastAPI and consumed by a React SPA.
Session lifecycle is plain REST: `POST/GET/DELETE /api/sessions`, `PATCH` to trigger
auto-naming, and a `/documents` sub-resource for upload/URL/ArXiv ingestion. Chat is
different on purpose: `POST /api/sessions/{sid}/chat` returns a `StreamingResponse` of
Server-Sent Events, because an LLM answer arrives token-by-token and the user shouldn't
wait for the whole thing. I picked SSE over WebSockets because it's one-directional,
works over plain HTTP, and `StreamingResponse` plus `astream()` gets me there with almost
no extra code — WebSockets would be overkill for a single streamed response per request.
Underneath the HTTP layer, the older typed contracts are still exactly what makes the graph
itself safe: Pydantic `args_schema` on tools (`RetrieverInput`, `WebSearchInput`) and
`with_structured_output` for `RouterDecision`, `RelevancyDecision`,
`ClaimVerificationResult` — those turn fuzzy LLM output into objects I can branch on
deterministically, independent of which HTTP client is calling in. The API layer is a
thin, typed wrapper around that — it doesn't duplicate any of the AI logic.

### Likely Follow-Up
Why did the graph need an async variant instead of just calling the sync one from FastAPI?

### Strong Follow-Up Direction
FastAPI's `StreamingResponse` wants an async generator driving `astream()` so the event
loop isn't blocked mid-stream; the LangGraph SQLite checkpointer has an async variant
(`AsyncSqliteSaver`) for exactly this, so I added `build_graph_async` alongside the
existing sync `build_graph` rather than forcing everything through a thread pool.

### Red Flags to Avoid
Claiming WebSockets were "better" without justifying it, or describing the API without
mentioning that it's a thin layer over unchanged graph logic.

---

## 6. Question
How is access to Papeer controlled? Walk me through the auth story.

### What the Interviewer Is Testing
Security judgment — knowing the difference between a documented plan and something actually enforced.

### Strong Answer
Today, right now, nothing enforces access — anyone who can reach the API or the SPA can use
it and spend my OpenAI/Tavily budget. I'm honest about that rather than papering over it.
But the plan isn't hand-rolled auth: it's Azure's platform-managed authentication —
Static Web Apps' built-in auth (Entra ID, configured multi-tenant so any reviewer can sign
in with their own Microsoft account, not just accounts in my directory) gating the SPA,
with the linked FastAPI backend receiving the forwarded identity via an
`X-MS-CLIENT-PRINCIPAL-NAME` header. I already added the read side of that —
`api/deps.get_current_user` reads the header today — so the only remaining step is turning
the gateway on when I actually deploy. I chose a platform-managed provider over writing
JWT/session auth myself because it's genuinely free on the student plan, it's what real
teams do for internal or demo tools, and it removes an entire class of bugs (token
handling, session fixation) I'd otherwise own. Before the URL goes public, that gate has to
actually be verified live — that's my release gate, not a "nice to have."

### Likely Follow-Up
Why trust a platform auth feature instead of something you control end-to-end?

### Strong Follow-Up Direction
For this project's risk profile — no real user data, the only asset to protect is API
spend — a well-audited platform feature is lower-risk than code I'd write once and rarely
revisit; I'd reconsider for a system handling real user data, where I'd want more control
over session semantics.

### Red Flags to Avoid
Claiming the app is "secured" when the gate isn't live yet, or dismissing the deployment
risk because "there are no real users."

---

## 7. Question
Your evaluation showed faithfulness dropping on one case, then you re-ran it. Walk me through the whole arc.

### What the Interviewer Is Testing
Debugging discipline and honesty about a real defect *and* about a re-measurement's limits.

### Strong Answer
This has two chapters, and both matter. First: in the hybrid+rerank A/B, the average
faithfulness dropped and Contextual Relevancy stayed flat, which looked like the retrieval
change hurt. I read the per-case data instead of trusting the average. On the `security`
question, Contextual Precision and Recall were both 1.0 — retrieval was *correct* — yet the
answer said "I wasn't able to find relevant information," so the judge flagged a
contradiction and faithfulness fell to 0.667. The root cause was in `generate_answer_node`:
when the relevancy gate returned false after a query rewrite, it emitted a canned "not
found" answer regardless of whether good chunks were actually retrieved. I fixed it to
generate from `retrieved_docs` whenever they're non-empty. Second chapter: I re-ran the
same evaluation to confirm the fix, and it worked — Faithfulness and Answer Relevancy are
now 1.0 across every case. But that same re-run also showed something I didn't expect: the
fine-grained retrieval numbers themselves (precision, recall) swung between the two
hybrid+rerank runs by more than the "improvement" I'd originally reported over the
baseline. At five cases, that's not a stable signal — it's run-to-run variance from the
non-deterministic agent loop and LLM-judge noise. I reported that honestly in
`eval_comparison.md` instead of keeping the flattering first number.

### Likely Follow-Up
Doesn't that undercut your whole retrieval-improvement story?

### Strong Follow-Up Direction
No — it sharpens it: the *reliable* wins (fixed faithfulness, fewer retrieved chunks,
cheaper judging) still stand on repeat measurement; the *precise* precision/recall deltas
don't, and I'd say so explicitly rather than imply more confidence than the sample size
supports. The fix is multi-seed evaluation, which I've scoped but not yet run (budget-gated).

### Red Flags to Avoid
Blaming the metric to avoid the bug, claiming a clean win when it was partial, or hiding
the variance finding to make the story sound cleaner than it is.

---

## 8. Question
Where does latency come from, and how would you reduce it without hurting quality?

### What the Interviewer Is Testing
Performance reasoning grounded in measurement.

### Strong Answer
Measured end-to-end latency went from ~23 s (dense baseline) to ~35–38 s (hybrid+rerank) per
eval query, and I felt this directly once I had a real streaming UI — there's a genuine
multi-second gap before the first token appears. The contributors are the agentic loop
(multiple sequential LLM calls), embedding on ingest, and the CPU cross-encoder reranking a
20-candidate pool. Before optimizing I'd use LangSmith per-node timings to confirm the
split, including SSE time-to-first-token now that it's user-visible. Likely wins: shrink the
candidate pool (`RETRIEVAL_CANDIDATE_K`) or use a lighter/GPU/hosted reranker; reduce
unnecessary rewrite loops; and parallelize independent LLM calls where safe. I'd validate
each change against the evaluation so I don't trade latency for accuracy blindly.

### Likely Follow-Up
Which single change would you try first?

### Strong Follow-Up Direction
Reduce the candidate pool from 20 and measure the reranker time vs. metric delta — cheapest
lever with a clear measurement.

### Red Flags to Avoid
Quoting invented latency numbers or optimizing without measuring.

---

## 9. Question
What breaks first at 10× or 100× users, given the current architecture?

### What the Interviewer Is Testing
Scalability limits — and whether I understand what the FastAPI/React split did and didn't fix.

### Strong Answer
The API/SPA split fixed the *client* concurrency story — FastAPI is async and can serve
many requests without one browser tab blocking another the way a single Streamlit process
did. But it didn't fix the *state* story: conversation state is still SQLite plus
`sessions.json` on one node, and per-session Qdrant collections would still proliferate.
External rate limits (OpenAI/Tavily) would bite regardless of the client architecture. To
scale further I'd move conversation state to a shared store (Postgres via LangGraph's
`PostgresSaver`), switch Qdrant to a filtered single collection, add a queue for heavy
ingestion, and turn on the auth + rate limiting that's already planned. That's a real
next re-architecture, and it's fine that it isn't done — this design was optimized for a
correct, demoable full-stack app, not for throughput nobody currently needs.

### Likely Follow-Up
Why keep Streamlit around at all if it's the thing you moved away from?

### Strong Follow-Up Direction
It cost nothing to keep — the backend never depended on it, and it's a useful sanity-check
client: if a change breaks Streamlit but not the API, the bug is almost certainly in that
file, not in the shared graph logic.

### Red Flags to Avoid
Claiming the migration alone "solved scaling," or proposing a full microservices rewrite
without justification.

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
Beyond the auth gap covered in Q6: prompt injection is real — uploaded paper text and web
results flow straight into prompts, so a malicious document or page could try to steer the
model. The loaders fetch arbitrary URLs server-side, and that's now reachable via a
documented HTTP endpoint rather than only a UI form, which is a marginally larger surface I
should acknowledge rather than hand-wave. Secrets are handled reasonably — `.env` is
gitignored, keys are read from the environment and never baked into the image, CORS is now
explicit and configurable, and my logs/traces are content-safe (hashed session ids, counts
not content). The biggest privacy note is that LangSmith receives prompts and paper text;
that's a deliberate, user-accepted trade-off, and I support masking if needed. For
production I'd turn on the Easy Auth gate, add budget caps, URL allow-listing, and consider
injection guardrails. These are recommendations — I'm not claiming an observed exploit.

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
I have two focused unit test files that pass (5/5, verified this session): `test_history.py`
covers chat-history reconstruction and the duplicate-message bug, and `test_reranker.py`
covers cross-encoder ordering, score metadata, and edge cases — offline, no API cost. On the
frontend, `tsc -b` (strict TypeScript) and `npm run build` both pass cleanly. Beyond that,
I did real, live, manual verification of the full stack this session — health checks,
session CRUD, document upload, full RAG retrieval, `/btw`, and an empty-collection fallback,
exercised through an actual running API and browser, not just imagined. I'm explicit that
this was manual, not an automated E2E suite — there's no Playwright/Cypress test yet, which
is a real gap. I lean on the evaluation harness as the AI-quality signal. If I were
hardening this further, I'd add unit tests around the routing/gating logic (which is where
the real bug lived), a few integration tests with mocked OpenAI/Qdrant/Tavily, and an
automated Playwright smoke test covering upload → ask → stream — a normal pyramid, with the
last layer being the most valuable gap right now given how much manual testing already
found.

### Likely Follow-Up
Given limited time, what one test would you add first?

### Strong Follow-Up Direction
An automated Playwright smoke test of upload → ask → stream → verify — it would have caught
the SSE double-emission bug and the frontend state races automatically instead of requiring
me to notice them by reading live output.

### Red Flags to Avoid
Saying "it's well tested," or claiming tests pass without having run them.

---

## 13. Question
Walk me through deploying this to Azure and keeping cost under control.

### What the Interviewer Is Testing
Deployment and cost awareness (real constraint), including the updated topology.

### Strong Answer
The target topology is two Azure services: the FastAPI backend on **Azure Container Apps**
(the existing Dockerfile pattern, scale-to-zero) and the React build on **Azure Static Web
Apps**, which has a generous free tier and — importantly — built-in Entra ID auth that gates
the SPA and forwards identity to the linked backend. That split also cleanly matches the
Azure-native auth story from Q6. On cost: Container Apps has a monthly free grant and bills
per vCPU-second/GiB-second; always-on at 1 vCPU / 2 GiB is roughly $30–75/month, which would
burn a small student credit fast, so I default to scale-to-zero and only flip to always-on
during an active demo window. I documented Portal and CLI steps, secrets passed as Container
App *secrets* referenced by env vars (never baked into the image), a budget alert, and a
one-command teardown. Honestly: **none of this is deployed yet.** The Dockerfile currently
still packages the Streamlit app — updating it to serve `uvicorn api.main:app` and to build
the React app is the concrete next step before I can actually execute this guide.

### Likely Follow-Up
What's the downside of scale-to-zero for this specific app?

### Strong Follow-Up Direction
Cold starts and session resets — conversation state is SQLite-backed, so a cold wake loses
in-memory/ephemeral state unless I mount Azure Files or keep min-replicas at 1 during demos.

### Red Flags to Avoid
Claiming it's deployed, inventing exact Azure prices, baking secrets into the image, or
describing the new topology as if it's already live.

---

## 14. Question
How do you observe this system in production, and what did you deliberately not add?

### What the Interviewer Is Testing
Observability judgment and restraint.

### Strong Answer
Two layers, unchanged by the migration since both sit below the API/UI split. A local,
secret-free JSONL ledger (`telemetry.py`) records route, latency, token counts, and
estimated cost per run using OpenAI usage callbacks and a central pricing table — that's my
cost truth, and it now captures both the Streamlit and API code paths since they call the
same `record_run` helper. And LangSmith gives distributed traces of the graph, LLM calls,
and my custom content-safe spans for Qdrant, Tavily, and rerank; I verified via the SDK that
runs land in the `papeer` project. I deliberately did *not* add Prometheus/Grafana: even
with a real async API now, this is still a small, single-instance service, and a second
monitoring stack would add cost and ops overhead before there's a measured need. That
restraint is itself a decision I can defend — add monitoring when a metric justifies it, not
by default.

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
Today, most external failures surface as exceptions; ingestion is wrapped in try/except in
both clients (rendered as a Streamlit error or an HTTP 400 from the API), and empty
retrieval correctly returns an honest "I don't know" now that the gating bug is fixed. But
there is still no retry/backoff, timeout tuning, or circuit breaker around the LLM/search/
vector calls — that gap is unchanged by the migration. So a transient OpenAI blip can fail a
turn; in the streaming API specifically, it surfaces as an SSE `error` event rather than a
hang, which I verified live, but the request still just fails rather than recovering. What
*should* happen: bounded retries with exponential backoff on transient errors, explicit
timeouts, and graceful user-visible degradation (e.g., "web search is unavailable, answering
from the paper only"). For claim verification, which does two Tavily calls, I'd degrade to
one source rather than fail entirely.

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
Two candidates, and I'd actually lead with the newer one: choosing to spend this round's
effort on the FastAPI + React migration instead of deeper backend hardening (auth
enforcement, CI, multi-paper eval). I accepted that because the target is SWE roles where
full-stack breadth is directly evaluated, and the backend was already decoupled enough that
the migration was low-risk. The cost is that some "known gaps" (no CI, auth not yet live)
persisted through this round instead of closing. The other trade-off, from the retrieval
work: hybrid search plus a local cross-encoder reranker buys quality and costs latency. I
accepted it because the baseline evaluation showed retrieval was the weak link, and running
the rerank locally on CPU means the quality gain comes with zero marginal OpenAI cost. The
honest, updated result is a *partial and re-verified* win: faithfulness and chunk-efficiency
gains hold up on repeat measurement; the precise precision/recall deltas don't, at this
sample size (see Q7).

### Likely Follow-Up
Would an LLM-based reranker have been better?

### Strong Follow-Up Direction
Possibly higher quality but it adds per-query token cost and latency; I chose the local
cross-encoder to keep marginal cost at zero, and I'd A/B it if budget allowed.

### Red Flags to Avoid
Presenting either trade-off as free, or overstating the quality gain from the retrieval work.

---

## 17. Question
What technical debt would you pay down first, and why that order?

### What the Interviewer Is Testing
Prioritization, re-evaluated now that some earlier debt is actually paid off.

### Strong Answer
The answer-gating bug is fixed and re-verified (Q7), and I initialized git this round — so
two items from an earlier version of this list are done. What's left, in order: first, a CI
pipeline (`pytest` + `npm run build` on push) — there's a repo now but no automated check,
so regressions could land silently, and it's cheap to add. Second, actually executing the
Azure deployment with Easy Auth turned on — I have a full plan but zero live enforcement,
which is the real release gate, not a nice-to-have. Third, retry/backoff and timeouts around
external calls for reliability. I'd defer bigger items — multi-paper eval, Postgres-backed
state, filtered Qdrant collections — behind those, because correctness, repeatability, and
closing the auth gap come before new capability or scale nobody's asking for yet.

### Likely Follow-Up
Why CI before finishing the deployment, if deployment unblocks the whole demo story?

### Strong Follow-Up Direction
CI is cheaper and protects everything that comes after it, including the deployment itself —
I'd rather catch a broken build in a 2-minute Actions run than during a live demo.

### Red Flags to Avoid
Listing debt without prioritizing, or claiming the answer-gating bug is still open when it's fixed.

---

## 18. Question
If you restarted this project with more time, what would you redesign?

### What the Interviewer Is Testing
Architectural hindsight — and whether I recognize what I've already acted on.

### Strong Answer
The biggest item on an earlier version of this answer — "split the UI from a stateless
backend service" — is exactly what I did this round, so I got to test that hindsight for
real rather than just claim it. What I'd still change if starting over: build the CI
pipeline and the async graph variant from day one instead of retrofitting them, use a single
filtered Qdrant collection with session lifecycle management instead of per-session
collections, bake evaluation into CI as a quality gate across multiple documents, and design
the answer node to always prefer generating from retrieved evidence from the start — that
bug only existed because the fallback path was easy to write and easy to forget about. I
wouldn't change the core RAG-with-managed-embeddings approach, the cost-first discipline, or
the decision to keep the backend UI-framework-agnostic — that last one is precisely what
made this session's migration safe.

### Likely Follow-Up
What would you keep exactly as is?

### Strong Follow-Up Direction
The bounded agentic graph, the content-safe cost/trace observability, the local reranker,
and the decision to keep Streamlit alive as a reference client during the migration instead
of a risky big-bang cutover.

### Red Flags to Avoid
Redesigning everything (signals poor judgment), claiming credit for hindsight without
mentioning it's now actually implemented, or nothing (signals no reflection).

---

## 19. Question
Where does this go over the next six months to become production-grade?

### What the Interviewer Is Testing
Roadmap realism, updated for what's already done.

### Strong Answer
Immediate: add a CI pipeline (pytest + frontend build); update the Dockerfile to serve the
FastAPI app and the React build instead of Streamlit; execute the Azure deployment and
verify Easy Auth is actually enforced before sharing the URL. Near term: multi-seed
evaluation for statistically defensible retrieval numbers, an automated Playwright E2E test,
and retry/backoff around external calls. Medium term: move to a filtered single Qdrant
collection with lifecycle cleanup, consider Postgres-backed state if real concurrent usage
ever shows up, tune reranker latency, and broaden evaluation to multiple papers and
web/claim cases. Each step is validated by the evaluation, the cost ledger, and CI status
rather than by feel — and unlike six months ago, roughly half of this list (the FastAPI/React
migration, the bug fix, the auth *plan*) is already behind me, which is itself evidence the
sequencing works.

### Likely Follow-Up
How do you avoid this roadmap becoming a wish list?

### Strong Follow-Up Direction
Gate each item on a measurable outcome (a metric moves, a cost drops, coverage rises, a URL
goes live and is verified authenticated) and do the cheapest high-impact item first — which
is exactly how the CI item beat "add Postgres" in the ordering above.

### Red Flags to Avoid
Calendar promises or a laundry list with no sequencing.

---

## 20. Question
How do you know Papeer actually works well? Defend your evaluation methodology.

### What the Interviewer Is Testing
AI evaluation rigor and honesty, including honesty about the methodology's own limits.

### Strong Answer
I use DeepEval with five RAG metrics — Contextual Precision, Recall, Relevancy, Answer
Relevancy, and Faithfulness — on a small hand-authored, human-audited question set that
spans factual, numeric, multi-section, security, and *unanswerable* cases. It's run as a
controlled A/B: the same questions through baseline dense retrieval vs. hybrid+rerank, same
judge model and threshold, with fresh Qdrant collections and cleanup each run. I measure
cost — application tokens via usage callbacks and DeepEval's per-metric `evaluation_cost` —
so the whole A/B, plus a validation re-run after a bug fix, ran for well under a dollar. I'm
honest about the limits, and I found a new one directly through the re-run: at this sample
size (five cases), the fine-grained retrieval-metric deltas are within run-to-run noise, so
I report the reliable findings (the bug fix, chunk-count reduction) separately from the
noisy ones (precise precision/recall deltas) rather than blending them into one confident
number. That's a *stronger* claim than "the numbers went up," because it's calibrated. To
strengthen it further I'd broaden to multiple papers and multiple seeds, and put it in CI.

### Likely Follow-Up
LLM-as-judge is itself unreliable — why trust these numbers?

### Strong Follow-Up Direction
I don't trust them blindly: I fix the judge model/threshold for comparability, keep it to
relative A/B deltas rather than absolute claims, human-audit outliers (that's how I caught
the answer-gating bug), and now explicitly separate signal from noise using the repeat run.

### Red Flags to Avoid
Presenting the numbers as a definitive benchmark, hiding the single-document limitation or
the run-to-run variance finding, or claiming the judge is objective truth.

---

## 21. Question
You found a bug where streamed answers appeared duplicated. Walk me through how you found it and fixed it.

### What the Interviewer Is Testing
Debugging methodology for a subtle, framework-specific streaming bug — and whether testing was real.

### Strong Answer
After building the SSE endpoint, I did a live end-to-end test — upload a paper, ask a
question, watch the streamed answer — and the text was visibly doubled. I isolated it with
a small script comparing `graph.stream()` and `graph.astream()` side by side, both filtered
to the `generate_answer` node, and counted the emitted chunk types. `stream_mode="messages"`
turned out to emit both incremental `AIMessageChunk`s *and* a final aggregated `AIMessage`
for the same turn — so forwarding every chunk with content, as the original Streamlit code
did, silently duplicated the answer. It affected both clients identically, which told me it
was a LangGraph streaming-mode behavior, not something specific to my new API code. The fix
was a one-line type check: only forward `isinstance(chunk, AIMessageChunk)`. I fixed it in
both `api/chat.py` and `app.py`, restarted the server, and re-ran the same live test to
confirm a single clean answer.

### Likely Follow-Up
Why didn't a unit test catch this?

### Strong Follow-Up Direction
Because nothing was exercising the actual streaming path end-to-end — my unit tests cover
pure logic (history reconstruction, reranking), not I/O-shaped behavior like SSE framing;
this is exactly the gap an automated Playwright or streaming-response integration test
would close, which is now on my near-term list.

### Red Flags to Avoid
Describing this as something caught by code review — it wasn't; it was caught by running
the thing and looking at the actual output.

---

## 22. Question
Two more bugs turned up in the React app itself during testing. What were they, and what do they say about your process?

### What the Interviewer Is Testing
Whether "verified" means something real, and whether I can reason about async UI state.

### Strong Answer
Both were state-timing bugs, and both were found the same way — by actually driving the app
in a browser and reading what happened, not by reading the code and assuming it was right.
First: my `streaming` flag only flipped to `false` in a `finally` block after the underlying
SSE fetch fully closed, but the `done` event — which already finalizes the message into
state — can arrive slightly before the connection technically closes. For a brief moment
both the finished bubble and a "Thinking…" placeholder rendered together. The fix was to
flip `streaming` to `false` as soon as the `done` event is processed, not only when the
fetch resolves. Second: the `/btw` side-channel exchange is deliberately not persisted to
session history, matching the original Streamlit behavior — but in Streamlit it only ever
rendered for one script execution, so switching sessions naturally cleared it. In React,
that state lived in a hook at the App level and had no reason to reset on its own, so it
kept showing after a session switch. I fixed it by resetting the `/btw` state in a
`useEffect` keyed on the active session id. Neither bug was severe, but both are exactly the
kind of thing that erodes trust in a demo, and both were invisible from reading the code —
they only showed up under real interaction.

### Likely Follow-Up
How would you prevent this class of bug going forward?

### Strong Follow-Up Direction
An automated E2E test that drives a real browser through upload → ask → switch session →
`/btw` would exercise these exact paths on every change; short of that, I'd keep doing what
worked here — don't call a feature done until it's been driven live, not just typechecked.

### Red Flags to Avoid
Downplaying these as "trivial" without explaining the fix, or claiming they were caught by
TypeScript (they weren't — both were logically valid, type-correct code with a timing bug).
