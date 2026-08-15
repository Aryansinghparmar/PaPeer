# Three-Minute Project Introduction

## Why

Research papers are slow to read and hard to cross-check. If I only need one method, one
number, or want to know whether a paper's conclusion still holds, I still have to scan a
long PDF and then go hunting through newer literature. Generic chatbots don't help much
because they aren't grounded in the specific paper and they hallucinate. I built Papeer so
a student or researcher can *talk to* a paper — ask grounded questions, and explicitly
check whether a claim has been superseded by more recent work.

## What

Papeer is a Retrieval-Augmented Generation assistant. You upload a paper — a PDF, a URL, or
an ArXiv ID — into an isolated session, and then you ask questions in natural language. An
LLM router classifies each question into one of three paths: answer from the uploaded paper
using retrieval, verify a claim by searching the live web and ArXiv, or answer a general
question directly. Answers stream back token by token, and every turn exposes the underlying
workflow state so I can see exactly what the system did. You can run several independent
sessions, each with its own papers and history, and there's a `/btw` side channel for
off-topic questions that deliberately don't pollute the paper conversation.

## How

The app is a single Streamlit process that drives a LangGraph workflow — a router, a
tool-calling retrieval agent, a relevancy gate with a bounded query-rewrite loop, a
claim-verification branch, and a final answer node. Documents are chunked, embedded with
OpenAI's small embedding model, and stored per session in Qdrant Cloud. The engineering
decision I'm most confident about is the retrieval upgrade: I added hybrid search — dense
vectors plus BM25 — and a local cross-encoder reranker that runs on CPU through fastembed,
so it improves quality with zero extra OpenAI cost per query. I proved it with a controlled
before/after evaluation using DeepEval, and I instrumented real token cost so the whole A/B
ran for under a dollar. The honest result is a *partial* win: precision, recall, and answer
relevancy improved and retrieved-chunk noise dropped, but one metric stayed flat and
faithfulness dipped on a single case. When I read the per-case data, I found the reranker
was actually correct there — the bug was downstream, in an answer-gating fallback that says
"I couldn't find anything" even when good chunks were retrieved. That's my current top fix.

## What Now

It's a working prototype, not a production system. The biggest limitations I'd call out
proactively are that there's no authentication — it assumes a single local user — and the
evaluation is still on one document, so it's a signal rather than a benchmark. The most
valuable next steps are fixing that answer-gating bug and re-measuring, then adding auth and
a budget cap before deploying it to Azure Container Apps, which I've already containerized
and written a deployment guide for. I'm happy to go deeper into the retrieval design, the
evaluation methodology, or the cost trade-offs.

## Thirty-Second Version

Papeer is a RAG assistant that lets you chat with research papers and verify whether their
claims still hold. You upload a paper, and a LangGraph workflow routes each question to
retrieval, web-based claim verification, or a direct answer, backed by Qdrant and OpenAI. I
recently added hybrid search plus a local cross-encoder reranker and proved the improvement
with a cost-controlled DeepEval A/B — an honest partial win that also surfaced a downstream
bug I'm fixing. It's a working prototype; next steps are auth and deploying to Azure.

## Key Points to Remember

- Problem: papers are slow to read and hard to fact-check against newer work.
- Three routes: retrieve from paper, verify claim (web + ArXiv), direct answer.
- Stack: Streamlit + LangGraph + Qdrant + OpenAI + Tavily.
- Strongest decision: hybrid + local cross-encoder rerank — better quality, no per-query cost.
- Proof: cost-controlled DeepEval A/B (whole run < $1), measured, reproducible.
- Honesty: a *partial* win that exposed a downstream answer-gating bug (my top fix).
- Limitations to own: no auth, single-document eval, prototype not deployed.
- Next: fix the gate + re-measure, add auth + budget cap, deploy to Azure Container Apps.
