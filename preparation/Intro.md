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
question directly. Answers stream back token by token over a real API, and every turn
exposes the underlying workflow state so I can see exactly what the system did. You can run
several independent sessions, each with its own papers and history, and there's a `/btw`
side channel for off-topic questions that deliberately don't pollute the paper conversation.

## How

The backend is a FastAPI service wrapping a LangGraph workflow — a router, a tool-calling
retrieval agent, a relevancy gate with a bounded query-rewrite loop, a claim-verification
branch, and a final answer node — and it streams responses to a React and TypeScript
frontend over Server-Sent Events. That split happened deliberately: the backend logic was
already framework-agnostic, so I wrapped it in a real REST+SSE API and built a React SPA
client, rather than staying on the single-process Streamlit app I started with — I kept
Streamlit working too, as a reference client, but the API and the SPA are the primary
surface now. Documents are chunked, embedded with OpenAI's small embedding model, and
stored per session in Qdrant Cloud, using hybrid search — dense vectors plus BM25 — with a
local cross-encoder reranker that runs on CPU through fastembed, so it improves quality
with zero extra OpenAI cost per query.

I proved the retrieval upgrade with a controlled before/after evaluation using DeepEval, and
I instrumented real token cost so the whole thing ran for well under a dollar. The honest
result had two layers. First: precision, recall, and answer relevancy improved, and I found
a real downstream bug — an answer node that refused to answer even when it had retrieved the
right chunks — which I fixed and re-verified. Second, when I re-ran the same retrieval
config to confirm the fix, the fine-grained precision numbers themselves swung as much as
the original "improvement" — a small-sample statistical-power problem, not a broken system.
I reported that honestly rather than keep the flattering number. While building the React
client I also found and fixed two more real bugs — an SSE stream that duplicated every
answer, and a UI state race — just by actually running the app and reading what it did.

## What Now

It's a working prototype with a real API and a real frontend, not a deployed production
system yet. The most valuable next steps, in order, are: turn on Azure's platform-managed
authentication — Entra ID through Azure Easy Auth — before the app goes live, since right
now nothing is enforcing access; add a CI pipeline so tests and the frontend build run on
every push; and broaden the evaluation with more seeds and more documents so the retrieval
claims are statistically solid, not just directionally true. I've already containerized the
app and written the Azure deployment plan for Container Apps plus Static Web Apps. I'm happy
to go deeper into the retrieval design, the evaluation methodology, the API/SSE
architecture, or the cost trade-offs.

## Thirty-Second Version

Papeer is a RAG assistant that lets you chat with research papers and verify whether their
claims still hold. You upload a paper, and a LangGraph workflow routes each question to
retrieval, web-based claim verification, or a direct answer, backed by Qdrant and OpenAI,
served through a FastAPI backend with SSE streaming to a React and TypeScript frontend. I
added hybrid search plus a local cross-encoder reranker and proved the improvement with a
cost-controlled DeepEval A/B, then found and fixed three real bugs — including one the
evaluation itself surfaced — by actually running the system end to end. It's a working
prototype with a real API and SPA; next steps are turning on Azure auth and deploying.

## Key Points to Remember

- Problem: papers are slow to read and hard to fact-check against newer work.
- Three routes: retrieve from paper, verify claim (web + ArXiv), direct answer.
- Stack: React + TypeScript SPA, FastAPI backend (SSE streaming), LangGraph, Qdrant, OpenAI, Tavily.
- Strongest decision: hybrid + local cross-encoder rerank — better quality, no per-query cost.
- Migration: decoupled backend let me wrap Streamlit's logic in FastAPI + React without
  touching the AI code — Streamlit still works as a reference client.
- Proof: cost-controlled DeepEval A/B (whole run under a dollar), measured, reproducible.
- Honesty: fixed a real answer-gating bug, then found the fine-grained retrieval deltas were
  within run-to-run noise at this sample size — reported both, not just the flattering one.
- Bugs found by actually running it: SSE token duplication, a UI streaming-state race, a
  session-switch state leak — none caught by static review.
- Limitations to own: not deployed, auth planned but not yet enforced, no CI yet, thin
  single-document evaluation.
- Next: turn on Azure Easy Auth, add CI, multi-seed evaluation, then deploy.
