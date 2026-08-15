# Papeer — Research Paper Assistant

A conversational AI assistant for students and researchers to upload, explore, and verify academic papers through natural language chat.

---

## Project Description

Papeer is a Retrieval-Augmented Generation (RAG) application built with LangGraph, LangChain, and Streamlit. Users upload research papers (PDF, TXT, Markdown, web URL, or ArXiv ID) into isolated sessions, then ask questions about them. The system routes each query intelligently — answering directly from paper content, searching the web for current developments, or verifying whether a claim from a paper has been superseded by newer research.

---

## Target Users

- **Students** reading and trying to understand dense academic papers
- **Researchers** who want to quickly cross-reference claims across multiple papers
- **Literature reviewers** checking whether findings or methods from older papers still hold today
- **Anyone** who wants a conversational interface to a set of documents without manual reading

---

## Features

| Feature | Description |
|---|---|
| **Paper Q&A** | Ask questions about uploaded papers; the system retrieves relevant chunks and generates grounded answers |
| **Claim Verification** | Ask the assistant to verify a claim — it searches the web and ArXiv to determine if the claim is current or superseded, and returns links to newer papers if applicable |
| **Web Search** | For questions about current developments or explicit search requests, live Tavily results are incorporated |
| **Direct Answers** | General knowledge questions are answered without retrieval or web calls |
| **`/btw` Command** | A side-channel for off-topic questions outside the session context. The LLM decides to answer directly or search the web. These exchanges are **not stored in session history** |
| **Multi-session UI** | Open multiple independent sessions simultaneously, each with its own paper collection and conversation history |
| **Auto Session Naming** | Session titles are automatically generated (3–5 words) from the first message using the LLM |
| **Multiple Paper Sources** | Load papers via file upload (PDF, TXT, MD), web URL, or ArXiv ID/title search |
| **Graph State Inspector** | Each assistant turn exposes an expandable JSON view of the LangGraph state for debugging |
| **Streaming Responses** | Assistant responses stream token-by-token with a cursor animation |

---

## How to Use

### 1. Start a session
Launch the app and a default session is created automatically. Use **New Chat** in the sidebar to start additional sessions.

### 2. Upload papers
In the sidebar, choose one of three loading methods:
- **File Upload** — drag and drop a PDF, TXT, or MD file
- **Web URL** — paste one or more URLs (one per line)
- **ArXiv** — enter a paper title or ArXiv ID (e.g. `2303.08774`)

Loaded papers are listed under "Loaded Papers" in the sidebar.

### 3. Ask questions
Type in the chat input. Example queries:
- *"What methodology does the paper use for evaluation?"*
- *"Verify the claim that encoder-decoder models are the best approach for translation."*
- *"What are the latest developments in diffusion models?"*

### 4. Use `/btw` for off-topic questions
Prefix any message with `/btw` to ask a question outside the current paper context. These exchanges are not saved to the session:
```
/btw What is the difference between RLHF and DPO?
```

---

## Installation

Papeer uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Clone the repository
git clone <repo-url>
cd rag-papeer-project

# Install all dependencies
uv sync

# Copy the example env file and fill in your keys
cp .env.example .env

# Run the Streamlit app
uv run streamlit run app.py
```

To add a new dependency:
```bash
uv add <package-name>
```

To run a backend module directly (useful during development):
```bash
uv run python -m backend.<module_name>
```

---

## Required API Keys

All keys are loaded from a `.env` file in the project root via `python-dotenv`.

| Variable | Purpose | Where to Get It |
|---|---|---|
| `OPENAI_API_KEY` | LLM inference (`gpt-5-mini`) and embeddings (`text-embedding-3-small`) | [platform.openai.com](https://platform.openai.com) |
| `TAVILY_API_KEY` | Web search for current developments and claim verification | [tavily.com](https://tavily.com) |
| `QDRANT_URL` | Qdrant Cloud endpoint for the vector store | [cloud.qdrant.io](https://cloud.qdrant.io) |
| `QDRANT_API_KEY` | Authentication for Qdrant Cloud | [cloud.qdrant.io](https://cloud.qdrant.io) |

`.env` file format:
```env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
```

---

## Architecture

```
app.py (Streamlit UI)
│
├── backend/rag_graph.py       — LangGraph RAG workflow (router → retrieve/verify/direct → answer)
├── backend/btw_handler.py     — Off-topic /btw handler (streaming, not stored in history)
├── backend/vector_store.py    — Qdrant Cloud store: dense or hybrid (dense+BM25) retrieval + rerank
├── backend/reranker.py        — Local cross-encoder reranker (fastembed, ONNX/CPU, no OpenAI cost)
├── backend/paper_loader.py    — Multi-source paper loader (PDF, TXT, MD, URL, ArXiv)
├── backend/config.py          — Central config (models, retrieval mode, rerank, cost estimates)
├── backend/telemetry.py       — Secret-free local run/cost ledger (observability/runs.jsonl)
├── backend/tracing.py         — Content-safe LangSmith spans (Qdrant, Tavily, rerank)
└── backend/models.py          — Pydantic models for routing and structured LLM outputs
```

### RAG Graph Decision Flow

```
User Query
    │
    ▼
 Router (LLM)
    │
    ├── direct_answer ──────────────────────────► Generate Answer
    │
    ├── retrieve ──► Agent (retriever + web tools) ──► Relevancy Check
    │                        │                              │
    │                        │◄── Query Rewrite (max 3) ────┘
    │                        └──────────────────────────────► Generate Answer
    │
    └── verify_claim ──► Web Search + ArXiv Search ──► Verdict + Paper Links
```

---

## How the Project Is Production Optimized

| Optimization | Details |
|---|---|
| **Hybrid retrieval + reranking** | Improved retrieval fuses dense vectors with a BM25 sparse vector (Qdrant hybrid), fetches a candidate pool, then a local cross-encoder (`fastembed`, ONNX/CPU) reranks and keeps the top N. Runs entirely locally — no OpenAI cost. Toggle via `RETRIEVAL_MODE` / `RERANK_ENABLED` |
| **Embedding cache** | `CacheBackedEmbeddings` writes to `./embedding_cache/` so identical text is never re-embedded across sessions — reduces OpenAI API calls and latency |
| **Session isolation** | Each session gets its own Qdrant collection (`papeer_{session_id}`) and a separate LangGraph SQLite checkpointer thread — prevents cross-session data leakage |
| **Graph caching** | The LangGraph graph is built once with `@st.cache_resource` and reused across all Streamlit reruns |
| **Streaming responses** | `graph.stream()` is used with message mode so responses appear token-by-token rather than waiting for the full generation |
| **Session persistence** | `sessions.json` persists session metadata; SQLite stores full conversation state — app restarts restore the previous session seamlessly |
| **Temp file cleanup** | Uploaded files are written to a temp path, processed, then deleted regardless of success or failure |
| **Async evaluation** | The evaluation pipeline uses throttled concurrency (3 workers, 5 s throttle) to stay within API rate limits |
| **ArXiv reliability** | Claim verification uses two targeted Tavily searches (general web + `site:arxiv.org`) instead of the `arxiv` Python library, which had reliability issues |

---

## Constraints and Why

| Constraint | Why |
|---|---|
| **Max 3 query rewrites** | The RAG graph caps query rewrites at 3 retries before falling back to a plain LLM answer. Without this cap, ambiguous or unanswerable queries would loop indefinitely, burning API tokens and blocking the user |
| **Chunk size 1000 / overlap 200** | Balances retrieval precision (smaller = more focused) against context preservation across chunk boundaries. The 200-char overlap ensures sentences split across chunks are still retrievable |
| **Tavily max 3 results for `/btw`** | Keeps the context window manageable for side-channel queries that are intentionally lightweight and unsaved |
| **`/btw` exchanges not stored** | These are deliberately out-of-context questions. Storing them would pollute session history and confuse the LLM's understanding of the paper-focused conversation |
| **Session-scoped Qdrant collections** | Prevents papers from one session leaking into another. Each collection is namespaced by session UUID |
| **Claim verification uses two searches** | A general web search catches blog posts and news; an `arxiv.org`-targeted search catches academic superseding work. One search alone misses one of these two important source types |
| **Candidate pool → rerank to top N** | Improved retrieval fetches a larger candidate pool (`RETRIEVAL_CANDIDATE_K`, default 20) so the cross-encoder has enough to choose from, then keeps only the top N (`RERANK_TOP_N`, default 4). Fewer, higher-quality chunks reduce noise, prompt length, and evaluation cost. Baseline dense retrieval uses the LLM-chosen `k` (1–10) |

---

## Evaluation

Papeer includes an automated RAG evaluation pipeline (`evaluate.py`) built on [DeepEval](https://github.com/confident-ai/deepeval), designed as a **controlled A/B** between the baseline (dense) retrieval and the improved (hybrid + cross-encoder rerank) retrieval on the same question set.

### Metrics (threshold: 0.7)

| Metric | What It Measures |
|---|---|
| **Contextual Precision** | Are the retrieved chunks relevant to the query? |
| **Contextual Recall** | Does the retrieved context cover all expected information? |
| **Contextual Relevancy** | Is the context relevant to both the input and the expected output? |
| **Answer Relevancy** | Does the generated answer actually address the question? |
| **Faithfulness** | Is the answer grounded in the retrieved context (no hallucination)? |

Every run also records **measured** OpenAI cost (application + DeepEval judge, via usage callbacks and DeepEval's per-metric `evaluation_cost`), latency, and average retrieved-chunk count. Reranking and BM25 run locally on CPU and add no OpenAI cost.

### Running the A/B evaluation

The retrieval strategy is read from config at import, so each side is run as its own process for a clean, reproducible comparison. Always run a 1-case probe first to size the spend.

```bash
# 1) cost probe (1 case)
RETRIEVAL_MODE=dense RERANK_ENABLED=false uv run python evaluate.py --goldens curated --limit 1 --label probe --output eval_probe.json

# 2) baseline (dense) and improved (hybrid + rerank)
RETRIEVAL_MODE=dense  RERANK_ENABLED=false uv run python evaluate.py --goldens curated --limit 5 --label baseline --output eval_baseline.json
RETRIEVAL_MODE=hybrid RERANK_ENABLED=true  uv run python evaluate.py --goldens curated --limit 5 --label improved --output eval_improved.json

# 3) before/after report (no API calls)
uv run python evaluate.py --compare eval_baseline.json eval_improved.json eval_comparison.md
```

- Questions come from `goldens_curated.json` — a small **hand-authored, human-audited** set (factual, numeric, multi-section, security, unanswerable). Use `--goldens synthetic` for the auto-generated set instead.
- Per-test scores, pass/fail, failure reasons, and measured cost are written to the output JSON; `eval_comparison.md` holds the before/after summary and an honest interpretation.
- **Latest measured result (5 cases):** hybrid + rerank improved Contextual Precision (0.849 → 0.922), Recall (0.933 → **1.000**), and Answer Relevancy (0.95 → **1.000**), and cut retrieved chunks per query (13.2 → 8.8). Contextual Relevancy stayed flat and Faithfulness dipped on one case — traced to a downstream answer-gating fallback, not retrieval. See `eval_comparison.md` for the full, honest write-up. A genuine but *partial* win with a clear next step.
- `eval_results.json` is the **original student synthetic run** (older code) and is kept for reference only — it is not a result of the current retrieval pipeline.
