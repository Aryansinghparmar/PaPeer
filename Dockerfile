# Papeer: FastAPI backend + built React SPA in one image, served same-origin.
#
#   docker build -t papeer:local .
#   docker compose up            # see docker-compose.yml and docs/DEPLOYMENT.md
#
# Secrets are never baked in: .env is excluded by .dockerignore and every key
# is passed at runtime.

# ── Stage 1: build the React SPA ─────────────────────────────────────────────
FROM node:22-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
# Empty base URL: the SPA calls /api on the same origin that serves it.
ENV VITE_API_BASE_URL=""
RUN npm run build

# ── Optional: pre-downloaded fastembed model cache ───────────────────────────
# Empty by default, so the models are downloaded from Hugging Face below. On a
# network whose proxy intercepts TLS to huggingface.co, pass a local cache:
#   docker build --build-context fastembed_seed=/path/to/fastembed_cache .
FROM scratch AS fastembed_seed

# ── Stage 2: Python runtime ───────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FASTEMBED_CACHE_PATH=/opt/fastembed_cache
WORKDIR /app

# Layer 1: runtime dependencies only (requirements.txt is the uv.lock export
# without the dev/eval groups), cached until requirements.txt changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Layer 2: pre-cache the local retrieval models. BM25 sparse embeddings and the
# cross-encoder reranker run on CPU via fastembed (ONNX). Baking them in means
# no model download at request time and reliable cold starts. fastembed reads
# the cache first and only downloads what is missing.
COPY --from=fastembed_seed / /opt/fastembed_cache/
RUN python -c "from fastembed.rerank.cross_encoder import TextCrossEncoder; TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2'); from fastembed import SparseTextEmbedding; SparseTextEmbedding('Qdrant/bm25')" \
    && chmod -R a+rX /opt/fastembed_cache

# Layer 3: unprivileged user that owns the only writable directory.
RUN useradd --create-home --uid 10001 papeer \
    && mkdir -p /data \
    && chown papeer:papeer /data

# Layer 4: application code and the built SPA (change most often).
COPY backend/ backend/
COPY api/ api/
COPY --from=frontend /frontend/dist/ static/

# All mutable state lives under /data (mount a volume there to keep it).
ENV CHECKPOINT_DB_PATH=/data/checkpoints.db \
    SESSIONS_FILE=/data/sessions.json \
    EMBEDDING_CACHE_DIR=/data/embedding_cache \
    METRICS_DIR=/data/observability \
    STATIC_DIR=/app/static \
    HF_HUB_OFFLINE=1

USER papeer
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

# One worker on purpose: sessions, rate limits and the daily budget are
# in-process state and the checkpoint store is SQLite (see docs/DEPLOYMENT.md).
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers"]
