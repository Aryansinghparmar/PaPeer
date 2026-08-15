FROM python:3.12

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ── Layer 1: install deps (cached until requirements.txt changes) ─────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Layer 2: pre-cache local retrieval models ─────────────────────────────────
# The BM25 sparse model and the cross-encoder reranker run locally (fastembed,
# ONNX/CPU). Baking them into the image means the container never downloads them
# at request time — reliable cold starts and no runtime network dependency for
# models. Adds ~90 MB to the image.
RUN python -c "from fastembed.rerank.cross_encoder import TextCrossEncoder; TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2'); from fastembed import SparseTextEmbedding; SparseTextEmbedding('Qdrant/bm25')"

# ── Layer 3: backend package ───────────────────────────────────────────────────
COPY backend/ backend/

# ── Layer 4: documents ─────────────────────────────────────────────────────────
COPY documents/ documents/

# ── Layer 5: application files (change most often — last for cache efficiency) ─
COPY app.py .
COPY evaluate.py .
COPY main.py .
COPY goldens.json .
COPY goldens_curated.json .
COPY sessions.json .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
