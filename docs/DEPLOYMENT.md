# Deploying Papeer

Papeer ships as **one container**: FastAPI serves the REST + SSE API under
`/api` and the built React SPA at `/`, on port 8000, from the same origin (no
CORS in production). This guide covers local development, Docker Compose, and
one recommended cloud path, Azure Container Apps. Nothing here has been deployed
by the assistant; the owner runs the cloud steps.

External services (always remote, never in the container): OpenAI (LLM and
embeddings), Qdrant Cloud (vectors), Tavily (web search), arXiv (public API),
and optionally LangSmith (traces).

---

## 1. Local development

```bash
uv sync                                   # runtime + dev + eval groups into .venv
cp .env.example .env                      # fill in OPENAI_API_KEY, TAVILY_API_KEY, QDRANT_URL, QDRANT_API_KEY
uv run uvicorn api.main:app --reload --port 8000

# second terminal
cd frontend && npm ci && npm run dev      # http://localhost:5173, proxies /api and /health to :8000
```

`AUTH_MODE` is unset, so it defaults to `none` (every caller is `local-dev`). If
`frontend/dist` exists (`npm run build`), the API also serves the SPA at
http://localhost:8000.

Tests and evaluation:

```bash
uv run pytest -q          # offline: fake graph, fake DNS/HTTP, dummy keys (tests/conftest.py)
uv run python evaluate.py # paid: calls OpenAI; see README for the A/B settings
```

`requirements.txt` is the **runtime-only** export of `uv.lock` used by the
Docker image. After changing dependencies:

```bash
uv lock
uv export --no-default-groups --no-hashes --format requirements-txt --no-emit-project -o requirements.txt
```

---

## 2. Docker Compose (single machine)

```bash
cp .env.example .env
# set PAPEER_ACCESS_CODE, e.g.: python -c "import secrets; print(secrets.token_urlsafe(24))"
docker compose up --build
# open http://localhost:8000 and enter the access code
```

- Compose defaults `AUTH_MODE` to `api_key`: every API call needs
  `Authorization: Bearer <PAPEER_ACCESS_CODE>`. The app refuses to start in
  `api_key` mode without a code of 8+ characters.
- State (`checkpoints.db`, `sessions.json`, `embedding_cache/`,
  `observability/runs.jsonl`) lives in the named volume `papeer-data` at
  `/data`. Compose pins those paths, so local-dev paths in `.env` cannot move
  them. This is a local Docker volume, so SQLite on it is fine.
- The image runs as an unprivileged user and contains no secrets (`.env` is
  in `.dockerignore`). The BM25 and reranker models are baked in at build
  time, and `HF_HUB_OFFLINE=1` stops any model download at runtime.
- If your network's proxy intercepts TLS to huggingface.co, the model step
  fails with `CERTIFICATE_VERIFY_FAILED`. Build with a pre-downloaded fastembed
  cache instead: `docker build --build-context fastembed_seed=<dir> -t papeer:local .`

Image without Compose:

```bash
docker build -t papeer:local .
docker run --rm -p 8000:8000 --env-file .env -e AUTH_MODE=api_key -v papeer-data:/data papeer:local
```

---

## 3. Recommended cloud path: Azure Container Apps

### Why this shape

| Setting | Value | Reason |
|---|---|---|
| Plan | Consumption | Billed per second of use; scale to zero costs no compute |
| Replicas | min 0, **max 1** | Sessions, rate limits and the daily budget are per-process memory, and checkpoints are SQLite. A second replica would split state |
| Workers | 1 (in the image `CMD`) | Same reason |
| Ingress | external, HTTP, target port 8000 | SSE is plain HTTP/1.1 chunked streaming. The app sends a `: ping` comment every 15 s, so idle-connection timeouts do not cut long claim-verification turns. Check the current ingress request-timeout limit in the Container Apps docs. `REQUEST_TIMEOUT_SECONDS` (180 s) should stay below it |
| Storage | **ephemeral container filesystem** | See the SQLite caveat below |
| Auth | Easy Auth + `AUTH_MODE=easyauth`, or `AUTH_MODE=api_key` | Never `none` on a public URL |

### The SQLite caveat (read before adding a volume)

Do **not** put `checkpoints.db` (or `sessions.json`) on an **Azure Files (SMB)**
share. SQLite's WAL mode, used by the LangGraph checkpointer, needs shared
memory and does not work over a network filesystem. SQLite's own
documentation also warns that file locking is unreliable on network
filesystems, which risks corruption:
[sqlite.org/wal.html](https://www.sqlite.org/wal.html),
[sqlite.org/useovernet.html](https://www.sqlite.org/useovernet.html).

So the demo uses the container's **ephemeral** storage:

- Chat history, session names and the local cost ledger **reset** when the
  replica scales to zero or a new revision starts.
- Uploaded papers survive in Qdrant Cloud, but they are orphaned once their
  session id is gone.
- The daily budget total also restarts from zero with a fresh replica. The
  OpenAI project budget (below) is the hard backstop.

For production, move checkpoints to Postgres with
`langgraph-checkpoint-postgres` (`AsyncPostgresSaver`), for example Azure
Database for PostgreSQL Flexible Server. Move the session table there too,
and rate limits and spend to Redis or the database. Then `max-replicas` and
`--workers` can go above 1.

### Steps (Azure CLI)

Prerequisites: Azure CLI (`az login`), Docker, a Docker Hub account (or Azure
Container Registry, see the note), and your keys.

```bash
# 0. Variables
RG=papeer-rg
LOC=centralindia                 # any region that offers Container Apps
ENV=papeer-env
APP=papeer
IMAGE=docker.io/<DOCKERHUB_USER>/papeer:v1

# 1. Build and push. The image contains no secrets, so a public repo is fine.
docker build -t papeer:local .
docker login
docker tag papeer:local $IMAGE
docker push $IMAGE

# 2. One-time setup
az extension add --name containerapp --upgrade
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights

# 3. Resource group + environment (creates a Log Analytics workspace)
az group create --name $RG --location $LOC
az containerapp env create --name $ENV --resource-group $RG --location $LOC

# 4. The app: scale to zero, at most one replica, secrets referenced by env vars
az containerapp create \
  --name $APP --resource-group $RG --environment $ENV \
  --image $IMAGE \
  --ingress external --target-port 8000 --transport http \
  --min-replicas 0 --max-replicas 1 \
  --cpu 1.0 --memory 2.0Gi \
  --secrets openai-key=<OPENAI_API_KEY> tavily-key=<TAVILY_API_KEY> \
            qdrant-url=<QDRANT_URL> qdrant-key=<QDRANT_API_KEY> \
            access-code=<PAPEER_ACCESS_CODE> \
  --env-vars OPENAI_API_KEY=secretref:openai-key TAVILY_API_KEY=secretref:tavily-key \
             QDRANT_URL=secretref:qdrant-url QDRANT_API_KEY=secretref:qdrant-key \
             PAPEER_ACCESS_CODE=secretref:access-code \
             AUTH_MODE=api_key DAILY_BUDGET_USD=1.00 \
             RETRIEVAL_MODE=hybrid RERANK_ENABLED=true LANGSMITH_TRACING=false

# 5. URL
az containerapp show --name $APP --resource-group $RG \
  --query properties.configuration.ingress.fqdn -o tsv
curl -s https://<fqdn>/health          # {"status":"ok"}
curl -s https://<fqdn>/api/auth/mode   # {"mode":"api_key"}
```

Notes:

- `--secrets` values end up in your shell history. Clear it afterwards, or
  use Key Vault references instead of literal values:
  `--secrets openai-key=keyvaultref:<secret-uri>,identityref:<managed-identity-id>`.
  The app's managed identity needs read access to the vault's secrets.
- To get LangSmith traces, add a `langsmith-key` secret,
  `LANGSMITH_API_KEY=secretref:langsmith-key` and `LANGSMITH_TRACING=true`.
- The Dockerfile `HEALTHCHECK` is a Docker feature. Container Apps uses its
  own probes; an HTTP liveness probe on `/health`, port 8000, matches it.
- Azure Container Registry is the alternative to Docker Hub. Even the Basic
  tier has a monthly charge (check current pricing), so Docker Hub is used
  here.
- Redeploy: push a new tag, then
  `az containerapp update --name $APP --resource-group $RG --image <new-image>`.
- Stop compute without deleting anything:
  `az containerapp update --name $APP --resource-group $RG --min-replicas 0 --max-replicas 0`.
  Remove everything: `az group delete --name $RG --yes --no-wait`.

### Option A: shared access code (`AUTH_MODE=api_key`)

This is the simplest option and what step 4 configures. Give reviewers the URL
and the code. Everyone with the code is one principal (`access-code`), so they
share one session list. It is a demo gate, not user accounts. Rotate the code
with `az containerapp secret set` and restart the revision.

### Option B: Microsoft sign-in (Easy Auth + `AUTH_MODE=easyauth`)

Easy Auth is the platform's built-in authentication. A sidecar gates requests
before they reach the container and forwards the signed-in identity as headers.
Papeer decodes `X-MS-CLIENT-PRINCIPAL` (base64 JSON claims) and uses the
user's object id as the session owner, so each user sees only their own
sessions.

**This mode is only safe when every request reaches the app through Easy
Auth.** Unauthenticated requests must be rejected or redirected by the
platform. If the container were reachable any other way, anyone could send a
forged `X-MS-CLIENT-PRINCIPAL` header.

Configure it multi-tenant with personal accounts, so reviewers can sign in
with their own Microsoft account. A single-tenant registration only admits
accounts in your own directory.

Portal (simplest, because it also creates the app registration and client secret):

1. Container App, then **Settings → Authentication → Add identity provider**.
2. Provider **Microsoft**. Let it create a new app registration.
3. **Supported account types:** "Any Microsoft Entra directory and personal
   Microsoft accounts".
4. **Restrict access:** *Require authentication*. **Unauthenticated requests:**
   *HTTP 302 redirect*, so the browser goes to the login page and the SPA then
   calls `/api` with the auth session cookie on the same origin.
5. Save, then set `AUTH_MODE=easyauth` on the app (new revision):
   `az containerapp update --name $APP --resource-group $RG --set-env-vars AUTH_MODE=easyauth`.

CLI equivalent (you also need to create and store a client secret for the
registration; the Portal does that for you):

```bash
az ad app create --display-name "papeer-auth" \
  --sign-in-audience AzureADandPersonalMicrosoftAccount
CLIENT_ID=<appId-from-above>
az containerapp auth microsoft update --name $APP --resource-group $RG \
  --client-id $CLIENT_ID --client-secret <client-secret> \
  --issuer https://login.microsoftonline.com/common/v2.0
az containerapp auth update --name $APP --resource-group $RG \
  --unauthenticated-client-action RedirectToLoginPage \
  --redirect-provider azureactivedirectory
```

Easy Auth has no separate charge, and the Entra ID free tier covers sign-in
for a demo (check current pricing).

### Cost controls

- **OpenAI:** create a dedicated project for Papeer with its own key. Set its
  monthly budget and usage alerts in the OpenAI dashboard (check whether your
  account offers a hard cap). This is the real ceiling, because the app's
  numbers are estimates.
- **`DAILY_BUDGET_USD`** (default 1.00): before any chat, `/btw` or ingestion
  call, the API compares today's (UTC) estimated spend with this cap and
  answers `429 budget_exceeded` once it is reached. The estimate comes from the
  token usage that LangChain reports, priced with the table in
  `backend/config.py`. A model missing from that table counts as $0.
- **Rate limits:** `RATE_LIMIT_CHAT` / `_BTW` / `_DOCUMENTS` / `_SESSIONS`
  per principal per `RATE_LIMIT_WINDOW_SECONDS`. They are in memory, per
  process.
- **Tavily:** set the plan's monthly credit limit in the Tavily dashboard.
  The free tier has a fixed monthly credit allowance (check current pricing).
  Claim verification uses Tavily, and so does `/btw` when it needs the web.
- **Qdrant Cloud:** the free cluster has a fixed storage size (check current
  limits). Deleting a session deletes its collection.
- **Azure:** Container Apps consumption has a monthly free grant of
  vCPU-seconds, GiB-seconds and requests. Scale-to-zero replicas use no
  compute. Log Analytics ingestion is billed separately. See
  [Container Apps pricing](https://azure.microsoft.com/pricing/details/container-apps/)
  and [billing](https://learn.microsoft.com/azure/container-apps/billing),
  and check current pricing. Add a Cost Management budget alert, for example
  $10/month at 80%, before sharing the URL. `min-replicas 1` removes the cold
  start but bills continuously, so use it only during a demo window.

### Configuration reference

| Variable | Deploy as | Notes |
|---|---|---|
| `OPENAI_API_KEY`, `TAVILY_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY` | secret | required |
| `PAPEER_ACCESS_CODE` | secret | required when `AUTH_MODE=api_key` |
| `LANGSMITH_API_KEY` | secret | optional, with `LANGSMITH_TRACING=true` |
| `AUTH_MODE` | env | `api_key` or `easyauth` in the cloud |
| `DAILY_BUDGET_USD` | env | 0 disables |
| `RATE_LIMIT_*`, `MAX_*`, `HEARTBEAT_SECONDS`, `REQUEST_TIMEOUT_SECONDS` | env | defaults in `api/settings.py`, `backend/paper_loader.py` |
| `CORS_ORIGINS` | env | not needed: the SPA is same-origin |
| `CHECKPOINT_DB_PATH`, `SESSIONS_FILE`, `EMBEDDING_CACHE_DIR`, `METRICS_DIR`, `STATIC_DIR` | baked into the image | under `/data` and `/app/static` |

### Troubleshooting

- **First request slow or 502 after idle:** cold start from zero (image pull
  plus model load). Retry after ~30-60 s, or use `min-replicas 1` for a demo.
- **OOM / restarts:** onnxruntime plus PyMuPDF need memory. Keep 2 GiB.
- **Every API call is 401:** wrong or missing access code, or `AUTH_MODE=easyauth`
  without Easy Auth enabled.
- **Sessions vanished:** expected after scale-to-zero (ephemeral storage, see
  above).
- **Logs:** `az containerapp logs show --name $APP --resource-group $RG --follow`.
  Stream errors are logged with the same `error id` the client shows.

---

## 4. Simpler alternative (not Azure)

**Fly.io:** `fly launch` with the existing Dockerfile (internal port 8000).
Then `fly volumes create papeer_data --size 1` and mount it at `/data`. That is
a local block volume, so SQLite with WAL is safe and history survives
restarts. Add keys with `fly secrets set OPENAI_API_KEY=... PAPEER_ACCESS_CODE=...`
and keep one machine (`fly scale count 1`). Machines can auto-stop when idle.
**Render** also works: a Docker web service plus a persistent disk at `/data`.
Persistent disks need a paid instance type. Check both providers' current
pricing before choosing.
