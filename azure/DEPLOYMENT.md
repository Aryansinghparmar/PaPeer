# Papeer — Azure Container Apps Deployment Guide

A step-by-step runbook to deploy the Papeer Docker image to **Azure Container Apps**
using your **Azure for Students** subscription. It covers the CLI and the Portal,
secret handling, an honest cost breakdown, persistence, and shutdown.

> **Golden rules**
> - Never bake `.env` or API keys into the image. Pass them as Container App secrets.
> - Container Apps bills for what runs. **Scale-to-zero unless you are actively
>   demoing**, and **delete the resource group** when you are done.
> - Verify live prices in the [Azure Pricing Calculator](https://azure.microsoft.com/pricing/calculator/)
>   and your Portal cost view before committing — rates change.

---

## 0. Architecture recap

- **App:** Streamlit (port 8501), served from the Docker image.
- **Vector DB:** Qdrant Cloud — *external and persistent* (not in the container).
- **Retrieval models:** BM25 + cross-encoder reranker run locally in the container
  (baked into the image at build time, ONNX/CPU, no external calls).
- **Local state (ephemeral in the container):** LangGraph SQLite checkpoints
  (`checkpoints.db`), `sessions.json`, and `embedding_cache/`. These reset when a
  revision restarts unless you mount Azure Files (see §8). For a portfolio demo,
  ephemeral is acceptable — Qdrant still holds the uploaded papers.

---

## 1. Cost — read this first

**Free grant (per subscription per month):** 180,000 vCPU-seconds, 360,000
GiB-seconds, and 2,000,000 requests. Beyond that, active usage is billed per second
(~**$0.000024 / vCPU-second** and ~**$0.000003 / GiB-second** in US East as of mid-2026),
plus **$0.40 per million** requests over the free 2M. Scaled-to-zero replicas cost
nothing.

**What that means for Papeer at 1 vCPU / 2 GiB:**

| Mode | Behaviour | Rough monthly cost | Verdict |
|---|---|---|---|
| **min-replicas = 0** (scale-to-zero) | Sleeps when idle; ~30–60 s cold start on first hit; sessions reset on wake | **≈ $0–a few $** (often within the free grant for light use) | ✅ **Recommended** — protects your $100 credit |
| **min-replicas = 1** (always-on) | No cold start; always running | **≈ $30–75/mo** (1 replica × ~2.6 M vCPU-s minus the free grant) | ⚠️ Burns the $100 credit in ~1.5–3 months. Only for a short, active demo window — then delete |

**Cost controls you should set (see §7):** a Cost Management **budget alert**, prefer
**scale-to-zero**, and **delete the resource group** after your interviews. There is
also a small pay-as-you-go **Log Analytics** cost for the Container Apps environment's
logs — minor, but real.

---

## 2. Prerequisites

- Active **Azure for Students** subscription (done ✅).
- **Azure CLI** installed and signed in: `az login`.
- The Papeer image pushed to Docker Hub as `<DOCKERHUB_USER>/papeer:v1` (see §3 —
  built this session).
- Your credentials ready to paste as secrets: `OPENAI_API_KEY`, `TAVILY_API_KEY`,
  `QDRANT_URL`, `QDRANT_API_KEY`, `LANGSMITH_API_KEY`.

---

## 3. Build & push the image (reference — done this session)

```bash
# From the project root
docker build -t papeer:latest .

# Tag for your Docker Hub account and push (you must be logged in)
docker login
docker tag papeer:latest <DOCKERHUB_USER>/papeer:v1
docker push <DOCKERHUB_USER>/papeer:v1
```

The image pre-caches the BM25 + reranker models, so the container needs no model
downloads at request time.

---

## 4. Deploy with the Azure CLI (recommended path)

Set variables (pick a region that offers Container Apps near you):

```bash
RG=papeer-rg
LOC=centralindia            # or eastus, etc.
ENV=papeer-env
APP=papeer
IMAGE=docker.io/<DOCKERHUB_USER>/papeer:v1
```

Register providers and the extension (one-time):

```bash
az login
az extension add --name containerapp --upgrade
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights
```

Create the resource group and the Container Apps environment:

```bash
az group create --name $RG --location $LOC
az containerapp env create --name $ENV --resource-group $RG --location $LOC
```

Create the app with **secrets**, env vars, and external ingress on port 8501.
**Scale-to-zero** (`--min-replicas 0`) is the cost-safe default:

```bash
az containerapp create \
  --name $APP --resource-group $RG --environment $ENV \
  --image $IMAGE \
  --target-port 8501 --ingress external --transport auto \
  --min-replicas 0 --max-replicas 1 \
  --cpu 1.0 --memory 2.0Gi \
  --secrets \
      openai-key=<OPENAI_API_KEY> \
      tavily-key=<TAVILY_API_KEY> \
      qdrant-url=<QDRANT_URL> \
      qdrant-key=<QDRANT_API_KEY> \
      langsmith-key=<LANGSMITH_API_KEY> \
  --env-vars \
      OPENAI_API_KEY=secretref:openai-key \
      TAVILY_API_KEY=secretref:tavily-key \
      QDRANT_URL=secretref:qdrant-url \
      QDRANT_API_KEY=secretref:qdrant-key \
      LANGSMITH_API_KEY=secretref:langsmith-key \
      LANGSMITH_TRACING=true \
      LANGSMITH_PROJECT=papeer \
      RETRIEVAL_MODE=hybrid \
      RERANK_ENABLED=true
```

> **Security note:** the keys appear in your shell history with `--secrets`. After
> deploying, clear history (`history -c` / close the terminal), or set secrets in a
> separate `az containerapp secret set` call and reference them. The values are stored
> encrypted as Container App secrets and are never in the image.

Get the public URL:

```bash
az containerapp show --name $APP --resource-group $RG \
  --query properties.configuration.ingress.fqdn -o tsv
```

Open `https://<that-fqdn>` in a browser.

**For an always-on demo window** (no cold start), update the app to `--min-replicas 1`
temporarily, then set it back to 0 when finished:

```bash
az containerapp update --name $APP --resource-group $RG --min-replicas 1
# ...later...
az containerapp update --name $APP --resource-group $RG --min-replicas 0
```

---

## 5. Deploy with the Azure Portal (click path)

1. **Create a resource → Container App.**
2. **Basics:** subscription, new resource group `papeer-rg`, name `papeer`, region.
3. **Container:** uncheck "Use quickstart image". Image source = **Docker Hub**,
   image = `<DOCKERHUB_USER>/papeer:v1`. CPU/memory = **1.0 / 2.0Gi**.
4. **Ingress:** enable, **external**, target port **8501**.
5. **Scaling:** min replicas **0**, max **1**.
6. After creation, open the app → **Settings → Secrets**, add `openai-key`,
   `tavily-key`, `qdrant-url`, `qdrant-key`, `langsmith-key`. Then
   **Containers → Environment variables**, add the vars from §4 (using "Reference a
   secret" for the five keys, plain values for `LANGSMITH_TRACING=true`,
   `LANGSMITH_PROJECT=papeer`, `RETRIEVAL_MODE=hybrid`, `RERANK_ENABLED=true`), and
   apply as a new revision.
7. Copy the **Application URL** from the Overview page.

---

## 6. Authentication (Easy Auth) — protect your public demo

A public demo URL runs on **your** OpenAI/Tavily keys, so leaving it open invites cost
abuse. Azure Container Apps has **built-in authentication ("Easy Auth")** — a
platform-managed sidecar that gates requests *before* they reach the app, so you write
**zero auth code**. Provider: **Microsoft Entra ID** (social providers optional). It is
**free on the student plan** (Entra ID free tier + no separate Easy Auth charge; the
sidecar's compute is ~$0 under scale-to-zero).

### Configure it multi-tenant (so reviewers can sign in)
For a portfolio demo you want *humans* (interviewers) to sign in with their own account
while *anonymous* traffic is blocked. Make the app registration **multi-tenant + personal
Microsoft accounts** (`AzureADandPersonalMicrosoftAccount`). A **single-tenant**
registration only admits accounts in *your* directory — reviewers would be locked out.

### Portal (recommended — it also creates the app registration + secret for you)
1. Container App → **Settings → Authentication → Add identity provider**.
2. Provider **Microsoft**; let it **create a new app registration**.
3. **Supported account types:** "Any Microsoft Entra directory and personal Microsoft
   accounts" (multi-tenant).
4. **Restrict access:** *Require authentication*. **Unauthenticated requests:**
   *HTTP 302 redirect to login* (the browser flow Streamlit needs).
5. Save. Every visitor now must sign in; anonymous bots are turned away.

### CLI (reference)
The Portal path is simplest because it provisions the registration *and* the client
secret. If you prefer CLI, create the registration multi-tenant, then enable auth (you
must also store a client secret, referenced by `--client-secret-setting-name`):

```bash
az ad app create --display-name "papeer-auth" \
  --sign-in-audience AzureADandPersonalMicrosoftAccount
CLIENT_ID=<appId-from-above>

az containerapp auth microsoft update \
  --name $APP --resource-group $RG \
  --client-id $CLIENT_ID \
  --issuer https://login.microsoftonline.com/common/v2.0
az containerapp auth update \
  --name $APP --resource-group $RG \
  --action RedirectToLoginPage --redirect-provider azureactivedirectory
```

### What your app receives
After sign-in, Easy Auth forwards the identity to the container as headers —
`X-MS-CLIENT-PRINCIPAL-NAME` (the user) and `X-MS-CLIENT-PRINCIPAL` (base64 claims). The
current Streamlit app doesn't need them, but the future FastAPI backend reads
`X-MS-CLIENT-PRINCIPAL-NAME` as its auth hook (no token parsing required).

> **After the React + FastAPI split:** auth moves to the frontend host. The plan puts the
> React SPA on **Azure Static Web Apps**, which has its **own built-in auth** (Entra ID +
> social) and links to the FastAPI backend on Container Apps — so the login gate lives at
> the SPA (sign in via `/.auth/login/aad`, identity at `/.auth/me`) and the linked API
> receives the forwarded principal header. At that point you configure auth on the Static
> Web App, not the Container App.

## 7. Cost controls & shutdown

- **Budget alert:** Portal → **Cost Management → Budgets → Add** → e.g. $10/month with
  an alert at 80%. Do this before leaving the app running.
- **Prefer scale-to-zero** (`--min-replicas 0`). No usage charges while asleep.
- **Stop entirely** (keeps config, no compute): `az containerapp update --name $APP
  --resource-group $RG --min-replicas 0 --max-replicas 0` — or simply delete.
- **Delete everything** when done (stops all charges):
  ```bash
  az group delete --name $RG --yes --no-wait
  ```

---

## 8. Persistence (optional)

Qdrant Cloud already persists uploaded papers, so a restart does **not** lose your
document collections. What resets on a revision restart is local chat history
(SQLite checkpoints), `sessions.json`, and the embedding cache.

If you want chat history to survive restarts, mount **Azure Files**:

1. Create a storage account + file share.
2. `az containerapp env storage set` to register the share on the environment.
3. Add a volume + volume mount in the container targeting the paths for
   `CHECKPOINT_DB_PATH` and `EMBEDDING_CACHE_DIR` (set those env vars to the mounted
   path). This adds a small storage cost. For a demo, skipping this is fine.

---

## 9. Configuration reference

| Variable | Purpose | Deploy as |
|---|---|---|
| `OPENAI_API_KEY` | LLM + embeddings | secret |
| `TAVILY_API_KEY` | web search / claim verification | secret |
| `QDRANT_URL`, `QDRANT_API_KEY` | Qdrant Cloud | secret |
| `LANGSMITH_API_KEY` | tracing | secret |
| `LANGSMITH_TRACING` | `true` to trace | env |
| `LANGSMITH_PROJECT` | `papeer` | env |
| `RETRIEVAL_MODE` | `hybrid` (or `dense`) | env |
| `RERANK_ENABLED` | `true` (or `false`) | env |

Other tunables (`RETRIEVAL_CANDIDATE_K`, `RERANK_TOP_N`, `TAVILY_MAX_RESULTS`, ...)
have safe defaults in `backend/config.py` and can be added as env vars if needed.

---

## 10. Update / redeploy

Build and push a new tag, then point the app at it:

```bash
docker build -t papeer:latest .
docker tag papeer:latest <DOCKERHUB_USER>/papeer:v2
docker push <DOCKERHUB_USER>/papeer:v2
az containerapp update --name $APP --resource-group $RG \
  --image docker.io/<DOCKERHUB_USER>/papeer:v2
```

---

## 11. Troubleshooting

- **Slow first load / 502 on first hit (scale-to-zero):** the container is cold-starting
  and pulling the image. Wait ~30–60 s and retry, or use `--min-replicas 1` during demos.
- **Container killed / OOM:** onnxruntime + embeddings need memory. Use **2.0Gi**;
  don't go below 1.0Gi.
- **App loads but chat errors:** a secret/env var is missing or wrong. Check
  **Revisions → console logs** and re-verify the five secrets.
- **Sessions "reset":** expected with scale-to-zero or multi-replica. Keep
  `--max-replicas 1` (Streamlit is single-node stateful) and mount Azure Files (§8)
  if persistence matters.
- **Health:** the app serves `/_stcore/health`; Container Apps' default probe on the
  target port is sufficient.

---

*Sources for pricing figures:*
[Azure Container Apps pricing](https://azure.microsoft.com/en-us/pricing/details/container-apps/) ·
[Billing in Azure Container Apps](https://learn.microsoft.com/en-us/azure/container-apps/billing)
