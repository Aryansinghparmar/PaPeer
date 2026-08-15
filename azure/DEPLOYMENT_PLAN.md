# Azure-first deployment plan

This file is the policy/decision record. The step-by-step runbook lives in
**[`DEPLOYMENT.md`](DEPLOYMENT.md)** (Azure Container Apps: CLI + Portal, secrets,
cost, persistence, shutdown).

## Current status (2026-08-15)

- [x] Azure for Students subscription is active (user confirmed).
- [x] Service chosen: **Azure Container Apps** (scale-to-zero to protect the
  $100 student credit). Cost analysis in `DEPLOYMENT.md` §1.
- [x] The Docker image builds locally with pre-cached retrieval models
  (`Dockerfile`); pushed to Docker Hub as `<DOCKERHUB_USER>/papeer:v1`.
- [ ] A budget alert and scale-to-zero are configured (do at deploy time —
  `DEPLOYMENT.md` §6).
- [ ] The application is deployed and smoke-tested on Azure.

The subscription is active, but **no billable Azure resource has been created by
the assistant** — deployment is executed by the user via `DEPLOYMENT.md`, or on an
explicit go-ahead. Student free quotas and current prices must still be verified in
the Azure portal at deploy time; they are not a permanent free guarantee.

## Proposed order

1. Build and test the Docker image locally.
2. Publish one versioned image to Docker Hub.
3. Check the current Azure for Students allowance and region availability.
4. Prefer the smallest suitable Azure service. A small Linux VM is simple but
   needs shutdown control. Azure Container Apps is easier to operate but can
   have different billing and persistence behaviour. Select only after the
   current price and student credit are visible.
5. Pass environment variables at runtime. Never copy `.env` into the image.
6. Use Qdrant Cloud as the remote vector store and keep the local embedding
   cache and SQLite checkpoint path on a persistent volume only if the chosen
   service supports it within the student allowance.
7. Add a shutdown rule and verify it before sharing the public URL.
8. Test upload, paper retrieval, direct answer, claim verification, logs,
   LangSmith traces, and health checks.

## Deployment stop rules

Stop before creation if the portal does not show enough student credit, if the
resource has an unclear charge, or if persistence requires a paid service.
Use AWS only after a separate current check proves that the selected setup has
no expected charge.
