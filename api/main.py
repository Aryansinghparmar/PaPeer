"""Papeer FastAPI backend.

Wraps the LangGraph RAG workflow in a REST + SSE API and, when a built React
SPA is present (``STATIC_DIR``, default ``frontend/dist``), serves it from the
same origin. The compiled async graph is built once in the app lifespan and
shared across requests; its aiosqlite connection is closed on shutdown.

All mutable state here (session file lock, rate-limit counters, daily spend)
is per process, so the app runs with one uvicorn worker and one replica.

``create_app`` takes overrides so tests can inject settings, a fake graph and
a fake /btw handler without building real clients.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.chat import router as chat_router
from api.deps import ApiError, api_error_handler, get_settings
from api.documents import UploadSizeLimit
from api.documents import router as documents_router
from api.ratelimit import SlidingWindowLimiter
from api.session_store import SessionStore
from api.sessions import router as sessions_router
from api.settings import Settings
from backend.config import CHECKPOINT_DB_PATH
from backend.logging_config import get_logger
from backend.telemetry import load_today_spend

logger = get_logger(__name__)


def _default_btw_handler(query: str):
    from backend.btw_handler import handle_btw

    return handle_btw(query)


async def _close_checkpointer(graph) -> None:
    conn = getattr(getattr(graph, "checkpointer", None), "conn", None)
    if conn is not None and hasattr(conn, "close"):
        await conn.close()


def _mount_spa(app: FastAPI, static_dir: Path) -> None:
    """Serve the built SPA: hashed assets under /assets, real files at the
    root (favicon etc.), and index.html for any other non-API GET path so
    client-side routes survive a reload."""

    index = static_dir / "index.html"
    if not index.is_file():
        logger.info("spa_not_served static_dir=%s (no index.html)", static_dir)
        return
    root = static_dir.resolve()
    if (static_dir / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa(full_path: str):
        if full_path == "api" or full_path.startswith("api/"):
            return JSONResponse({"detail": "Not found", "code": "not_found"}, status_code=404)
        if full_path:
            candidate = (root / full_path).resolve()
            if candidate.is_relative_to(root) and candidate.is_file():
                return FileResponse(candidate)
        return FileResponse(index, headers={"Cache-Control": "no-cache"})


def create_app(
    settings: Settings | None = None,
    *,
    graph=None,
    btw_handler: Callable | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        spent = load_today_spend()
        logger.info(
            "startup auth_mode=%s spent_today_usd=%.4f daily_budget_usd=%.2f",
            settings.auth_mode,
            spent,
            settings.daily_budget_usd,
        )
        owns_graph = app.state.graph is None
        if owns_graph:
            from backend.rag_graph import build_graph_async

            app.state.graph = await build_graph_async(CHECKPOINT_DB_PATH)
        try:
            yield
        finally:
            if owns_graph:
                await _close_checkpointer(app.state.graph)

    app = FastAPI(title="Papeer API", version="1.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.graph = graph
    app.state.btw_handler = btw_handler or _default_btw_handler
    app.state.session_store = SessionStore(settings.sessions_file)
    app.state.rate_limiter = SlidingWindowLimiter()

    app.add_exception_handler(ApiError, api_error_handler)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        error_id = uuid.uuid4().hex[:12]
        logger.error("unhandled_error path=%s error_id=%s", request.url.path, error_id, exc_info=exc)
        return JSONResponse(
            {"detail": f"Internal error (error id {error_id})", "code": "internal"},
            status_code=500,
        )

    app.add_middleware(UploadSizeLimit, settings=settings)
    # The SPA is served same-origin in production, so CORS only matters for the
    # Vite dev server. No cookies are used, so credentials stay off.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )

    app.include_router(sessions_router)
    app.include_router(documents_router)
    app.include_router(chat_router)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/auth/mode")
    def auth_mode(request: Request) -> dict:
        return {"mode": get_settings(request).auth_mode}

    _mount_spa(app, Path(settings.static_dir))
    return app


app = create_app()
