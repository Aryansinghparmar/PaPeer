"""API-layer settings: auth, abuse limits, budget, storage paths, static SPA.

Values come from environment variables (``.env`` is loaded by
``backend.config``). ``Settings.from_env()`` is read once per app instance and
stored on ``app.state.settings`` so tests can build an app with overrides.
Request-body length limits are module constants because pydantic models need
them at class-definition time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from backend.config import env_int, env_str

AUTH_MODES = ("none", "api_key", "easyauth")

# Request body limits (pydantic max_length -> 422).
MAX_MESSAGE_CHARS = env_int("MAX_MESSAGE_CHARS", 4000, 100, 100_000)
MAX_BTW_QUERY_CHARS = env_int("MAX_BTW_QUERY_CHARS", 2000, 100, 100_000)
MAX_ARXIV_QUERY_CHARS = env_int("MAX_ARXIV_QUERY_CHARS", 300, 20, 2000)
MAX_URL_CHARS = 2048
MAX_URLS_PER_REQUEST = env_int("MAX_URLS_PER_REQUEST", 5, 1, 50)


def env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    return max(minimum, min(value, maximum))


def _origins() -> list[str]:
    raw = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000",
    )
    return [o.strip() for o in raw.split(",") if o.strip()]


@dataclass(frozen=True)
class Settings:
    auth_mode: str = "none"
    access_code: str = ""
    cors_origins: list[str] = field(default_factory=list)
    sessions_file: str = "sessions.json"
    static_dir: str = "frontend/dist"

    # Streaming
    heartbeat_seconds: float = 15.0
    request_timeout_seconds: float = 180.0

    # Rate limits: max requests per principal in a sliding window. These are
    # in-memory, so they hold per process only (the app runs one worker).
    rate_limit_window_seconds: float = 60.0
    rate_limit_chat: int = 10
    rate_limit_btw: int = 10
    rate_limit_documents: int = 10
    rate_limit_sessions: int = 30

    # Daily spend ceiling across all requests, from the telemetry ledger.
    # 0 disables the cap.
    daily_budget_usd: float = 1.00

    # Uploads
    max_upload_bytes: int = 20 * 1024 * 1024
    max_files_per_request: int = 5
    max_urls_per_request: int = MAX_URLS_PER_REQUEST

    @classmethod
    def from_env(cls) -> "Settings":
        auth_mode = env_str("AUTH_MODE", "none").lower()
        if auth_mode not in AUTH_MODES:
            raise RuntimeError(f"AUTH_MODE must be one of {AUTH_MODES}, got {auth_mode!r}")
        access_code = os.getenv("PAPEER_ACCESS_CODE", "").strip()
        if auth_mode == "api_key" and len(access_code) < 8:
            # Fail closed: an api_key deployment without a real code would be open.
            raise RuntimeError("AUTH_MODE=api_key requires PAPEER_ACCESS_CODE (8+ characters)")
        return cls(
            auth_mode=auth_mode,
            access_code=access_code,
            cors_origins=_origins(),
            sessions_file=env_str("SESSIONS_FILE", "sessions.json"),
            static_dir=env_str("STATIC_DIR", "frontend/dist"),
            heartbeat_seconds=env_float("HEARTBEAT_SECONDS", 15.0, 0.01, 300.0),
            request_timeout_seconds=env_float("REQUEST_TIMEOUT_SECONDS", 180.0, 1.0, 1800.0),
            rate_limit_window_seconds=env_float("RATE_LIMIT_WINDOW_SECONDS", 60.0, 1.0, 86400.0),
            rate_limit_chat=env_int("RATE_LIMIT_CHAT", 10, 1, 10_000),
            rate_limit_btw=env_int("RATE_LIMIT_BTW", 10, 1, 10_000),
            rate_limit_documents=env_int("RATE_LIMIT_DOCUMENTS", 10, 1, 10_000),
            rate_limit_sessions=env_int("RATE_LIMIT_SESSIONS", 30, 1, 10_000),
            daily_budget_usd=env_float("DAILY_BUDGET_USD", 1.00, 0.0, 10_000.0),
            max_upload_bytes=env_int("MAX_UPLOAD_MB", 20, 1, 200) * 1024 * 1024,
            max_files_per_request=env_int("MAX_FILES_PER_REQUEST", 5, 1, 50),
            max_urls_per_request=MAX_URLS_PER_REQUEST,
        )
