"""FastAPI dependencies: auth, session ownership, rate limits, daily budget.

Auth modes (``AUTH_MODE``):

- ``none`` (default, local dev): every caller is ``local-dev``.
- ``api_key``: requires ``Authorization: Bearer <PAPEER_ACCESS_CODE>``,
  compared in constant time. Everyone holding the code is one principal
  (``access-code``) and therefore shares sessions; it is a demo gate, not
  per-user accounts.
- ``easyauth``: trusts the base64 ``X-MS-CLIENT-PRINCIPAL`` header that Azure
  Easy Auth injects after sign-in. This is only safe when every request reaches
  the app through Easy Auth (Container Apps auth set to "require
  authentication"): the platform strips client-supplied ``X-MS-*`` headers, but
  if the container is reachable any other way the header can be forged.

Errors use one JSON shape, ``{"detail": str, "code": str}``, so the frontend
can branch on ``code``.
"""

from __future__ import annotations

import base64
import binascii
import hmac
import json
import math

from fastapi import Depends, Request
from fastapi.responses import JSONResponse

from api.session_store import SessionStore
from api.settings import Settings
from backend.telemetry import spent_today_usd

LOCAL_PRINCIPAL = "local-dev"
ACCESS_CODE_PRINCIPAL = "access-code"
_EASYAUTH_ID_CLAIMS = (
    "http://schemas.microsoft.com/identity/claims/objectidentifier",
    "oid",
    "sub",
    "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/nameidentifier",
)


class ApiError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        detail: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.code = code
        self.detail = detail
        self.headers = headers


async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(
        {"detail": exc.detail, "code": exc.code},
        status_code=exc.status_code,
        headers=exc.headers,
    )


def unauthorized(detail: str) -> ApiError:
    return ApiError(401, "unauthorized", detail, headers={"WWW-Authenticate": "Bearer"})


def not_found() -> ApiError:
    return ApiError(404, "not_found", "Session not found")


# ── App state accessors ──────────────────────────────────────────────────────

def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_graph(request: Request):
    """The single compiled async graph, built once in the app lifespan."""
    return request.app.state.graph


def get_store(request: Request) -> SessionStore:
    return request.app.state.session_store


# ── Auth ─────────────────────────────────────────────────────────────────────

def _easyauth_principal(request: Request) -> str:
    raw = request.headers.get("X-MS-CLIENT-PRINCIPAL")
    if not raw:
        raise unauthorized("Sign-in required")
    try:
        data = json.loads(base64.b64decode(raw + "=" * (-len(raw) % 4)).decode("utf-8"))
    except (binascii.Error, UnicodeDecodeError, ValueError):
        raise unauthorized("Invalid identity header")
    if not isinstance(data, dict):
        raise unauthorized("Invalid identity header")
    # Static Web Apps shape: {"userId", "userDetails", ...}.
    principal = data.get("userId") or data.get("userDetails")
    if not principal:
        # App Service / Container Apps shape: {"auth_typ", "claims": [{"typ","val"}]}.
        claims = {
            c.get("typ"): c.get("val")
            for c in data.get("claims") or []
            if isinstance(c, dict)
        }
        principal = next((claims[t] for t in _EASYAUTH_ID_CLAIMS if claims.get(t)), None)
    if not principal or not isinstance(principal, str):
        raise unauthorized("Identity header has no user id")
    return principal


def get_principal(request: Request, settings: Settings = Depends(get_settings)) -> str:
    if settings.auth_mode == "none":
        return LOCAL_PRINCIPAL
    if settings.auth_mode == "api_key":
        header = request.headers.get("Authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise unauthorized("Access code required")
        if not hmac.compare_digest(
            token.strip().encode("utf-8"), settings.access_code.encode("utf-8")
        ):
            raise unauthorized("Invalid access code")
        return ACCESS_CODE_PRINCIPAL
    return _easyauth_principal(request)


# ── Ownership ────────────────────────────────────────────────────────────────

def owned_session(
    sid: str,
    principal: str = Depends(get_principal),
    store: SessionStore = Depends(get_store),
) -> dict:
    """404 (never 403) for malformed, unknown, or someone else's session ids."""

    record = store.get_owned(sid, principal)
    if record is None:
        raise not_found()
    return record


# ── Abuse controls ───────────────────────────────────────────────────────────

def _client_key(request: Request, settings: Settings, principal: str) -> str:
    if settings.auth_mode == "none":
        host = request.client.host if request.client else "unknown"
        return f"{principal}|{host}"
    return principal


def rate_limited(bucket: str):
    """Dependency factory: sliding-window limit for one route group."""

    def dependency(
        request: Request,
        principal: str = Depends(get_principal),
        settings: Settings = Depends(get_settings),
    ) -> None:
        limit = getattr(settings, f"rate_limit_{bucket}")
        retry_after = request.app.state.rate_limiter.hit(
            bucket,
            _client_key(request, settings, principal),
            limit,
            settings.rate_limit_window_seconds,
        )
        if retry_after is not None:
            raise ApiError(
                429,
                "rate_limited",
                f"Too many requests. Try again in {math.ceil(retry_after)} s.",
                headers={"Retry-After": str(max(1, math.ceil(retry_after)))},
            )

    return dependency


def within_budget(settings: Settings = Depends(get_settings)) -> None:
    """Refuse cost-bearing work once today's estimated spend hits the cap."""

    if settings.daily_budget_usd > 0 and spent_today_usd() >= settings.daily_budget_usd:
        raise ApiError(
            429,
            "budget_exceeded",
            "The daily usage budget for this demo has been reached. It resets at 00:00 UTC.",
        )
