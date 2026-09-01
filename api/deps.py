"""FastAPI dependencies: graph access and identity.

`get_current_user` reads the identity header injected by Azure Easy Auth / Static
Web Apps when the app runs behind them. Locally (no gateway) it falls back to a
dev user, so development needs no auth setup. This is an auth *hook*, not an
in-app auth system — the platform enforces the login.
"""

from __future__ import annotations

from fastapi import Request


def get_graph(request: Request):
    """The single compiled async graph, built once in the app lifespan."""
    return request.app.state.graph


def get_current_user(request: Request) -> str:
    return request.headers.get("X-MS-CLIENT-PRINCIPAL-NAME", "local-dev")
