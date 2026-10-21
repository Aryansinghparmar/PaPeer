"""Session CRUD + history endpoints, scoped to the calling principal."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from api import session_store
from api.deps import (
    get_graph,
    get_principal,
    get_store,
    owned_session,
    rate_limited,
    within_budget,
)
from api.settings import MAX_MESSAGE_CHARS
from backend.logging_config import get_logger
from backend.vector_store import delete_session_collection

logger = get_logger(__name__)

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


class RenameRequest(BaseModel):
    first_message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


@router.post("", dependencies=[Depends(rate_limited("sessions"))])
def create(
    principal: str = Depends(get_principal),
    store: session_store.SessionStore = Depends(get_store),
) -> dict:
    return store.create_session(principal)


@router.get("")
def list_mine(
    principal: str = Depends(get_principal),
    store: session_store.SessionStore = Depends(get_store),
) -> list[dict]:
    return store.list_sessions(principal)


@router.get("/{sid}/messages", dependencies=[Depends(owned_session)])
async def messages(sid: str, graph=Depends(get_graph)):
    return await session_store.get_history(graph, sid)


@router.patch(
    "/{sid}",
    dependencies=[
        Depends(owned_session),
        Depends(rate_limited("chat")),
        Depends(within_budget),
    ],
)
def rename(
    sid: str,
    body: RenameRequest,
    store: session_store.SessionStore = Depends(get_store),
) -> dict:
    name = session_store.rename_and_record(store, sid, body.first_message)
    return {"id": sid, "name": name}


@router.delete("/{sid}", dependencies=[Depends(owned_session)])
async def delete(
    sid: str,
    graph=Depends(get_graph),
    store: session_store.SessionStore = Depends(get_store),
) -> dict:
    """Remove the metadata, the Qdrant collection and the checkpoint thread."""

    deleted = await asyncio.to_thread(store.delete_session, sid)
    try:
        await asyncio.to_thread(delete_session_collection, sid)
    except Exception:
        logger.exception("delete_collection_failed")
    checkpointer = getattr(graph, "checkpointer", None)
    if checkpointer is not None and hasattr(checkpointer, "adelete_thread"):
        try:
            await checkpointer.adelete_thread(sid)
        except Exception:
            logger.exception("delete_checkpoint_thread_failed")
    return {"deleted": deleted}
