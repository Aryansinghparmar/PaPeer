"""Session CRUD + history endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api import session_store
from api.deps import get_current_user, get_graph

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


class RenameRequest(BaseModel):
    first_message: str


@router.post("")
def create(user: str = Depends(get_current_user)) -> dict:
    return session_store.create_session()


@router.get("")
def list_all(user: str = Depends(get_current_user)) -> list[dict]:
    return session_store.list_sessions()


@router.get("/{sid}/messages")
async def messages(sid: str, graph=Depends(get_graph), user: str = Depends(get_current_user)):
    return await session_store.get_history(graph, sid)


@router.patch("/{sid}")
def rename(sid: str, body: RenameRequest, user: str = Depends(get_current_user)) -> dict:
    name = session_store.maybe_rename_session(sid, body.first_message)
    return {"id": sid, "name": name}


@router.delete("/{sid}")
def delete(sid: str, user: str = Depends(get_current_user)) -> dict:
    return {"deleted": session_store.delete_session(sid)}
