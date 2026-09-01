"""Session metadata store + history reconstruction, decoupled from Streamlit.

This lifts the session helpers that used to live in `app.py` into a reusable
module the API (and, if desired, Streamlit) can share. Metadata persists in
`sessions.json`; conversation state lives in the LangGraph checkpointer and is
reconstructed for display via `backend.history.clean_history_messages`.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

from langchain_openai import ChatOpenAI

from backend.config import OPENAI_RENAME_MODEL
from backend.history import clean_history_messages
from backend.vector_store import delete_session_collection

SESSIONS_FILE = Path("sessions.json")
_rename_llm = ChatOpenAI(model=OPENAI_RENAME_MODEL, stream_usage=True)


def load_sessions() -> dict:
    try:
        return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_sessions(meta: dict) -> None:
    SESSIONS_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def create_session() -> dict:
    sid = str(uuid.uuid4())
    meta = load_sessions()
    record = {
        "id": sid,
        "name": "New Session",
        "created_at": datetime.now().isoformat(),
        "is_named": False,
    }
    meta[sid] = record
    save_sessions(meta)
    return record


def list_sessions() -> list[dict]:
    return sorted(load_sessions().values(), key=lambda s: s["created_at"], reverse=True)


def delete_session(sid: str) -> bool:
    meta = load_sessions()
    existed = sid in meta
    if existed:
        del meta[sid]
        save_sessions(meta)
    try:
        delete_session_collection(sid)
    except Exception:
        pass
    return existed


def generate_session_name(first_message: str) -> str:
    try:
        response = _rename_llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "Generate a concise 3-5 word title for a research chat session "
                        "based on the user's first message. Return only the title, no "
                        "punctuation at the end, no quotes."
                    ),
                },
                {"role": "user", "content": first_message[:500]},
            ]
        )
        return response.content.strip()
    except Exception:
        return "New Session"


def maybe_rename_session(sid: str, first_message: str) -> str | None:
    meta = load_sessions()
    if sid not in meta or meta[sid].get("is_named"):
        return None
    name = generate_session_name(first_message)
    meta[sid]["name"] = name
    meta[sid]["is_named"] = True
    save_sessions(meta)
    return name


async def get_history(graph, sid: str) -> list[dict]:
    config = {"configurable": {"thread_id": sid}}
    try:
        state = await graph.aget_state(config)
        if not state or not state.values:
            return []
        return clean_history_messages(state.values.get("messages", []))
    except Exception:
        return []
