"""Session metadata store + history reconstruction.

Metadata (id, name, owner, created_at) persists in a JSON file
(``SESSIONS_FILE``); conversation state lives in the LangGraph checkpointer and
is reconstructed for display via ``backend.history.clean_history_messages``.

Concurrency: FastAPI runs sync handlers in a thread pool, so every
read-modify-write holds one ``threading.Lock`` and every write is atomic (temp
file + ``os.replace``), so a reader never sees a half-written file. This is
safe within one process only; the app runs a single worker. A multi-worker
deployment would move this to Postgres along with the checkpoints.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable

from langchain_core.callbacks import get_usage_metadata_callback

from backend.config import OPENAI_RENAME_MODEL
from backend.history import clean_history_messages
from backend.logging_config import get_logger
from backend.telemetry import monotonic_seconds, record_run

logger = get_logger(__name__)

# Session ids are str(uuid.uuid4()); anything else is rejected before it can
# name a Qdrant collection or a checkpoint thread.
SESSION_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
# Sessions written before ownership existed belong to the local dev principal.
LEGACY_OWNER = "local-dev"
DEFAULT_NAME = "New Session"


def is_valid_session_id(sid: str) -> bool:
    return bool(SESSION_ID_RE.match(sid or ""))


def generate_session_name(first_message: str) -> str:
    from backend.llm import get_chat_model

    try:
        response = get_chat_model(OPENAI_RENAME_MODEL).invoke(
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
        name = str(response.content).strip().strip('"').strip()
        return name[:80] or DEFAULT_NAME
    except Exception:
        logger.warning("session_rename_failed", exc_info=True)
        return DEFAULT_NAME


class SessionStore:
    def __init__(
        self,
        path: str | Path,
        namer: Callable[[str], str] = generate_session_name,
    ) -> None:
        self.path = Path(path)
        self._namer = namer
        self._lock = threading.Lock()

    # ── File I/O (call with the lock held) ──────────────────────────────────

    def _load(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            # Writes are atomic, so this means a corrupt file, not a torn read.
            # Keep it aside instead of silently overwriting it on the next save.
            backup = self.path.with_name(self.path.name + ".corrupt")
            logger.error("sessions_file_corrupt moved_to=%s", backup)
            os.replace(self.path, backup)
            return {}
        return data if isinstance(data, dict) else {}

    def _save(self, meta: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(meta, file, indent=2)
                file.flush()
                os.fsync(file.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    # ── Public API ───────────────────────────────────────────────────────────

    def create_session(self, owner: str) -> dict:
        sid = str(uuid.uuid4())
        record = {
            "id": sid,
            "name": DEFAULT_NAME,
            "created_at": datetime.now().isoformat(),
            "is_named": False,
            "owner": owner,
        }
        with self._lock:
            meta = self._load()
            meta[sid] = record
            self._save(meta)
        return record

    def get_session(self, sid: str) -> dict | None:
        with self._lock:
            return self._load().get(sid)

    def get_owned(self, sid: str, owner: str) -> dict | None:
        """The session if ``sid`` is well-formed, exists and belongs to ``owner``."""

        if not is_valid_session_id(sid):
            return None
        record = self.get_session(sid)
        if record is None or record.get("owner", LEGACY_OWNER) != owner:
            return None
        return record

    def list_sessions(self, owner: str) -> list[dict]:
        with self._lock:
            records = list(self._load().values())
        mine = [r for r in records if r.get("owner", LEGACY_OWNER) == owner]
        return sorted(mine, key=lambda s: s.get("created_at", ""), reverse=True)

    def delete_session(self, sid: str) -> bool:
        with self._lock:
            meta = self._load()
            existed = meta.pop(sid, None) is not None
            if existed:
                self._save(meta)
        return existed

    def maybe_rename_session(self, sid: str, first_message: str) -> str | None:
        """Name a session from its first message (no-op once named).

        The LLM call runs outside the lock, then the file is re-read under the
        lock, so a session created or deleted meanwhile is never clobbered by a
        stale copy.
        """

        with self._lock:
            record = self._load().get(sid)
            if record is None or record.get("is_named"):
                return None
        name = self._namer(first_message)
        with self._lock:
            meta = self._load()
            record = meta.get(sid)
            if record is None or record.get("is_named"):
                return None
            record["name"] = name
            record["is_named"] = True
            self._save(meta)
        return name


def rename_and_record(store: SessionStore, sid: str, first_message: str) -> str | None:
    """Rename, and add the rename call's token cost to the ledger / daily budget."""

    started = monotonic_seconds()
    with get_usage_metadata_callback() as usage:
        name = store.maybe_rename_session(sid, first_message)
    if usage.usage_metadata:
        record_run(
            source="api_session_rename",
            session_id=sid,
            state={},
            latency_seconds=monotonic_seconds() - started,
            model_names=[OPENAI_RENAME_MODEL],
            usage_by_model=usage.usage_metadata,
        )
    return name


async def get_history(graph, sid: str) -> list[dict]:
    config = {"configurable": {"thread_id": sid}}
    try:
        state = await graph.aget_state(config)
        if not state or not state.values:
            return []
        return clean_history_messages(state.values.get("messages", []))
    except Exception:
        logger.exception("history_load_failed")
        return []
