"""Session store: atomic writes, lock-protected updates, rename without clobbering."""

from __future__ import annotations

import json
import os
import threading

import pytest

from api.session_store import SessionStore, is_valid_session_id


def test_session_ids_are_validated():
    store_id = SessionStore("unused.json")  # no file I/O until used
    assert store_id.get_owned("not-a-uuid", "local-dev") is None
    assert is_valid_session_id("0b6f6f3e-2f1d-4c8a-9a51-6c3e2a1f0d9b")
    assert not is_valid_session_id("0B6F6F3E-2F1D-4C8A-9A51-6C3E2A1F0D9B ")
    assert not is_valid_session_id("../../etc/passwd")


def test_atomic_write_keeps_old_file_when_replace_fails(tmp_path, monkeypatch):
    path = tmp_path / "sessions.json"
    store = SessionStore(path)
    first = store.create_session("alice")
    before = path.read_text(encoding="utf-8")

    def broken_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr("api.session_store.os.replace", broken_replace)
    with pytest.raises(OSError):
        store.create_session("alice")

    assert path.read_text(encoding="utf-8") == before
    assert [p.name for p in tmp_path.iterdir()] == ["sessions.json"]  # no temp left behind
    monkeypatch.undo()
    assert [s["id"] for s in store.list_sessions("alice")] == [first["id"]]


def test_legacy_sessions_without_owner_belong_to_local_dev(tmp_path):
    path = tmp_path / "sessions.json"
    path.write_text(
        json.dumps({"abc": {"id": "abc", "name": "Old", "created_at": "2026-01-01", "is_named": True}}),
        encoding="utf-8",
    )
    store = SessionStore(path)
    assert [s["id"] for s in store.list_sessions("local-dev")] == ["abc"]
    assert store.list_sessions("someone-else") == []


def test_corrupt_file_is_moved_aside_not_silently_overwritten(tmp_path):
    path = tmp_path / "sessions.json"
    path.write_text("{not json", encoding="utf-8")
    store = SessionStore(path)
    assert store.list_sessions("local-dev") == []
    assert (tmp_path / "sessions.json.corrupt").read_text(encoding="utf-8") == "{not json"


def test_rename_does_not_clobber_sessions_created_during_the_llm_call(tmp_path):
    entered, release = threading.Event(), threading.Event()

    def slow_namer(message):
        entered.set()
        assert release.wait(5)
        return "Transformer questions"

    store = SessionStore(tmp_path / "sessions.json", namer=slow_namer)
    first = store.create_session("alice")
    renamer = threading.Thread(target=store.maybe_rename_session, args=(first["id"], "hi"))
    renamer.start()
    assert entered.wait(5)
    second = store.create_session("alice")  # happens while the rename LLM call is in flight
    release.set()
    renamer.join(5)

    sessions = {s["id"]: s for s in store.list_sessions("alice")}
    assert set(sessions) == {first["id"], second["id"]}
    assert sessions[first["id"]]["name"] == "Transformer questions"
    assert sessions[first["id"]]["is_named"] is True
    assert sessions[second["id"]]["is_named"] is False


def test_rename_of_session_deleted_meanwhile_is_dropped(tmp_path):
    store = SessionStore(tmp_path / "sessions.json")
    record = store.create_session("alice")

    def namer(message):
        store.delete_session(record["id"])
        return "Too late"

    store._namer = namer
    assert store.maybe_rename_session(record["id"], "hi") is None
    assert store.list_sessions("alice") == []


def test_concurrent_creates_and_renames_lose_nothing(tmp_path):
    store = SessionStore(tmp_path / "sessions.json", namer=lambda m: f"name {m}")
    seeded = [store.create_session("alice")["id"] for _ in range(10)]
    created: list[str] = []
    errors: list[BaseException] = []

    def creator():
        try:
            for _ in range(10):
                created.append(store.create_session("alice")["id"])
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    def renamer(sid):
        try:
            store.maybe_rename_session(sid, sid[:4])
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=creator) for _ in range(4)]
    threads += [threading.Thread(target=renamer, args=(sid,)) for sid in seeded]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    assert errors == []
    sessions = {s["id"]: s for s in store.list_sessions("alice")}
    assert set(sessions) == set(seeded) | set(created)
    assert len(sessions) == 50
    assert all(sessions[sid]["is_named"] for sid in seeded)
    assert not any(name.endswith(".tmp") for name in os.listdir(tmp_path))
