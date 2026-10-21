import { useCallback, useEffect, useRef, useState } from "react";
import { createSession, deleteSession, errorMessage, listSessions } from "../api";
import type { SessionMeta } from "../types";

/** Session list + active-session lifecycle, mirroring app.py's bootstrap logic:
 * load sessions, pick the most recently created as active, or create one if none
 * exist. */
export function useSessions() {
  const [sessions, setSessions] = useState<SessionMeta[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  /** Bootstrap failed — the app can't start until `retry` succeeds. */
  const [bootError, setBootError] = useState<string | null>(null);
  /** A non-fatal failure of new chat / delete / refresh. */
  const [actionError, setActionError] = useState<string | null>(null);
  const bootstrapped = useRef(false); // guards against StrictMode's double-invoked effect

  const refresh = useCallback(async () => {
    const list = await listSessions();
    setSessions(list);
    return list;
  }, []);

  const bootstrap = useCallback(async () => {
    setLoading(true);
    setBootError(null);
    try {
      const list = await refresh();
      if (list.length > 0) {
        setActiveId(list[0].id); // already sorted newest-first by the API
      } else {
        const created = await createSession();
        setSessions([created]);
        setActiveId(created.id);
      }
    } catch (err) {
      setBootError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  }, [refresh]);

  useEffect(() => {
    if (bootstrapped.current) return;
    bootstrapped.current = true;
    void bootstrap();
  }, [bootstrap]);

  /** Refresh the list without throwing (e.g. to pick up an auto-generated name). */
  const refreshQuietly = useCallback(async () => {
    try {
      await refresh();
    } catch (err) {
      setActionError(`Couldn't refresh the chat list. ${errorMessage(err)}`);
    }
  }, [refresh]);

  const newChat = useCallback(async () => {
    setActionError(null);
    try {
      const created = await createSession();
      setSessions((prev) => [created, ...prev]);
      setActiveId(created.id);
      return created.id;
    } catch (err) {
      setActionError(`Couldn't start a new chat. ${errorMessage(err)}`);
      return null;
    }
  }, []);

  const remove = useCallback(
    async (sid: string) => {
      setActionError(null);
      try {
        await deleteSession(sid);
        const list = await refresh();
        if (activeId === sid) {
          setActiveId(list[0]?.id ?? null);
        }
      } catch (err) {
        setActionError(`Couldn't delete the chat. ${errorMessage(err)}`);
      }
    },
    [activeId, refresh],
  );

  const dismissActionError = useCallback(() => setActionError(null), []);

  return {
    sessions,
    activeId,
    setActiveId,
    loading,
    bootError,
    retry: bootstrap,
    actionError,
    dismissActionError,
    refresh: refreshQuietly,
    newChat,
    remove,
  };
}
