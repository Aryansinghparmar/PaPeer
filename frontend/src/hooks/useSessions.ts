import { useCallback, useEffect, useRef, useState } from "react";
import { createSession, deleteSession, listSessions } from "../api";
import type { SessionMeta } from "../types";

/** Session list + active-session lifecycle, mirroring app.py's bootstrap logic:
 * load sessions, pick the most recently created as active, or create one if none
 * exist. */
export function useSessions() {
  const [sessions, setSessions] = useState<SessionMeta[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const bootstrapped = useRef(false); // guards against StrictMode's double-invoked effect

  const refresh = useCallback(async () => {
    const list = await listSessions();
    setSessions(list);
    return list;
  }, []);

  useEffect(() => {
    if (bootstrapped.current) return;
    bootstrapped.current = true;
    (async () => {
      const list = await refresh();
      if (list.length > 0) {
        setActiveId(list[0].id); // already sorted newest-first by the API
      } else {
        const created = await createSession();
        setSessions([created]);
        setActiveId(created.id);
      }
      setLoading(false);
    })();
  }, [refresh]);

  const newChat = useCallback(async () => {
    const created = await createSession();
    setSessions((prev) => [created, ...prev]);
    setActiveId(created.id);
    return created.id;
  }, []);

  const remove = useCallback(
    async (sid: string) => {
      await deleteSession(sid);
      const list = await refresh();
      if (activeId === sid) {
        setActiveId(list[0]?.id ?? null);
      }
    },
    [activeId, refresh],
  );

  return { sessions, activeId, setActiveId, loading, refresh, newChat, remove };
}
