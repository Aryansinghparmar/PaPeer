import { useCallback, useEffect, useRef, useState } from "react";
import { getMessages, streamChat } from "../api";
import type { ChatMessage } from "../types";

/** Per-session chat state: hydrates history once per session (cached, like
 * Streamlit's `st.session_state.chats`), then streams new turns via SSE. */
export function useChat(sid: string | null, onFirstMessageSent?: (sid: string) => void) {
  const cacheRef = useRef<Record<string, ChatMessage[]>>({});
  const abortRef = useRef<AbortController | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [streamText, setStreamText] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    abortRef.current?.abort();
    setStreaming(false);
    setStreamText("");
    setError(null);
    if (!sid) {
      setMessages([]);
      return;
    }
    if (cacheRef.current[sid]) {
      setMessages(cacheRef.current[sid]);
      return;
    }
    let cancelled = false;
    (async () => {
      const history = await getMessages(sid);
      if (!cancelled) {
        cacheRef.current[sid] = history;
        setMessages(history);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [sid]);

  const send = useCallback(
    async (text: string) => {
      if (!sid || streaming) return;
      const isFirst = (cacheRef.current[sid]?.length ?? 0) === 0;
      const userMsg: ChatMessage = { role: "user", content: text };
      const withUser = [...(cacheRef.current[sid] ?? []), userMsg];
      cacheRef.current[sid] = withUser;
      setMessages(withUser);
      setStreaming(true);
      setStreamText("");
      setError(null);

      const controller = new AbortController();
      abortRef.current = controller;
      let buffer = "";
      try {
        await streamChat(sid, text, {
          signal: controller.signal,
          onEvent: (event) => {
            if (event.type === "token") {
              buffer += event.content;
              setStreamText(buffer);
            } else if (event.type === "done") {
              const turn =
                (cacheRef.current[sid]?.filter((m) => m.role === "assistant").length ?? 0) + 1;
              const assistantMsg: ChatMessage = {
                role: "assistant",
                content: event.answer ?? buffer,
                turn,
                graph_state: event.state ?? {},
              };
              const finalList = [...(cacheRef.current[sid] ?? []), assistantMsg];
              cacheRef.current[sid] = finalList;
              setMessages(finalList);
              setStreamText("");
              // Flip immediately on `done` rather than waiting for the SSE fetch to
              // fully close — otherwise the finished bubble and the streaming
              // placeholder can both render for a moment.
              setStreaming(false);
            } else if (event.type === "error") {
              setError(event.detail);
            }
          },
        });
      } catch (err) {
        if ((err as { name?: string })?.name !== "AbortError") {
          setError(err instanceof Error ? err.message : String(err));
        }
      } finally {
        setStreaming(false);
        abortRef.current = null;
        if (isFirst) onFirstMessageSent?.(sid);
      }
    },
    [sid, streaming, onFirstMessageSent],
  );

  return { messages, streaming, streamText, error, send };
}
