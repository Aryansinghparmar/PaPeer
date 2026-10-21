import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, errorMessage, getMessages, streamChat, streamErrorMessage } from "../api";
import type { ChatMessage } from "../types";

/** Per-session chat state: hydrates history once per session (cached, like
 * Streamlit's `st.session_state.chats`), then streams new turns via SSE. */
export function useChat(sid: string | null, onFirstMessageSent?: (sid: string) => void) {
  const cacheRef = useRef<Record<string, ChatMessage[]>>({});
  const abortRef = useRef<AbortController | null>(null);
  const userStoppedRef = useRef(false);
  const sidRef = useRef(sid);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [streamText, setStreamText] = useState("");
  const [statusLabel, setStatusLabel] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyAttempt, setHistoryAttempt] = useState(0);

  useEffect(() => {
    sidRef.current = sid;
    // Switching sessions cancels any in-flight turn (not a user "stop").
    userStoppedRef.current = false;
    abortRef.current?.abort();
    abortRef.current = null;
    setStreaming(false);
    setStreamText("");
    setStatusLabel(null);
    setError(null);
    setHistoryError(null);
    setHistoryLoading(false);
    if (!sid) {
      setMessages([]);
      return;
    }
    if (cacheRef.current[sid]) {
      setMessages(cacheRef.current[sid]);
      return;
    }
    setMessages([]);
    setHistoryLoading(true);
    let cancelled = false;
    (async () => {
      try {
        const history = await getMessages(sid);
        if (!cancelled) {
          cacheRef.current[sid] = history;
          setMessages(history);
        }
      } catch (err) {
        if (!cancelled) setHistoryError(`Couldn't load this chat's history. ${errorMessage(err)}`);
      } finally {
        if (!cancelled) setHistoryLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [sid, historyAttempt]);

  const retryHistory = useCallback(() => setHistoryAttempt((n) => n + 1), []);

  const stop = useCallback(() => {
    if (!abortRef.current) return;
    userStoppedRef.current = true;
    abortRef.current.abort();
  }, []);

  /** Sends one turn. Resolves `false` if the server rejected it before the
   * stream opened (so the caller can restore the draft). */
  const send = useCallback(
    async (text: string): Promise<boolean> => {
      if (!sid || streaming || historyError || !cacheRef.current[sid]) return false;
      const before = cacheRef.current[sid];
      const isFirst = before.length === 0;
      const withUser = [...before, { role: "user", content: text } as ChatMessage];
      cacheRef.current[sid] = withUser;
      setMessages(withUser);
      setStreaming(true);
      setStreamText("");
      setStatusLabel(null);
      setError(null);

      const controller = new AbortController();
      abortRef.current = controller;
      userStoppedRef.current = false;
      const isCurrent = () => abortRef.current === controller && sidRef.current === sid;
      // Fresh per request — never shared with a previous or retried stream.
      let buffer = "";
      let accepted = true;

      const append = (msg: ChatMessage) => {
        const list = [...(cacheRef.current[sid] ?? []), msg];
        cacheRef.current[sid] = list;
        if (sidRef.current === sid) setMessages(list);
      };

      try {
        const outcome = await streamChat(sid, text, {
          signal: controller.signal,
          onEvent: (event) => {
            if (!isCurrent()) return;
            if (event.type === "status") {
              setStatusLabel(event.label);
            } else if (event.type === "token") {
              buffer += event.content;
              setStreamText(buffer);
            } else if (event.type === "done") {
              const turn =
                (cacheRef.current[sid]?.filter((m) => m.role === "assistant" && !m.stopped)
                  .length ?? 0) + 1;
              append({
                role: "assistant",
                content: event.answer ?? buffer,
                turn,
                graph_state: event.state ?? {},
                observability: event.observability,
              });
              setStreamText("");
              // Flip immediately on `done` rather than waiting for the SSE fetch to
              // fully close — otherwise the finished bubble and the streaming
              // placeholder can both render for a moment.
              setStreaming(false);
            } else if (event.type === "error") {
              setError(streamErrorMessage(event.detail, event.code));
              setStreaming(false);
            }
          },
        });
        if (outcome === "aborted" && userStoppedRef.current) {
          append({ role: "assistant", content: buffer, stopped: true });
        }
      } catch (err) {
        if (err instanceof ApiError && cacheRef.current[sid] === withUser) {
          // Rejected before the stream opened: the server never saw this turn.
          accepted = false;
          cacheRef.current[sid] = before;
          if (sidRef.current === sid) setMessages(before);
        }
        if (isCurrent()) setError(errorMessage(err));
      } finally {
        if (abortRef.current === controller) {
          abortRef.current = null;
          setStreaming(false);
          setStreamText("");
          setStatusLabel(null);
        }
        userStoppedRef.current = false;
        if (isFirst && accepted) onFirstMessageSent?.(sid);
      }
      return accepted;
    },
    [sid, streaming, historyError, onFirstMessageSent],
  );

  return {
    messages,
    streaming,
    streamText,
    statusLabel,
    error,
    historyError,
    historyLoading,
    retryHistory,
    send,
    stop,
  };
}
