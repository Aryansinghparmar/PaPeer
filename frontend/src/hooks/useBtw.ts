import { useCallback, useRef, useState } from "react";
import { errorMessage, streamBtw, streamErrorMessage } from "../api";

export interface BtwExchange {
  query: string;
  answer: string;
  stopped?: boolean;
  /** Empty `/btw`: show usage help instead of calling the API. */
  usage?: boolean;
}

/** The `/btw` off-topic side channel: streamed, never written to session history. */
export function useBtw() {
  const [exchange, setExchange] = useState<BtwExchange | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const userStoppedRef = useRef(false);

  const reset = useCallback(() => {
    userStoppedRef.current = false;
    abortRef.current?.abort();
    abortRef.current = null;
    setStreaming(false);
    setExchange(null);
    setError(null);
  }, []);

  const stop = useCallback(() => {
    if (!abortRef.current) return;
    userStoppedRef.current = true;
    abortRef.current.abort();
  }, []);

  const ask = useCallback(async (query: string) => {
    abortRef.current?.abort();
    setError(null);
    if (!query) {
      setExchange({ query: "", answer: "", usage: true });
      return;
    }
    const controller = new AbortController();
    abortRef.current = controller;
    userStoppedRef.current = false;
    const isCurrent = () => abortRef.current === controller;
    setStreaming(true);
    setExchange({ query, answer: "" });
    let buffer = "";
    try {
      const outcome = await streamBtw(query, {
        signal: controller.signal,
        onEvent: (event) => {
          if (!isCurrent()) return;
          if (event.type === "token") {
            buffer += event.content;
            setExchange({ query, answer: buffer });
          } else if (event.type === "done") {
            setExchange({ query, answer: event.answer ?? buffer });
            setStreaming(false);
          } else if (event.type === "error") {
            setError(streamErrorMessage(event.detail, event.code));
            setStreaming(false);
          }
        },
      });
      if (outcome === "aborted" && userStoppedRef.current && isCurrent()) {
        setExchange({ query, answer: buffer, stopped: true });
      }
    } catch (err) {
      if (isCurrent()) setError(errorMessage(err));
    } finally {
      if (isCurrent()) {
        abortRef.current = null;
        setStreaming(false);
      }
      userStoppedRef.current = false;
    }
  }, []);

  return { exchange, streaming, error, ask, stop, reset };
}
