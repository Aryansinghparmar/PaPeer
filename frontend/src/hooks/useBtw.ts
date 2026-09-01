import { useCallback, useState } from "react";
import { streamBtw } from "../api";

export interface BtwExchange {
  query: string;
  answer: string;
}

/** The `/btw` off-topic side channel: streamed, never written to session history. */
export function useBtw() {
  const [exchange, setExchange] = useState<BtwExchange | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const reset = useCallback(() => {
    setExchange(null);
    setError(null);
  }, []);

  const ask = useCallback(async (query: string) => {
    setStreaming(true);
    setError(null);
    setExchange({ query, answer: "" });
    let buffer = "";
    try {
      await streamBtw(query, {
        onEvent: (event) => {
          if (event.type === "token") {
            buffer += event.content;
            setExchange({ query, answer: buffer });
          } else if (event.type === "error") {
            setError(event.detail);
          }
        },
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setStreaming(false);
    }
  }, []);

  return { exchange, streaming, error, ask, reset };
}
