export interface SessionMeta {
  id: string;
  name: string;
  created_at: string;
  is_named: boolean;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  turn?: number;
  graph_state?: Record<string, unknown>;
}

export interface Observability {
  latency_seconds: number;
  estimated_cost_usd: number | null;
  input_tokens: number;
  output_tokens: number;
}

/** Server-sent event payloads emitted by /api/sessions/{sid}/chat and /api/btw. */
export type ChatEvent =
  | { type: "token"; content: string }
  | {
      type: "done";
      answer?: string;
      state?: Record<string, unknown>;
      observability?: Observability;
    }
  | { type: "error"; detail: string };

export interface AuthIdentity {
  clientPrincipal: {
    userId: string;
    userDetails: string;
    identityProvider: string;
  } | null;
}
