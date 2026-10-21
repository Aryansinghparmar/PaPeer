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
  /** Present only on turns streamed in this page view (not on loaded history). */
  observability?: Observability;
  /** The user pressed Stop; `content` holds whatever had streamed so far. */
  stopped?: boolean;
}

export interface Observability {
  latency_seconds: number;
  estimated_cost_usd: number | null;
  input_tokens: number;
  output_tokens: number;
}

/** Server-sent event payloads emitted by /api/sessions/{sid}/chat and /api/btw. */
export type ChatEvent =
  | { type: "status"; stage?: string; label: string }
  | { type: "token"; content: string }
  | {
      type: "done";
      answer?: string;
      state?: Record<string, unknown>;
      observability?: Observability;
    }
  | { type: "error"; detail: string; code?: string /* "timeout" | "loop_limit" | "internal" */ };

export type AuthMode = "none" | "api_key" | "easyauth";

export interface UploadResult {
  added: string[];
  skipped: { name: string; reason: string }[];
  failed: { name: string; error: string }[];
}

export interface UrlLoadResult {
  loaded: string[];
  skipped: { url?: string; name?: string; reason?: string }[];
  failed: { url: string; error: string }[];
}

export interface ArxivResult {
  loaded: string;
}

export interface AuthIdentity {
  clientPrincipal: {
    userId: string;
    userDetails: string;
    identityProvider: string;
  } | null;
}
