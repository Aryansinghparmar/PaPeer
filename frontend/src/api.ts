import { fetchEventSource } from "@microsoft/fetch-event-source";
import type {
  ArxivResult,
  AuthIdentity,
  AuthMode,
  ChatEvent,
  ChatMessage,
  SessionMeta,
  UploadResult,
  UrlLoadResult,
} from "./types";

// Empty by default: same-origin in production, and the Vite dev proxy forwards
// `/api` to the backend in development (see vite.config.ts).
const BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

// ── Errors ───────────────────────────────────────────────────────────────────

/** A non-2xx API response, carrying the backend's JSON `{detail, code}` body. */
export class ApiError extends Error {
  readonly status: number;
  readonly code: string | null;
  readonly detail: string;

  constructor(status: number, detail: string, code: string | null = null) {
    super(detail || `Request failed (${status})`);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.detail = detail;
  }
}

async function toApiError(res: Response): Promise<ApiError> {
  const text = await res.text().catch(() => "");
  let detail = text || res.statusText;
  let code: string | null = null;
  try {
    const body = JSON.parse(text) as { detail?: unknown; code?: unknown };
    if (typeof body.code === "string") code = body.code;
    if (typeof body.detail === "string") {
      detail = body.detail;
    } else if (Array.isArray(body.detail)) {
      // FastAPI validation errors: [{loc, msg, type}, …]
      detail = body.detail
        .map((d) => (d && typeof d === "object" && "msg" in d ? String(d.msg) : String(d)))
        .join("; ");
    }
  } catch {
    // Not JSON — keep the raw text.
  }
  if (res.status === 401) unauthorizedHandler?.();
  return new ApiError(res.status, detail, code);
}

/** Turns any thrown value into a short message suitable for the UI. */
export function errorMessage(err: unknown): string {
  if (err instanceof ApiError) {
    switch (err.code) {
      case "rate_limited":
        return "You're sending requests too quickly. Please wait a moment and try again.";
      case "budget_exceeded":
        return "The daily usage budget has been reached. Please try again tomorrow.";
      case "unauthorized":
        return "Please enter the access code to continue.";
      case "not_found":
        return err.detail || "Not found. This chat may have been deleted.";
    }
    if (err.status === 401) return "Please enter the access code to continue.";
    if (err.status === 413) return "That upload is too large.";
    if (err.status === 422) return `Input rejected: ${err.detail || "it may be too long."}`;
    if (err.status === 429) return "Too many requests. Please try again shortly.";
    if (err.status >= 500) return `The server hit an error (${err.status}). Please try again.`;
    return err.detail || `Request failed (${err.status}).`;
  }
  if (err instanceof TypeError) {
    return "Couldn't reach the Papeer server. Check your connection and that the API is running.";
  }
  return err instanceof Error ? err.message : String(err);
}

/** Maps an in-stream `error` event to a friendly message. */
export function streamErrorMessage(detail: string, code?: string): string {
  switch (code) {
    case "timeout":
      return "The answer took too long and was stopped. Try a narrower question.";
    case "loop_limit":
      return "Papeer couldn't settle on an answer (step limit reached). Try rephrasing the question.";
    default:
      return detail || "Something went wrong while generating the answer.";
  }
}

// ── Access code (api_key auth mode) ──────────────────────────────────────────
// Held in memory and mirrored to sessionStorage when it is available, so a
// blocked storage still works for the lifetime of the page.

const ACCESS_CODE_KEY = "papeer-access-code";

function readStoredCode(): string | null {
  try {
    return window.sessionStorage.getItem(ACCESS_CODE_KEY);
  } catch {
    return null;
  }
}

let accessCode: string | null = readStoredCode();
let unauthorizedHandler: (() => void) | null = null;

export const getAccessCode = () => accessCode;

export function setAccessCode(code: string | null) {
  accessCode = code;
  try {
    if (code) window.sessionStorage.setItem(ACCESS_CODE_KEY, code);
    else window.sessionStorage.removeItem(ACCESS_CODE_KEY);
  } catch {
    // Storage blocked — the in-memory copy is enough for this session.
  }
}

/** Registers the callback invoked whenever any API call gets a 401. */
export function setUnauthorizedHandler(handler: (() => void) | null) {
  unauthorizedHandler = handler;
}

function authHeaders(): Record<string, string> {
  return accessCode ? { Authorization: `Bearer ${accessCode}` } : {};
}

// ── JSON helper ──────────────────────────────────────────────────────────────

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(),
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) throw await toApiError(res);
  return res.json() as Promise<T>;
}

// ── Auth mode ────────────────────────────────────────────────────────────────

export async function getAuthMode(): Promise<AuthMode> {
  const res = await fetch(`${BASE_URL}/api/auth/mode`);
  // Older backends without the endpoint have no auth gate.
  if (res.status === 404) return "none";
  if (!res.ok) throw await toApiError(res);
  const data = (await res.json()) as { mode?: AuthMode };
  return data.mode ?? "none";
}

// ── Sessions ─────────────────────────────────────────────────────────────────

export const createSession = () => json<SessionMeta>("/api/sessions", { method: "POST" });

export const listSessions = () => json<SessionMeta[]>("/api/sessions");

export const getMessages = (sid: string) =>
  json<ChatMessage[]>(`/api/sessions/${sid}/messages`);

export const renameSession = (sid: string, firstMessage: string) =>
  json<{ id: string; name: string | null }>(`/api/sessions/${sid}`, {
    method: "PATCH",
    body: JSON.stringify({ first_message: firstMessage }),
  });

export const deleteSession = (sid: string) =>
  json<{ deleted: boolean }>(`/api/sessions/${sid}`, { method: "DELETE" });

// ── Documents ────────────────────────────────────────────────────────────────

export const listDocuments = (sid: string) =>
  json<{ documents: string[] }>(`/api/sessions/${sid}/documents`);

export async function uploadDocuments(sid: string, files: File[]): Promise<UploadResult> {
  const form = new FormData();
  for (const f of files) form.append("files", f);
  const res = await fetch(`${BASE_URL}/api/sessions/${sid}/documents`, {
    method: "POST",
    headers: authHeaders(), // no Content-Type: the browser sets the multipart boundary
    body: form,
  });
  if (!res.ok) throw await toApiError(res);
  const data = (await res.json()) as Partial<UploadResult>;
  return { added: data.added ?? [], skipped: data.skipped ?? [], failed: data.failed ?? [] };
}

export async function loadUrls(sid: string, urls: string[]): Promise<UrlLoadResult> {
  const data = await json<Partial<UrlLoadResult>>(`/api/sessions/${sid}/documents/url`, {
    method: "POST",
    body: JSON.stringify({ urls }),
  });
  return { loaded: data.loaded ?? [], skipped: data.skipped ?? [], failed: data.failed ?? [] };
}

export const loadArxiv = (sid: string, query: string) =>
  json<ArxivResult>(`/api/sessions/${sid}/documents/arxiv`, {
    method: "POST",
    body: JSON.stringify({ query }),
  });

// ── Streaming chat (SSE over POST) ──────────────────────────────────────────

interface StreamOptions {
  signal?: AbortSignal;
  onEvent: (event: ChatEvent) => void;
}

/** How a stream ended: with a terminal `done`/`error` event, or aborted by us. */
export type StreamOutcome = "completed" | "aborted";

/**
 * POSTs and consumes an SSE response. Exactly one request is ever sent: the
 * library's visibility-based reconnect is disabled (`openWhenHidden`) and every
 * error is rethrown from `onerror`, which stops its automatic retry. A chat turn
 * must never be re-POSTed, since that re-runs the whole graph.
 *
 * Rejects with `ApiError` if the response is non-2xx (before any event), and
 * with a plain `Error` if the stream closes without a `done` or `error` event.
 */
async function streamPost(path: string, body: unknown, opts: StreamOptions): Promise<StreamOutcome> {
  let terminal = false;
  await fetchEventSource(`${BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(body),
    signal: opts.signal,
    openWhenHidden: true,
    async onopen(res) {
      if (!res.ok) throw await toApiError(res);
      const type = res.headers.get("content-type") ?? "";
      if (!type.startsWith("text/event-stream")) {
        throw new ApiError(res.status, `Unexpected response type: ${type || "unknown"}`);
      }
    },
    onmessage(msg) {
      // Heartbeat comments (`: ping`) never reach here; ignore empty data too.
      if (!msg.data) return;
      const event = JSON.parse(msg.data) as ChatEvent;
      if (event.type === "done" || event.type === "error") terminal = true;
      opts.onEvent(event);
    },
    onerror(err) {
      throw err; // never retry — a single request/response, not a live feed
    },
  });
  // On abort the library resolves instead of rejecting.
  if (opts.signal?.aborted) return "aborted";
  if (!terminal) throw new Error("The response ended unexpectedly. Please try again.");
  return "completed";
}

export const streamChat = (sid: string, message: string, opts: StreamOptions) =>
  streamPost(`/api/sessions/${sid}/chat`, { message }, opts);

export const streamBtw = (query: string, opts: StreamOptions) =>
  streamPost(`/api/btw`, { query }, opts);

// ── Auth identity (Azure Static Web Apps / Easy Auth) ───────────────────────
// Present only when deployed behind the platform auth; resolves to no identity
// in local dev.

export async function getAuthIdentity(): Promise<AuthIdentity> {
  try {
    const res = await fetch("/.auth/me");
    if (!res.ok) return { clientPrincipal: null };
    const data = await res.json();
    return { clientPrincipal: data?.clientPrincipal ?? null };
  } catch {
    return { clientPrincipal: null };
  }
}
