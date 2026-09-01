import { fetchEventSource } from "@microsoft/fetch-event-source";
import type { AuthIdentity, ChatEvent, ChatMessage, SessionMeta } from "./types";

const BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status} ${detail}`);
  }
  return res.json() as Promise<T>;
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

export async function uploadDocuments(sid: string, files: File[]) {
  const form = new FormData();
  for (const f of files) form.append("files", f);
  const res = await fetch(`${BASE_URL}/api/sessions/${sid}/documents`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(await res.text().catch(() => res.statusText));
  return res.json() as Promise<{ added: string[] }>;
}

export const loadUrls = (sid: string, urls: string[]) =>
  json<{ loaded: string[] }>(`/api/sessions/${sid}/documents/url`, {
    method: "POST",
    body: JSON.stringify({ urls }),
  });

export const loadArxiv = (sid: string, query: string) =>
  json<{ loaded: string }>(`/api/sessions/${sid}/documents/arxiv`, {
    method: "POST",
    body: JSON.stringify({ query }),
  });

// ── Streaming chat (SSE over POST) ──────────────────────────────────────────

interface StreamOptions {
  signal?: AbortSignal;
  onEvent: (event: ChatEvent) => void;
  onError?: (err: unknown) => void;
}

async function streamPost(path: string, body: unknown, opts: StreamOptions) {
  await fetchEventSource(`${BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: opts.signal,
    async onopen(res) {
      if (!res.ok) throw new Error(`Stream failed to open: ${res.status}`);
    },
    onmessage(msg) {
      if (!msg.data) return;
      opts.onEvent(JSON.parse(msg.data) as ChatEvent);
    },
    onerror(err) {
      opts.onError?.(err);
      throw err; // stop retrying — a single request/response, not a live feed
    },
  });
}

export const streamChat = (sid: string, message: string, opts: StreamOptions) =>
  streamPost(`/api/sessions/${sid}/chat`, { message }, opts);

export const streamBtw = (query: string, opts: StreamOptions) =>
  streamPost(`/api/btw`, { query }, opts);

// ── Auth identity (Azure Static Web Apps) ───────────────────────────────────
// Present only when deployed behind SWA; resolves to no identity in local dev.

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
