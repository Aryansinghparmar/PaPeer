# Papeer frontend

React single-page app for Papeer, the research-paper RAG assistant. It talks to the
FastAPI backend in `../api` and covers chat sessions, document ingestion (PDF/text
upload, web URLs, arXiv), streamed answers, the `/btw` side channel and a per-turn
graph-state inspector.

Stack: Vite, React 19, TypeScript, Tailwind CSS v4, shadcn/ui (Radix), `react-markdown`,
and `@microsoft/fetch-event-source` for SSE over POST.

## Scripts

| Command           | What it does                                        |
| ----------------- | --------------------------------------------------- |
| `npm install`     | Install dependencies                                |
| `npm run dev`     | Dev server on http://localhost:5173 with API proxy  |
| `npm run build`   | Type-check (`tsc -b`) and build to `dist/`          |
| `npm run lint`    | Lint with oxlint                                    |
| `npm run preview` | Serve the production build locally                  |

Local development: start the API (`uv run uvicorn api.main:app --port 8000` from the
repo root), then `npm run dev` here.

## Environment variables

Copy `.env.example` to `.env.local` if you need to change anything.

| Variable            | Used by     | Default                 | Meaning                                                                                       |
| ------------------- | ----------- | ----------------------- | --------------------------------------------------------------------------------------------- |
| `VITE_API_BASE_URL` | the bundle  | `""` (same origin)      | Backend origin. Leave empty in production and in dev (the proxy handles it). Cross-origin needs CORS. |
| `API_PROXY_TARGET`  | dev server  | `http://127.0.0.1:8000` | Where `npm run dev` proxies `/api` and `/health`.                                              |

## How it talks to the API

- All calls live in `src/api.ts`. Non-2xx responses are parsed as JSON
  `{detail, code}` and turned into friendly messages (rate limit, daily budget,
  access code, not found, input too long).
- **Auth.** On load the app calls `GET /api/auth/mode`:
  - `none`: no prompt.
  - `api_key`: asks for an access code, keeps it in `sessionStorage` (in memory
    if storage is blocked) and sends `Authorization: Bearer <code>` on every
    request, including the SSE POSTs. Any 401 brings the prompt back.
  - `easyauth`: the platform handles sign-in; the sidebar shows the identity from
    `/.auth/me`.
- **Uploads.** `POST /api/sessions/{sid}/documents` (multipart `files`, up to 5 files
  of 20 MB each) and `POST …/documents/url` (`{urls}`, up to 5) return per-item
  `added`/`loaded`, `skipped` and `failed` lists. The panel shows all three and
  always re-reads the document list afterwards.

## Streaming contract (SSE)

`POST /api/sessions/{sid}/chat` with `{"message"}` (max 4000 chars) and
`POST /api/btw` with `{"query"}` (max 2000 chars) respond with `text/event-stream`.
Each event is `data: <json>`:

| `type`   | Fields                                                                              | UI                                          |
| -------- | ----------------------------------------------------------------------------------- | ------------------------------------------- |
| `status` | `stage`, `label`                                                                    | Progress line, such as "Searching arXiv and the web…" |
| `token`  | `content`                                                                           | Appended to the streaming answer            |
| `done`   | `answer`, `state`, `observability` (`latency_seconds`, `input_tokens`, `output_tokens`, `estimated_cost_usd`) | Final message, graph-state drawer, muted stats line |
| `error`  | `detail`, `code` (`timeout`, `loop_limit`, `internal`)                              | Error banner                                |

Heartbeat comment lines (`: ping`) are ignored. Errors raised before the stream
opens (401, 404, 422, 429) arrive as ordinary JSON responses.

The client sends each turn exactly once. It does not reconnect when the tab is
hidden (`openWhenHidden: true`) and never retries automatically, because a
re-POST would run the whole graph again. A stream that closes without `done` or
`error` is reported as an error. **Stop** aborts the request and keeps any partial
text, marked "Stopped". Switching sessions also aborts the in-flight turn.

Model output is rendered as Markdown with raw HTML disabled. Images are not
loaded; they appear as a link, so a prompt-injected `![](https://…)` cannot leak
data. Links open in a new tab with `rel="noopener noreferrer nofollow"`.
