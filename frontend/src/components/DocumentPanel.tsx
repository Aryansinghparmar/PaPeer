import { useEffect, useState } from "react";
import { FileText, Link2, Loader2, Upload } from "lucide-react";
import { listDocuments, loadArxiv, loadUrls, uploadDocuments } from "../api";
import { Button } from "./ui/button";
import { Separator } from "./ui/separator";
import { Textarea } from "./ui/textarea";

interface Props {
  sid: string;
}

type Status = { kind: "success" | "error"; text: string } | null;

/** Mirrors app.py's sidebar: file upload, web URL loader, ArXiv loader, and the
 * loaded-documents list — scoped to the active session. */
export function DocumentPanel({ sid }: Props) {
  const [documents, setDocuments] = useState<string[] | null>(null);
  const [urlText, setUrlText] = useState("");
  const [arxivQuery, setArxivQuery] = useState("");
  const [busy, setBusy] = useState<"upload" | "url" | "arxiv" | null>(null);
  const [status, setStatus] = useState<Status>(null);

  const refresh = async () => {
    try {
      const { documents } = await listDocuments(sid);
      setDocuments(documents);
    } catch {
      setDocuments(null);
    }
  };

  useEffect(() => {
    setDocuments(null);
    setStatus(null);
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sid]);

  const handleUpload = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    setBusy("upload");
    setStatus(null);
    try {
      const { added } = await uploadDocuments(sid, Array.from(files));
      setStatus({ kind: "success", text: `Added: ${added.join(", ")}` });
      await refresh();
    } catch (err) {
      setStatus({ kind: "error", text: String(err) });
    } finally {
      setBusy(null);
    }
  };

  const handleLoadUrls = async () => {
    const urls = urlText.split("\n").map((u) => u.trim()).filter(Boolean);
    if (urls.length === 0) return;
    setBusy("url");
    setStatus(null);
    try {
      const { loaded } = await loadUrls(sid, urls);
      setStatus({ kind: "success", text: `Loaded: ${loaded.length} page(s)` });
      setUrlText("");
      await refresh();
    } catch (err) {
      setStatus({ kind: "error", text: String(err) });
    } finally {
      setBusy(null);
    }
  };

  const handleLoadArxiv = async () => {
    if (!arxivQuery.trim()) return;
    setBusy("arxiv");
    setStatus(null);
    try {
      const { loaded } = await loadArxiv(sid, arxivQuery.trim());
      setStatus({ kind: "success", text: `Loaded: ${loaded}` });
      setArxivQuery("");
      await refresh();
    } catch (err) {
      setStatus({ kind: "error", text: String(err) });
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="space-y-4 text-sidebar-foreground">
      <div className="flex items-center gap-2 px-2">
        <FileText className="size-4 text-muted-foreground" />
        <h2 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Documents</h2>
      </div>

      <div>
        <p className="mb-1 px-2 text-xs font-medium text-muted-foreground">Upload files</p>
        <input
          type="file"
          multiple
          accept=".pdf,.txt,.md,.markdown"
          disabled={busy !== null}
          onChange={(e) => handleUpload(e.target.files)}
          className="block w-full text-xs text-muted-foreground file:mr-2 file:rounded-lg file:border-0 file:bg-sidebar-accent file:px-2 file:py-1 file:text-sidebar-accent-foreground"
        />
      </div>

      <div>
        <p className="mb-1 flex items-center gap-1 px-2 text-xs font-medium text-muted-foreground">
          <Link2 className="size-3.5" /> Web pages
        </p>
        <Textarea
          value={urlText}
          onChange={(e) => setUrlText(e.target.value)}
          placeholder="https://example.com/paper"
          rows={2}
          disabled={busy !== null}
          className="min-h-14 rounded-lg border-sidebar-border bg-sidebar text-xs"
        />
        <Button
          type="button"
          variant="secondary"
          size="sm"
          onClick={handleLoadUrls}
          disabled={busy !== null || !urlText.trim()}
          className="mt-1 w-full bg-sidebar-accent text-xs text-sidebar-accent-foreground hover:bg-sidebar-accent/80"
        >
          {busy === "url" && <Loader2 className="size-3.5 animate-spin" />}
          {busy === "url" ? "Loading…" : "Load URLs"}
        </Button>
      </div>

      <div>
        <p className="mb-1 px-2 text-xs font-medium text-muted-foreground">ArXiv papers</p>
        <input
          value={arxivQuery}
          onChange={(e) => setArxivQuery(e.target.value)}
          placeholder="1706.03762 or Attention Is All You Need"
          disabled={busy !== null}
          className="h-9 w-full rounded-lg border border-sidebar-border bg-sidebar px-2 py-1 text-xs text-sidebar-foreground outline-none placeholder:text-muted-foreground focus-visible:ring-2 focus-visible:ring-ring"
        />
        <Button
          type="button"
          variant="secondary"
          size="sm"
          onClick={handleLoadArxiv}
          disabled={busy !== null || !arxivQuery.trim()}
          className="mt-1 w-full bg-sidebar-accent text-xs text-sidebar-accent-foreground hover:bg-sidebar-accent/80"
        >
          {busy === "arxiv" && <Loader2 className="size-3.5 animate-spin" />}
          {busy === "arxiv" ? "Loading…" : "Load ArXiv Paper"}
        </Button>
      </div>

      {status && (
        <p className={`text-xs ${status.kind === "error" ? "text-red-600" : "text-emerald-600"}`}>
          {status.text}
        </p>
      )}

      <div>
        <Separator className="mb-3 bg-sidebar-border" />
        <p className="mb-1 px-2 text-xs font-medium text-muted-foreground">Loaded documents</p>
        {documents === null ? (
          <p className="px-2 text-xs text-muted-foreground">Could not load document list — try refreshing.</p>
        ) : documents.length === 0 ? (
          <p className="px-2 text-xs text-muted-foreground">No documents loaded yet.</p>
        ) : (
          <ul className="space-y-0.5 text-xs text-sidebar-foreground">
            {documents.map((title) => (
              <li key={title} className="flex items-center gap-2 truncate rounded-md px-2 py-1">
                <Upload className="size-3 shrink-0 text-muted-foreground" />
                <span className="truncate">{title}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
