import { useEffect, useRef } from "react";
import { BookOpen, CircleAlert, Loader2, Sparkles } from "lucide-react";
import type { BtwExchange } from "../hooks/useBtw";
import type { ChatMessage } from "../types";
import { Markdown } from "./Markdown";
import { MessageBubble } from "./MessageBubble";
import { Button } from "./ui/button";

interface Props {
  messages: ChatMessage[];
  streaming: boolean;
  streamText: string;
  statusLabel: string | null;
  error: string | null;
  historyError: string | null;
  historyLoading: boolean;
  onRetryHistory: () => void;
  btw: BtwExchange | null;
  btwStreaming: boolean;
  btwError: string | null;
}

function ErrorBanner({ children, onRetry }: { children: string; onRetry?: () => void }) {
  return (
    <div
      role="alert"
      className="my-2 flex items-start gap-2 rounded-xl border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive"
    >
      <CircleAlert className="mt-0.5 size-4 shrink-0" />
      <span className="min-w-0 flex-1">{children}</span>
      {onRetry && (
        <Button type="button" variant="outline" size="sm" onClick={onRetry} className="h-7 shrink-0">
          Retry
        </Button>
      )}
    </div>
  );
}

/** "Searching arXiv and the web" → "Searching arXiv and the web…" */
function progressText(label: string | null) {
  const text = label?.trim() || "Thinking";
  return /[….]$/.test(text) ? text : `${text}…`;
}

function Pending({ label }: { label: string | null }) {
  return (
    <p className="flex items-center gap-2 text-sm leading-7 text-muted-foreground" aria-live="polite">
      <Loader2 className="size-3.5 animate-spin" />
      {progressText(label)}
    </p>
  );
}

export function ChatView({
  messages,
  streaming,
  streamText,
  statusLabel,
  error,
  historyError,
  historyLoading,
  onRetryHistory,
  btw,
  btwStreaming,
  btwError,
}: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamText, statusLabel, btw, error, btwError]);

  const isEmpty =
    messages.length === 0 && !streaming && !btw && !historyError && !historyLoading;

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto flex min-h-full w-full max-w-3xl flex-col px-4 py-8 sm:px-6 sm:py-10">
        {isEmpty && (
          <div className="m-auto flex max-w-md flex-col items-center text-center">
            <div className="mb-5 flex size-12 items-center justify-center rounded-2xl bg-muted text-foreground">
              <BookOpen className="size-6" />
            </div>
            <h2 className="text-xl font-semibold tracking-tight">How can I help with your papers?</h2>
            <p className="mt-2 text-sm leading-6 text-muted-foreground">
              Upload a paper in the sidebar, then ask a question about its methods, results, or claims.
            </p>
          </div>
        )}
        {historyLoading && (
          <div className="m-auto">
            <Pending label="Loading chat" />
          </div>
        )}
        {historyError && <ErrorBanner onRetry={onRetryHistory}>{historyError}</ErrorBanner>}
        {messages.map((m, i) => (
          <MessageBubble key={i} message={m} />
        ))}
        {streaming && (
          <div className="flex gap-3 py-5">
            <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-foreground text-background">
              <Sparkles className="size-3.5" />
            </div>
            <div className="min-w-0 flex-1 pt-0.5">
              {streamText ? (
                <div className="markdown-body stream-cursor text-sm leading-7">
                  <Markdown>{streamText}</Markdown>
                </div>
              ) : (
                <Pending label={statusLabel} />
              )}
            </div>
          </div>
        )}
        {error && <ErrorBanner>{error}</ErrorBanner>}
        {btw && (
          <>
            <div className="flex justify-end py-3">
              <div className="max-w-[85%] rounded-3xl bg-muted px-4 py-3 text-sm">
                <p className="whitespace-pre-wrap">/btw {btw.query}</p>
                <p className="mt-1 text-xs text-muted-foreground">
                  Side channel — not saved to session history.
                </p>
              </div>
            </div>
            <div className="flex gap-3 py-3">
              <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-foreground text-background">
                <Sparkles className="size-3.5" />
              </div>
              <div className="min-w-0 flex-1 pt-0.5">
                {btw.usage ? (
                  <p className="text-sm leading-7">
                    Please add a question after <code className="rounded bg-muted px-1">/btw</code>, e.g.{" "}
                    <code className="rounded bg-muted px-1">/btw What is attention?</code>
                  </p>
                ) : btw.answer ? (
                  <div className={`markdown-body text-sm leading-7 ${btwStreaming ? "stream-cursor" : ""}`}>
                    <Markdown>{btw.answer}</Markdown>
                  </div>
                ) : btwStreaming ? (
                  <Pending label={null} />
                ) : null}
                {btw.stopped && (
                  <p className="mt-1 text-xs italic text-muted-foreground">
                    {btw.answer ? "Stopped." : "Stopped before an answer was generated."}
                  </p>
                )}
                {btwError && <ErrorBanner>{btwError}</ErrorBanner>}
                <p className="mt-1 text-xs text-muted-foreground">
                  Side channel — not saved to session history.
                </p>
              </div>
            </div>
          </>
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
