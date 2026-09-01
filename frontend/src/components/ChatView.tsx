import { useEffect, useRef } from "react";
import { BookOpen, CircleAlert, Sparkles } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { BtwExchange } from "../hooks/useBtw";
import type { ChatMessage } from "../types";
import { MessageBubble } from "./MessageBubble";

interface Props {
  messages: ChatMessage[];
  streaming: boolean;
  streamText: string;
  error: string | null;
  btw: BtwExchange | null;
  btwStreaming: boolean;
}

export function ChatView({ messages, streaming, streamText, error, btw, btwStreaming }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamText, btw]);

  const isEmpty = messages.length === 0 && !streaming && !btw;

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
        {messages.map((m, i) => (
          <MessageBubble key={i} message={m} />
        ))}
        {streaming && (
          <div className="flex gap-3 py-5">
            <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-foreground text-background">
              <Sparkles className="size-3.5" />
            </div>
            <div className="min-w-0 flex-1 pt-0.5">
              <div className="markdown-body stream-cursor text-sm leading-7">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {streamText || "Thinking…"}
                </ReactMarkdown>
              </div>
            </div>
          </div>
        )}
        {error && (
          <div className="flex items-start gap-2 rounded-xl border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
            <CircleAlert className="mt-0.5 size-4 shrink-0" />
            {error}
          </div>
        )}
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
                <div className={`markdown-body text-sm leading-7 ${btwStreaming ? "stream-cursor" : ""}`}>
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {btw.answer || "Thinking…"}
                  </ReactMarkdown>
                </div>
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
