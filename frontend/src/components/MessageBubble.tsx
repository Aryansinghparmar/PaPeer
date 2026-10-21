import { Sparkles, UserRound } from "lucide-react";
import type { ChatMessage, Observability } from "../types";
import { GraphStateDrawer } from "./GraphStateDrawer";
import { Markdown } from "./Markdown";

interface Props {
  message: ChatMessage;
}

function formatObservability(o: Observability): string {
  const parts = [`${o.latency_seconds.toFixed(1)}s`];
  parts.push(`${o.input_tokens.toLocaleString()} in / ${o.output_tokens.toLocaleString()} out tokens`);
  if (o.estimated_cost_usd != null) parts.push(`$${o.estimated_cost_usd.toFixed(4)}`);
  return parts.join(" · ");
}

export function MessageBubble({ message }: Props) {
  const isUser = message.role === "user";
  return (
    <div className={`flex gap-3 py-5 ${isUser ? "justify-end" : "justify-start"}`}>
      {!isUser && (
        <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-foreground text-background">
          <Sparkles className="size-3.5" />
        </div>
      )}
      <div className={`min-w-0 ${isUser ? "max-w-[85%]" : "flex-1"}`}>
        {isUser ? (
          <div className="flex items-start gap-2 rounded-3xl bg-muted px-4 py-3 text-sm leading-6">
            <p className="whitespace-pre-wrap">{message.content}</p>
            <UserRound className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
          </div>
        ) : (
          message.content && (
            <div className="markdown-body text-sm leading-7">
              <Markdown>{message.content}</Markdown>
            </div>
          )
        )}
        {message.stopped && (
          <p className="mt-1 text-xs italic text-muted-foreground">
            {message.content ? "Stopped." : "Stopped before an answer was generated."}
          </p>
        )}
        {!isUser && message.observability && (
          <p className="mt-1 text-[11px] text-muted-foreground" title="Latency · tokens · estimated cost">
            {formatObservability(message.observability)}
          </p>
        )}
        {!isUser && <GraphStateDrawer turn={message.turn} state={message.graph_state} />}
      </div>
    </div>
  );
}
