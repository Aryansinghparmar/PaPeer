import { Sparkles, UserRound } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ChatMessage } from "../types";
import { GraphStateDrawer } from "./GraphStateDrawer";

interface Props {
  message: ChatMessage;
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
          <div className="markdown-body text-sm leading-7">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
          </div>
        )}
        {!isUser && <GraphStateDrawer turn={message.turn} state={message.graph_state} />}
      </div>
    </div>
  );
}
