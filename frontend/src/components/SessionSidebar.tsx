import type { SessionMeta } from "../types";
import { BookOpen, MessageSquarePlus, Trash2 } from "lucide-react";
import { AuthBadge } from "./AuthBadge";
import { DocumentPanel } from "./DocumentPanel";
import { ModeToggle } from "./mode-toggle";
import { Button } from "./ui/button";
import { Separator } from "./ui/separator";

interface Props {
  sessions: SessionMeta[];
  activeId: string | null;
  onSelect: (sid: string) => void;
  onNewChat: () => void;
  onDelete: (sid: string) => void;
}

export function SessionSidebar({ sessions, activeId, onSelect, onNewChat, onDelete }: Props) {
  return (
    <div className="flex h-full min-h-0 w-full flex-col overflow-y-auto bg-sidebar px-3 py-4">
      <div className="flex items-center gap-2 px-2 pb-5">
        <div className="flex size-8 items-center justify-center rounded-lg bg-sidebar-accent text-sidebar-accent-foreground">
          <BookOpen className="size-4" />
        </div>
        <div>
          <p className="text-sm font-semibold tracking-tight">Papeer</p>
          <p className="text-[11px] text-muted-foreground">Research assistant</p>
        </div>
      </div>

      <Button
        type="button"
        variant="outline"
        className="w-full justify-start border-sidebar-border bg-transparent text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
        onClick={onNewChat}
      >
        <MessageSquarePlus className="size-4" />
        New chat
      </Button>

      <div className="mt-6 flex items-center justify-between px-2">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Chats</h2>
        <span className="text-[11px] text-muted-foreground">{sessions.length}</span>
      </div>

      <ul className="mt-2 space-y-0.5">
        {sessions.map((s) => (
          <li key={s.id} className="group flex items-center gap-1">
            <button
              type="button"
              onClick={() => onSelect(s.id)}
              className={`flex min-w-0 flex-1 items-center gap-2 rounded-lg px-3 py-2 text-left text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${
                s.id === activeId
                  ? "bg-sidebar-accent font-medium text-sidebar-accent-foreground"
                  : "text-sidebar-foreground hover:bg-sidebar-accent/70"
              }`}
              title={s.name}
            >
              <span className="truncate">{s.name}</span>
            </button>
            <button
              type="button"
              onClick={() => onDelete(s.id)}
              title="Delete session"
              aria-label={`Delete ${s.name}`}
              className="invisible rounded-md p-1.5 text-muted-foreground opacity-0 transition-all hover:bg-destructive/10 hover:text-destructive focus-visible:visible focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring group-hover:visible group-hover:opacity-100"
            >
              <Trash2 className="size-3.5" />
            </button>
          </li>
        ))}
      </ul>

      <Separator className="my-4 bg-sidebar-border" />

      {activeId && <DocumentPanel sid={activeId} />}

      <div className="mt-auto pt-4">
        <Separator className="mb-3 bg-sidebar-border" />
        <ModeToggle />
        <AuthBadge />
      </div>
    </div>
  );
}
