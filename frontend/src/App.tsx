import { Menu } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { ChatView } from "./components/ChatView";
import { MessageInput } from "./components/MessageInput";
import { SessionSidebar } from "./components/SessionSidebar";
import { Button } from "./components/ui/button";
import { Sheet, SheetContent, SheetDescription, SheetTitle } from "./components/ui/sheet";
import { useBtw } from "./hooks/useBtw";
import { useChat } from "./hooks/useChat";
import { useSessions } from "./hooks/useSessions";

export default function App() {
  const { sessions, activeId, setActiveId, loading, refresh, newChat, remove } = useSessions();
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);

  const onFirstMessageSent = useCallback(
    (_sid: string) => {
      // The backend auto-names the session on its first turn; refresh to pick it up.
      refresh();
    },
    [refresh],
  );

  const selectSession = useCallback(
    (sid: string) => {
      setActiveId(sid);
      setMobileSidebarOpen(false);
    },
    [setActiveId],
  );

  const createNewChat = useCallback(() => {
    setMobileSidebarOpen(false);
    void newChat();
  }, [newChat]);

  const renderSidebar = () => (
    <SessionSidebar
      sessions={sessions}
      activeId={activeId}
      onSelect={selectSession}
      onNewChat={createNewChat}
      onDelete={remove}
    />
  );

  const { messages, streaming, streamText, error, send } = useChat(activeId, onFirstMessageSent);
  const { exchange: btw, streaming: btwStreaming, ask: askBtw, reset: resetBtw } = useBtw();

  // /btw is a side channel that isn't persisted — clear it on session switch so
  // it doesn't linger in a different session's chat view, matching the
  // Streamlit original where it only rendered for a single script run.
  useEffect(() => {
    resetBtw();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeId]);

  if (loading) {
    return (
      <div className="flex min-h-svh items-center justify-center bg-background text-muted-foreground">
        Loading Papeer…
      </div>
    );
  }

  return (
    <div className="flex min-h-svh bg-background text-foreground">
      <aside className="hidden h-svh w-64 shrink-0 border-r border-sidebar-border bg-sidebar lg:flex">
        {renderSidebar()}
      </aside>

      <Sheet open={mobileSidebarOpen} onOpenChange={setMobileSidebarOpen}>
        <SheetContent side="left" className="w-[18rem] border-sidebar-border bg-sidebar p-0 text-sidebar-foreground sm:max-w-xs">
          <SheetTitle className="sr-only">Papeer navigation</SheetTitle>
          <SheetDescription className="sr-only">Sessions and document controls</SheetDescription>
          {renderSidebar()}
        </SheetContent>
      </Sheet>

      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-14 shrink-0 items-center border-b border-border/70 px-3 sm:px-6">
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="mr-2 lg:hidden"
            onClick={() => setMobileSidebarOpen(true)}
            aria-label="Open navigation"
          >
            <Menu className="size-5" />
          </Button>
          <div className="min-w-0">
            <h1 className="truncate text-sm font-semibold sm:text-base">Papeer</h1>
            <p className="hidden text-xs text-muted-foreground sm:block">
              Research paper assistant
            </p>
          </div>
        </header>

        <ChatView
          messages={messages}
          streaming={streaming}
          streamText={streamText}
          error={error}
          btw={btw}
          btwStreaming={btwStreaming}
        />

        <MessageInput disabled={streaming || !activeId} onSend={send} onBtw={askBtw} />
      </main>
    </div>
  );
}
