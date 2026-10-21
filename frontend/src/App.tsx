import { CircleAlert, Loader2, Menu, X } from "lucide-react";
import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { errorMessage, getAccessCode, getAuthMode, setUnauthorizedHandler } from "./api";
import { AccessCodePrompt } from "./components/AccessCodePrompt";
import { ChatView } from "./components/ChatView";
import { MessageInput } from "./components/MessageInput";
import { SessionSidebar } from "./components/SessionSidebar";
import { Button } from "./components/ui/button";
import { Sheet, SheetContent, SheetDescription, SheetTitle } from "./components/ui/sheet";
import { useBtw } from "./hooks/useBtw";
import { useChat } from "./hooks/useChat";
import { useSessions } from "./hooks/useSessions";
import type { AuthMode } from "./types";

function FullScreenMessage({
  children,
  onRetry,
}: {
  children: ReactNode;
  onRetry?: () => void;
}) {
  return (
    <div className="flex min-h-svh flex-col items-center justify-center gap-4 bg-background px-4 text-center text-muted-foreground">
      {children}
      {onRetry && (
        <Button type="button" variant="outline" onClick={onRetry}>
          Retry
        </Button>
      )}
    </div>
  );
}

/** Resolves the backend auth mode first, then gates the workspace behind an
 * access code when the API runs in `api_key` mode. */
export default function App() {
  const [mode, setMode] = useState<AuthMode | null>(null);
  const [modeError, setModeError] = useState<string | null>(null);
  const [needCode, setNeedCode] = useState(false);
  const [codeReason, setCodeReason] = useState<string | null>(null);
  const [authEpoch, setAuthEpoch] = useState(0); // bumps on every accepted code

  const loadMode = useCallback(async () => {
    setModeError(null);
    try {
      const m = await getAuthMode();
      setMode(m);
      setNeedCode(m === "api_key" && !getAccessCode());
    } catch (err) {
      setModeError(errorMessage(err));
    }
  }, []);

  useEffect(() => {
    void loadMode();
  }, [loadMode]);

  // Any 401 (including from an SSE POST) re-prompts for the access code.
  useEffect(() => {
    if (mode !== "api_key") return;
    setUnauthorizedHandler(() => {
      setCodeReason("Your access code was rejected or has expired. Please enter it again.");
      setNeedCode(true);
    });
    return () => setUnauthorizedHandler(null);
  }, [mode]);

  if (modeError) {
    return (
      <FullScreenMessage onRetry={loadMode}>
        <CircleAlert className="size-6 text-destructive" />
        <p className="max-w-md text-sm">Couldn't reach Papeer. {modeError}</p>
      </FullScreenMessage>
    );
  }

  if (!mode) {
    return (
      <FullScreenMessage>
        <Loader2 className="size-5 animate-spin" />
        Loading Papeer…
      </FullScreenMessage>
    );
  }

  const prompt = needCode && (
    <AccessCodePrompt
      reason={codeReason}
      onAccepted={() => {
        setNeedCode(false);
        setCodeReason(null);
        setAuthEpoch((n) => n + 1);
      }}
    />
  );

  // Never bootstrap sessions before an access code exists (it would just 401).
  if (mode === "api_key" && !getAccessCode()) return prompt;

  return (
    <>
      <Workspace authMode={mode} authEpoch={authEpoch} />
      {prompt}
    </>
  );
}

function Workspace({ authMode, authEpoch }: { authMode: AuthMode; authEpoch: number }) {
  const {
    sessions,
    activeId,
    setActiveId,
    loading,
    bootError,
    retry,
    actionError,
    dismissActionError,
    refresh,
    newChat,
    remove,
  } = useSessions();
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);

  const onFirstMessageSent = useCallback(
    (_sid: string) => {
      // The backend auto-names the session on its first turn; refresh to pick it up.
      void refresh();
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
      showAuthBadge={authMode === "easyauth"}
    />
  );

  const chat = useChat(activeId, onFirstMessageSent);
  const {
    exchange: btw,
    streaming: btwStreaming,
    error: btwError,
    ask: askBtw,
    stop: stopBtw,
    reset: resetBtw,
  } = useBtw();

  // /btw is a side channel that isn't persisted — clear it on session switch so
  // it doesn't linger in a different session's chat view, matching the
  // Streamlit original where it only rendered for a single script run.
  useEffect(() => {
    resetBtw();
  }, [activeId, resetBtw]);

  const stopChat = chat.stop;
  const stop = useCallback(() => {
    stopChat();
    stopBtw();
  }, [stopChat, stopBtw]);

  // After a re-entered access code, retry a bootstrap that failed on the 401.
  const handledEpoch = useRef(authEpoch);
  useEffect(() => {
    if (authEpoch === handledEpoch.current) return;
    handledEpoch.current = authEpoch;
    if (bootError) void retry();
  }, [authEpoch, bootError, retry]);

  if (loading) {
    return (
      <FullScreenMessage>
        <Loader2 className="size-5 animate-spin" />
        Loading Papeer…
      </FullScreenMessage>
    );
  }

  if (bootError) {
    return (
      <FullScreenMessage onRetry={retry}>
        <CircleAlert className="size-6 text-destructive" />
        <p className="max-w-md text-sm">Couldn't load your chats. {bootError}</p>
      </FullScreenMessage>
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

        {actionError && (
          <div
            role="alert"
            className="flex items-center gap-2 border-b border-destructive/30 bg-destructive/10 px-4 py-2 text-sm text-destructive sm:px-6"
          >
            <CircleAlert className="size-4 shrink-0" />
            <span className="min-w-0 flex-1">{actionError}</span>
            <button
              type="button"
              onClick={dismissActionError}
              className="rounded p-1 hover:bg-destructive/10"
              aria-label="Dismiss"
            >
              <X className="size-3.5" />
            </button>
          </div>
        )}

        <ChatView
          messages={chat.messages}
          streaming={chat.streaming}
          streamText={chat.streamText}
          statusLabel={chat.statusLabel}
          error={chat.error}
          historyError={chat.historyError}
          historyLoading={chat.historyLoading}
          onRetryHistory={chat.retryHistory}
          btw={btw}
          btwStreaming={btwStreaming}
          btwError={btwError}
        />

        <MessageInput
          disabled={!activeId || chat.historyLoading || !!chat.historyError}
          streaming={chat.streaming || btwStreaming}
          onSend={chat.send}
          onBtw={askBtw}
          onStop={stop}
        />
      </main>
    </div>
  );
}
