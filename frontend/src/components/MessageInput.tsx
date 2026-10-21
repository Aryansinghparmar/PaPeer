import { ArrowUp, Square } from "lucide-react";
import { useState, type FormEvent, type KeyboardEvent } from "react";
import { Button } from "./ui/button";
import { Textarea } from "./ui/textarea";

// Server-side limits (422 beyond these); checked here to fail fast.
const MAX_MESSAGE_CHARS = 4000;
const MAX_BTW_CHARS = 2000;

interface Props {
  /** No active session / history unavailable: nothing can be sent. */
  disabled: boolean;
  /** A chat turn or /btw answer is streaming: show Stop instead of Send. */
  streaming: boolean;
  /** May resolve `false` when the message was rejected, to restore the draft. */
  onSend: (text: string) => Promise<boolean> | void;
  onBtw: (query: string) => void;
  onStop: () => void;
}

function parse(text: string) {
  const isBtw = text.toLowerCase().startsWith("/btw");
  const body = isBtw ? text.slice(4).trim() : text;
  const limit = isBtw ? MAX_BTW_CHARS : MAX_MESSAGE_CHARS;
  return { isBtw, body, limit, tooLong: body.length > limit };
}

/** Mirrors app.py's chat_input: a leading `/btw` routes to the side channel
 * instead of the main graph. */
export function MessageInput({ disabled, streaming, onSend, onBtw, onStop }: Props) {
  const [value, setValue] = useState("");
  const text = value.trim();
  const { isBtw, body, limit, tooLong } = parse(text);
  const canSubmit = !!text && !disabled && !streaming && !tooLong;

  const submit = () => {
    if (!canSubmit) return;
    setValue("");
    if (isBtw) {
      onBtw(body); // empty body → usage hint
      return;
    }
    const result = onSend(body);
    if (result) {
      void result.then((ok) => {
        if (!ok) setValue((current) => current || text);
      });
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  const handleSubmit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    submit();
  };

  return (
    <div className="bg-background px-3 pb-4 pt-2 sm:px-6 sm:pb-6">
      <form onSubmit={handleSubmit} className="mx-auto flex max-w-3xl items-end gap-2 rounded-3xl border border-input bg-card px-3 py-2 shadow-sm focus-within:ring-2 focus-within:ring-ring/30">
        <Textarea
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          rows={1}
          placeholder="Message Papeer…"
          aria-label="Message Papeer"
          aria-invalid={tooLong || undefined}
          className="max-h-40 min-h-10 flex-1 border-0 bg-transparent px-2 py-2 shadow-none focus-visible:ring-0"
        />
        {streaming ? (
          <Button
            type="button"
            variant="default"
            size="icon"
            onClick={onStop}
            className="size-9 shrink-0 rounded-full"
            aria-label="Stop generating"
            title="Stop generating"
          >
            <Square className="size-3.5 fill-current" />
          </Button>
        ) : (
          <Button
            type="submit"
            variant="default"
            size="icon"
            disabled={!canSubmit}
            className="size-9 shrink-0 rounded-full"
            aria-label="Send message"
          >
            <ArrowUp className="size-4" />
          </Button>
        )}
      </form>
      <p
        className={`mx-auto mt-2 max-w-3xl text-center text-[11px] ${tooLong ? "text-destructive" : "text-muted-foreground"}`}
      >
        {tooLong
          ? `${isBtw ? "/btw questions" : "Messages"} are limited to ${limit.toLocaleString()} characters (currently ${body.length.toLocaleString()}).`
          : "Papeer can make mistakes. Check important information in the source paper."}
      </p>
    </div>
  );
}
