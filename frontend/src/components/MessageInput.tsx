import { ArrowUp, Loader2 } from "lucide-react";
import { useState, type FormEvent, type KeyboardEvent } from "react";
import { Button } from "./ui/button";
import { Textarea } from "./ui/textarea";

interface Props {
  disabled: boolean;
  onSend: (text: string) => void;
  onBtw: (query: string) => void;
}

/** Mirrors app.py's chat_input: a leading `/btw` routes to the side channel
 * instead of the main graph. */
export function MessageInput({ disabled, onSend, onBtw }: Props) {
  const [value, setValue] = useState("");

  const submit = () => {
    const text = value.trim();
    if (!text || disabled) return;
    if (text.toLowerCase().startsWith("/btw")) {
      const query = text.slice(4).trim();
      if (query) onBtw(query);
    } else {
      onSend(text);
    }
    setValue("");
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
          className="max-h-40 min-h-10 flex-1 border-0 bg-transparent px-2 py-2 shadow-none focus-visible:ring-0"
        />
        <Button
          type="submit"
          variant="default"
          size="icon"
          disabled={disabled || !value.trim()}
          className="size-9 shrink-0 rounded-full"
          aria-label={disabled ? "Waiting for response" : "Send message"}
        >
          {disabled ? <Loader2 className="size-4 animate-spin" /> : <ArrowUp className="size-4" />}
        </Button>
      </form>
      <p className="mx-auto mt-2 max-w-3xl text-center text-[11px] text-muted-foreground">
        Papeer can make mistakes. Check important information in the source paper.
      </p>
    </div>
  );
}
