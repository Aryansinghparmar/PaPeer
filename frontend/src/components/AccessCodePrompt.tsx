import { KeyRound, Loader2 } from "lucide-react";
import { useState, type FormEvent } from "react";
import { ApiError, errorMessage, listSessions, setAccessCode } from "../api";
import { Button } from "./ui/button";

interface Props {
  /** Shown when re-prompting after a 401. */
  reason?: string | null;
  onAccepted: () => void;
}

/** Minimal gate for the backend's `api_key` auth mode. The code is checked with
 * one cheap authenticated request before the app continues. */
export function AccessCodePrompt({ reason, onAccepted }: Props) {
  const [code, setCode] = useState("");
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const value = code.trim();
    if (!value || checking) return;
    setChecking(true);
    setError(null);
    setAccessCode(value);
    try {
      await listSessions();
      onAccepted();
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        setAccessCode(null);
        setError("That access code wasn't accepted.");
      } else {
        // The code may be fine; let the app surface the other failure itself.
        setError(null);
        onAccepted();
        console.warn("Access code check failed:", errorMessage(err));
      }
    } finally {
      setChecking(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-background/90 px-4 backdrop-blur-sm">
      <form
        onSubmit={handleSubmit}
        className="w-full max-w-sm rounded-2xl border border-border bg-card p-6 text-card-foreground shadow-lg"
      >
        <div className="mb-4 flex items-center gap-2">
          <div className="flex size-8 items-center justify-center rounded-lg bg-muted">
            <KeyRound className="size-4" />
          </div>
          <h1 className="text-base font-semibold">Enter access code</h1>
        </div>
        <p className="mb-3 text-sm text-muted-foreground">
          {reason ?? "This Papeer deployment is private. Enter the access code you were given."}
        </p>
        <input
          type="password"
          autoFocus
          autoComplete="current-password"
          value={code}
          onChange={(e) => setCode(e.target.value)}
          aria-label="Access code"
          placeholder="Access code"
          disabled={checking}
          className="h-10 w-full rounded-lg border border-input bg-background px-3 text-sm outline-none placeholder:text-muted-foreground focus-visible:ring-2 focus-visible:ring-ring"
        />
        {error && <p className="mt-2 text-xs text-destructive">{error}</p>}
        <Button type="submit" className="mt-4 w-full" disabled={checking || !code.trim()}>
          {checking && <Loader2 className="size-4 animate-spin" />}
          Continue
        </Button>
      </form>
    </div>
  );
}
