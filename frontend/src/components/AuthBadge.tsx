import { useEffect, useState } from "react";
import { LogOut, UserCircle } from "lucide-react";
import { getAuthIdentity } from "../api";
import type { AuthIdentity } from "../types";

/** Reads Azure Static Web Apps identity (`/.auth/me`). Renders nothing locally,
 * where that route doesn't exist — the auth gate is enforced by the platform,
 * not by this component. */
export function AuthBadge() {
  const [identity, setIdentity] = useState<AuthIdentity["clientPrincipal"]>(null);

  useEffect(() => {
    getAuthIdentity().then((res) => setIdentity(res.clientPrincipal));
  }, []);

  if (!identity) return null;

  return (
    <div className="mt-1 flex items-center gap-2 rounded-lg px-2 py-2 text-xs text-muted-foreground">
      <UserCircle className="size-4 shrink-0" />
      <span className="min-w-0 flex-1 truncate">{identity.userDetails}</span>
      <a
        href="/.auth/logout"
        className="rounded p-1 hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
        aria-label="Sign out"
        title="Sign out"
      >
        <LogOut className="size-3.5" />
      </a>
    </div>
  );
}
