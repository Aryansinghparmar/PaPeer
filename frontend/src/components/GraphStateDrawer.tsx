interface Props {
  turn?: number;
  state?: Record<string, unknown>;
}

/** Per-turn LangGraph state inspector — parity with the Streamlit expander. */
export function GraphStateDrawer({ turn, state }: Props) {
  if (!state || Object.keys(state).length === 0) return null;
  return (
    <details className="mt-3 border-t border-border pt-2 text-xs">
      <summary className="cursor-pointer select-none text-muted-foreground hover:text-foreground">
        Graph state · turn {turn ?? "?"}
      </summary>
      <pre className="mt-2 max-h-80 overflow-auto rounded-lg bg-muted p-3 text-xs text-muted-foreground">
        {JSON.stringify(state, null, 2)}
      </pre>
    </details>
  );
}
