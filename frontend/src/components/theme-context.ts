import { createContext, useContext } from "react";

export type Theme = "dark" | "light";

export interface ThemeProviderState {
  theme: Theme;
  toggleTheme: () => void;
}

export const ThemeProviderContext = createContext<ThemeProviderState | undefined>(undefined);

export function useTheme() {
  const context = useContext(ThemeProviderContext);
  if (!context) throw new Error("useTheme must be used inside ThemeProvider");
  return context;
}
