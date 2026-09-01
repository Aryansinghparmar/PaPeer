import { useEffect, useState, type ReactNode } from "react";
import { ThemeProviderContext, type Theme } from "./theme-context";

const STORAGE_KEY = "papeer-theme";
function readStoredTheme(defaultTheme: Theme): Theme {
  try {
    return localStorage.getItem(STORAGE_KEY) === "light" ? "light" : defaultTheme;
  } catch {
    return defaultTheme;
  }
}

export function ThemeProvider({ children, defaultTheme = "dark" }: { children: ReactNode; defaultTheme?: Theme }) {
  const [theme, setTheme] = useState<Theme>(() => readStoredTheme(defaultTheme));

  useEffect(() => {
    const root = window.document.documentElement;
    root.classList.toggle("dark", theme === "dark");
    try {
      localStorage.setItem(STORAGE_KEY, theme);
    } catch {
      // Theme switching still works when local storage is unavailable.
    }
  }, [theme]);

  const toggleTheme = () => setTheme((current) => (current === "dark" ? "light" : "dark"));

  return (
    <ThemeProviderContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeProviderContext.Provider>
  );
}
