"use client";

/** Persistent accessibility settings: theme, text size, dyslexia font.
 *  Applied as attributes / CSS vars on <html>; a tiny inline script in the
 *  layout applies them before first paint so there is no flash. */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";

export type Theme = "system" | "light" | "dark" | "hc";
export const TEXT_STEPS = [1, 1.125, 1.25, 1.5, 1.75, 2] as const;

export interface A11ySettings {
  theme: Theme;
  textStep: number; // index into TEXT_STEPS
  dyslexicFont: boolean;
}

const DEFAULTS: A11ySettings = { theme: "system", textStep: 0, dyslexicFont: false };
const STORAGE_KEY = "iris-a11y";

export function applyToDocument(s: A11ySettings) {
  const root = document.documentElement;
  if (s.theme === "system") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", s.theme);
  if (s.dyslexicFont) root.setAttribute("data-font", "dyslexic");
  else root.removeAttribute("data-font");
  root.style.setProperty(
    "--text-zoom",
    String(TEXT_STEPS[s.textStep] ?? 1),
  );
}

/** Inline bootstrap: runs before paint, mirrors applyToDocument(). */
export const BOOT_SCRIPT = `(function(){try{
var s=JSON.parse(localStorage.getItem(${JSON.stringify(STORAGE_KEY)})||"{}");
var r=document.documentElement;
if(s.theme&&s.theme!=="system")r.setAttribute("data-theme",s.theme);
if(s.dyslexicFont)r.setAttribute("data-font","dyslexic");
var steps=[1,1.125,1.25,1.5,1.75,2];
r.style.setProperty("--text-zoom",String(steps[s.textStep||0]||1));
}catch(e){}})();`;

interface Ctx {
  settings: A11ySettings;
  update: (patch: Partial<A11ySettings>) => void;
}

const SettingsContext = createContext<Ctx>({
  settings: DEFAULTS,
  update: () => {},
});

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<A11ySettings>(DEFAULTS);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) setSettings({ ...DEFAULTS, ...JSON.parse(raw) });
    } catch {
      /* private mode etc. — defaults stand */
    }
  }, []);

  const update = useCallback((patch: Partial<A11ySettings>) => {
    setSettings((prev) => {
      const next = { ...prev, ...patch };
      next.textStep = Math.min(Math.max(next.textStep, 0), TEXT_STEPS.length - 1);
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      } catch {
        /* storage unavailable */
      }
      applyToDocument(next);
      return next;
    });
  }, []);

  return (
    <SettingsContext.Provider value={{ settings, update }}>
      {children}
    </SettingsContext.Provider>
  );
}

export function useSettings() {
  return useContext(SettingsContext);
}
