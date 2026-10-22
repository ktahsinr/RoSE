"use client";

/** Screen-reader announcements: one polite live region for the whole app.
 *  Visible status text elsewhere mirrors these strings but is NOT itself a
 *  live region, so nothing is announced twice. */

import {
  createContext,
  useCallback,
  useContext,
  useRef,
  type ReactNode,
} from "react";

const AnnouncerContext = createContext<(message: string) => void>(() => {});

export function AnnouncerProvider({ children }: { children: ReactNode }) {
  const regionRef = useRef<HTMLDivElement>(null);
  const clearTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const announce = useCallback((message: string) => {
    const el = regionRef.current;
    if (!el) return;
    // Clearing first makes repeated identical messages re-announce.
    el.textContent = "";
    requestAnimationFrame(() => {
      el.textContent = message;
    });
    if (clearTimer.current) clearTimeout(clearTimer.current);
    clearTimer.current = setTimeout(() => {
      if (el.textContent === message) el.textContent = "";
    }, 10000);
  }, []);

  return (
    <AnnouncerContext.Provider value={announce}>
      {children}
      <div ref={regionRef} aria-live="polite" role="status" className="sr-only" />
    </AnnouncerContext.Provider>
  );
}

export function useAnnounce() {
  return useContext(AnnouncerContext);
}
