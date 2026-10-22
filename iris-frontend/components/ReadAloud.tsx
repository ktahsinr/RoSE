"use client";

/** Read-aloud via the browser's Web Speech API (zero-cost: no cloud TTS).
 *  Every answer and chart description gets one, always with a stop control. */

import { useCallback, useEffect, useRef, useState } from "react";

export default function ReadAloud({
  text,
  label,
}: {
  text: string;
  label: string;
}) {
  const [speaking, setSpeaking] = useState(false);
  const [supported, setSupported] = useState(true);
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);

  useEffect(() => {
    setSupported(
      typeof window !== "undefined" && "speechSynthesis" in window,
    );
    return () => {
      if (typeof window !== "undefined" && "speechSynthesis" in window) {
        window.speechSynthesis.cancel();
      }
    };
  }, []);

  const start = useCallback(() => {
    const synth = window.speechSynthesis;
    synth.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.onend = () => setSpeaking(false);
    u.onerror = () => setSpeaking(false);
    utteranceRef.current = u;
    synth.speak(u);
    setSpeaking(true);
  }, [text]);

  const stop = useCallback(() => {
    window.speechSynthesis.cancel();
    setSpeaking(false);
  }, []);

  if (!supported) return null;

  return speaking ? (
    <button
      type="button"
      onClick={stop}
      aria-label={`Stop reading ${label}`}
      className="inline-flex items-center gap-1 rounded-full border border-err px-3 py-1 text-sm font-bold text-err hover:bg-err-soft"
    >
      <span aria-hidden="true">■</span> Stop
    </button>
  ) : (
    <button
      type="button"
      onClick={start}
      aria-label={`Read aloud: ${label}`}
      className="inline-flex items-center gap-1 rounded-full border border-accent px-3 py-1 text-sm font-bold text-accent hover:bg-accent-soft"
    >
      <span aria-hidden="true">🔊</span> Read aloud
    </button>
  );
}
