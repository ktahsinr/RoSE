"use client";

/** The visible status band: current stage, elapsed time, progress. The
 *  same strings go to the screen reader through the app's single live
 *  region (via useAnnounce in the page), so this band is NOT aria-live. */

import { useEffect, useState } from "react";

const STAGE_NAMES = ["Queued", "Planning", "Voting", "Reasoning", "Done"];

export default function RunStatus({
  statusText,
  stage,
  running,
  startedAt,
  error,
}: {
  statusText: string;
  stage: number;
  running: boolean;
  startedAt: number | null;
  error: string | null;
}) {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    if (!running) return;
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [running]);

  if (!statusText && !running && !error) return null;

  const elapsed = startedAt ? Math.max(0, Math.round((now - startedAt) / 1000)) : 0;
  const mm = String(Math.floor(elapsed / 60)).padStart(2, "0");
  const ss = String(elapsed % 60).padStart(2, "0");

  return (
    <div
      className={`rounded-lg border-2 p-3 ${
        error ? "border-err bg-err-soft" : "border-accent bg-accent-soft"
      }`}
    >
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
        <p className="min-w-0 flex-1 font-semibold">
          {error ? `Error: ${error}` : statusText || "Working…"}
        </p>
        {running && startedAt && (
          <p className="font-bold tabular-nums" aria-hidden="true">
            {mm}:{ss}
          </p>
        )}
      </div>
      {!error && stage > 0 && (
        <div className="mt-2">
          <div
            role="progressbar"
            aria-label="Run progress"
            aria-valuemin={0}
            aria-valuemax={4}
            aria-valuenow={stage}
            aria-valuetext={`Stage ${stage} of 4: ${STAGE_NAMES[stage]}`}
            className="h-3 overflow-hidden rounded-full border border-accent bg-bg"
          >
            <div
              className={`h-full bg-accent ${running ? "pulse-soft" : ""}`}
              style={{ width: `${(stage / 4) * 100}%` }}
            />
          </div>
          <p className="mt-1 text-sm">
            Stage {stage} of 4 — {STAGE_NAMES[stage]}
            {running && " · live runs typically take 1–5 minutes"}
          </p>
        </div>
      )}
    </div>
  );
}
