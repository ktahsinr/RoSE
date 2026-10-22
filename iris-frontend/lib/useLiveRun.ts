"use client";

/** Shared live-run plumbing: POST /run, follow the SSE stream, fold events
 *  into a TraceState, announce every stage. Used by the Explorer and the
 *  PDF Chart QA page. */

import { useCallback, useEffect, useRef, useState } from "react";
import { apiUrl, postJSON, ApiError, type RunStarted } from "./api";
import { EMPTY_TRACE, reduceEvent, type TraceState } from "./trace";
import { useAnnounce } from "./announcer";

const SSE_EVENTS = [
  "queued",
  "started",
  "stage.decompose",
  "stage.vote",
  "stage.describe",
  "stage.path",
  "stage.final",
  "error",
  "done",
];

export interface RunBody {
  question_id?: string;
  image_id?: string;
  question?: string;
  planners: string[];
  final_model: string;
  fast_mode: boolean;
}

export function useLiveRun() {
  const announce = useAnnounce();
  const [trace, setTrace] = useState<TraceState>(EMPTY_TRACE);
  const [running, setRunning] = useState(false);
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const sourceRef = useRef<EventSource | null>(null);

  const stop = useCallback(() => {
    sourceRef.current?.close();
    sourceRef.current = null;
    setRunning(false);
  }, []);

  useEffect(() => stop, [stop]);

  const start = useCallback(
    async (body: RunBody, onFinish?: (trace: TraceState) => void) => {
      stop();
      setTrace({ ...EMPTY_TRACE, source: "live", statusText: "Starting run…" });
      setStartedAt(Date.now());
      setRunning(true);
      announce("Starting a live run.");
      try {
        const started = await postJSON<RunStarted>("/run", body);
        const es = new EventSource(apiUrl(`/run/${started.run_id}/events`));
        sourceRef.current = es;
        for (const name of SSE_EVENTS) {
          es.addEventListener(name, (ev) => {
            const data = JSON.parse((ev as MessageEvent).data);
            if (typeof data.announce === "string") announce(data.announce);
            setTrace((prev) => {
              const next = reduceEvent(prev, { event: name, data });
              if (name === "done" || name === "error") onFinish?.(next);
              return next;
            });
            if (name === "done" || name === "error") {
              es.close();
              sourceRef.current = null;
              setRunning(false);
            }
          });
        }
        es.onerror = () => {
          if (es.readyState === EventSource.CLOSED) {
            setTrace((prev) => ({
              ...prev,
              error: "Lost the connection to the backend during the run.",
              stage: 4,
            }));
            announce("Lost the connection to the backend during the run.");
            setRunning(false);
          }
        };
      } catch (e) {
        const msg = e instanceof ApiError ? e.message : String(e);
        setTrace((prev) => ({ ...prev, error: msg, stage: 4 }));
        announce(`The run could not start: ${msg}`);
        setRunning(false);
      }
    },
    [announce, stop],
  );

  return { trace, setTrace, running, startedAt, setStartedAt, start, stop };
}
