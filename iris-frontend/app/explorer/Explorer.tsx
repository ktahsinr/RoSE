"use client";

/** The main screen. Left: question picker, model selectors, run controls and
 *  the live reasoning trace. Right: the chart viewer with the chart
 *  description. Narrow screens stack with the chart first (the chart pane
 *  comes first in the DOM; on wide screens the grid places it right). */

import { useCallback, useEffect, useRef, useState } from "react";
import {
  apiUrl,
  getJSON,
  postJSON,
  ApiError,
  type Health,
  type ModelInfo,
  type QuestionDetail,
  type QuestionItem,
  type RunStarted,
} from "@/lib/api";
import {
  EMPTY_TRACE,
  reduceEvent,
  traceFromStored,
  type TraceState,
} from "@/lib/trace";
import { useAnnounce } from "@/lib/announcer";
import AnswerCard from "@/components/AnswerCard";
import ChartViewer from "@/components/ChartViewer";
import ModelPanel, { type ModelChoice } from "@/components/ModelPanel";
import QuestionPicker from "@/components/QuestionPicker";
import RunStatus from "@/components/RunStatus";
import Trace from "@/components/Trace";

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

export default function Explorer() {
  const announce = useAnnounce();

  const [health, setHealth] = useState<Health | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [items, setItems] = useState<QuestionItem[]>([]);
  const [total, setTotal] = useState(0);
  const [search, setSearch] = useState("");
  const [typeFilter, setTypeFilter] = useState("");

  const [detail, setDetail] = useState<QuestionDetail | null>(null);
  const [choice, setChoice] = useState<ModelChoice>({
    planners: ["gemma3-4b", "qwen3.5-4b"],
    finalModel: "qwen2.5-vl-7b",
    fastMode: true,
  });

  const [trace, setTrace] = useState<TraceState>(EMPTY_TRACE);
  const [running, setRunning] = useState(false);
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const sourceRef = useRef<EventSource | null>(null);

  // ── bootstrap ─────────────────────────────────────────────
  useEffect(() => {
    Promise.all([getJSON<Health>("/health"), getJSON<ModelInfo[]>("/models")])
      .then(([h, m]) => {
        setHealth(h);
        setModels(m);
        if (h.mode === "demo")
          announce("Live runs are disabled — no GPU. Stored results are available.");
      })
      .catch((e: ApiError) => setLoadError(e.message));
  }, [announce]);

  // ── question list (debounced search) ──────────────────────
  useEffect(() => {
    const t = setTimeout(() => {
      const params = new URLSearchParams({ limit: "100" });
      if (search) params.set("search", search);
      if (typeFilter) params.set("type", typeFilter);
      getJSON<{ total: number; items: QuestionItem[] }>(
        `/questions?${params.toString()}`,
      )
        .then((out) => {
          setItems(out.items);
          setTotal(out.total);
        })
        .catch((e: ApiError) => setLoadError(e.message));
    }, 250);
    return () => clearTimeout(t);
  }, [search, typeFilter]);

  const stopStream = useCallback(() => {
    sourceRef.current?.close();
    sourceRef.current = null;
    setRunning(false);
  }, []);

  useEffect(() => stopStream, [stopStream]);

  // ── selection ─────────────────────────────────────────────
  const selectQuestion = useCallback(
    (q: QuestionItem) => {
      stopStream();
      setTrace(EMPTY_TRACE);
      getJSON<QuestionDetail>(`/questions/${q.id}`)
        .then((d) => {
          setDetail(d);
          announce(`Selected ${d.id}: ${d.question}`);
        })
        .catch((e: ApiError) => setLoadError(e.message));
    },
    [announce, stopStream],
  );

  // ── replay ────────────────────────────────────────────────
  const showStored = useCallback(() => {
    const stored = detail?.replay?.alg2;
    if (!stored) return;
    stopStream();
    const t = traceFromStored(stored);
    setTrace(t);
    setStartedAt(null);
    announce(
      `Showing the stored result. Final answer: ${stored.final_answer}.` +
        (stored.is_correct ? " It matches the expected answer." : ""),
    );
  }, [announce, detail, stopStream]);

  // ── live run ──────────────────────────────────────────────
  const runLive = useCallback(async () => {
    if (!detail || running || choice.planners.length === 0) return;
    stopStream();
    setTrace({ ...EMPTY_TRACE, source: "live", statusText: "Starting run…" });
    setStartedAt(Date.now());
    setRunning(true);
    announce("Starting a live run.");
    try {
      const started = await postJSON<RunStarted>("/run", {
        question_id: detail.id,
        planners: choice.planners,
        final_model: choice.finalModel,
        fast_mode: choice.fastMode,
      });
      const es = new EventSource(apiUrl(`/run/${started.run_id}/events`));
      sourceRef.current = es;
      for (const name of SSE_EVENTS) {
        es.addEventListener(name, (ev) => {
          const data = JSON.parse((ev as MessageEvent).data);
          if (typeof data.announce === "string") announce(data.announce);
          setTrace((prev) => reduceEvent(prev, { event: name, data }));
          if (name === "done" || name === "error") {
            es.close();
            sourceRef.current = null;
            setRunning(false);
          }
        });
      }
      es.onerror = () => {
        // EventSource retries by itself; only report when it gave up closed.
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
  }, [announce, choice, detail, running, stopStream]);

  // ── derived ───────────────────────────────────────────────
  const demoMode = health?.mode === "demo";
  const storedAvailable = Boolean(detail?.replay?.alg2);
  const description =
    trace.chartDescription || detail?.replay?.alg2?.chart_description || "";
  const types = Array.from(new Set(items.map((i) => i.question_type)));

  return (
    <div className="flex flex-col gap-4">
      <h1 className="text-2xl font-bold">ChartQAPro Explorer</h1>

      {loadError && (
        <p
          role="alert"
          className="rounded-lg border-2 border-err bg-err-soft p-3 font-semibold text-err"
        >
          {loadError}
        </p>
      )}
      {demoMode && (
        <p className="rounded-lg border-2 border-amber bg-amber-soft p-3 font-semibold text-amber">
          Demo mode: no GPU is available, so live runs are disabled. Stored
          results replay instantly.
          {health?.live_load_error && ` (Model loading failed: ${health.live_load_error})`}
        </p>
      )}

      <RunStatus
        statusText={trace.statusText}
        stage={trace.stage}
        running={running}
        startedAt={startedAt}
        error={trace.error}
      />

      <div className="grid gap-6 lg:grid-cols-[minmax(0,5fr)_minmax(0,4fr)]">
        {/* Chart pane — first in the DOM so narrow screens show it first;
            on wide screens the grid places it in the right column. */}
        <div className="min-w-0 lg:order-2 lg:sticky lg:top-4 lg:self-start">
          <ChartViewer
            src={detail?.image_url ? apiUrl(detail.image_url) : null}
            description={description}
            title={
              detail ? `Chart for question ${detail.id}` : "No chart selected"
            }
          />
        </div>

        {/* Work pane */}
        <div className="flex min-w-0 flex-col gap-5 lg:order-1">
          <QuestionPicker
            items={items}
            total={total}
            selectedId={detail?.id ?? null}
            onSelect={selectQuestion}
            search={search}
            onSearch={setSearch}
            typeFilter={typeFilter}
            onTypeFilter={setTypeFilter}
            types={types}
          />

          <ModelPanel
            models={models}
            choice={choice}
            onChange={setChoice}
            disabled={demoMode || running}
          />

          <div className="flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={runLive}
              disabled={
                !detail || demoMode || running || choice.planners.length === 0
              }
              className="rounded-lg border-2 border-amber bg-amber-soft px-4 py-2 font-bold text-amber hover:opacity-90 disabled:opacity-40"
            >
              ▶ Run live
            </button>
            <button
              type="button"
              onClick={showStored}
              disabled={!storedAvailable}
              className="rounded-lg border-2 border-accent bg-accent-soft px-4 py-2 font-bold text-accent hover:opacity-90 disabled:opacity-40"
            >
              ⚡ Show stored result
            </button>
            {!detail && <span className="text-muted">Pick a question first.</span>}
            {detail && !storedAvailable && (
              <span className="text-muted">No stored result for this question.</span>
            )}
          </div>

          {trace.final && <AnswerCard final={trace.final} />}

          <Trace trace={trace} />
        </div>
      </div>
    </div>
  );
}
