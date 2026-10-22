"use client";

/** The reasoning trace: planner proposals, the vote with its similarities,
 *  the chart description, each reasoning path, and the final vote — as
 *  disclosure sections that fill in live. */

import { useState, type ReactNode } from "react";
import type { TraceState } from "@/lib/trace";

function Stage({
  title,
  meta,
  children,
  defaultOpen = true,
}: {
  title: string;
  meta?: string;
  children: ReactNode;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className="rounded-lg border-2 border-line">
      <h3>
        <button
          type="button"
          aria-expanded={open}
          onClick={() => setOpen((v) => !v)}
          className="flex w-full items-center justify-between gap-2 rounded-t-lg bg-surface px-3 py-2 text-left font-bold hover:bg-surface2"
        >
          <span>
            <span aria-hidden="true">{open ? "▾" : "▸"} </span>
            {title}
          </span>
          {meta && <span className="text-sm font-normal text-muted">{meta}</span>}
        </button>
      </h3>
      {open && <div className="flex flex-col gap-3 p-3">{children}</div>}
    </section>
  );
}

function SubQueryList({ items }: { items: string[] }) {
  return (
    <ol className="list-decimal pl-6">
      {items.map((s, i) => (
        <li key={i}>{s}</li>
      ))}
    </ol>
  );
}

const MODEL_NAMES: Record<string, string> = {
  "gemma3-4b": "Gemma 3 4B",
  "qwen3.5-4b": "Qwen3.5 4B",
  "qwen2.5-vl-7b": "Qwen2.5-VL 7B",
};

export function modelName(key: string): string {
  return MODEL_NAMES[key] ?? key;
}

export default function Trace({ trace }: { trace: TraceState }) {
  if (trace.stage === 0 && trace.planners.length === 0) return null;

  return (
    <section aria-label="Reasoning trace" className="flex flex-col gap-3">
      <h2 className="text-lg font-bold">
        Reasoning trace
        {trace.source === "stored" && (
          <span className="ml-2 rounded-full bg-amber-soft px-2 py-0.5 text-sm font-semibold text-amber">
            stored result
          </span>
        )}
      </h2>

      {trace.planners.map((p) => (
        <Stage
          key={p.model}
          title={`Plan — ${modelName(p.model)}`}
          meta={p.latencyS != null ? `${p.latencyS}s` : undefined}
        >
          {p.fallback ? (
            <p className="rounded bg-err-soft p-2 font-semibold text-err">
              This model&apos;s output could not be parsed as two sub-queries;
              its proposal fell back to the original question and is excluded
              from the vote.
            </p>
          ) : (
            <SubQueryList items={p.subQueries} />
          )}
          {p.raw && (
            <details>
              <summary className="cursor-pointer font-semibold">
                Raw model output
              </summary>
              <pre className="mt-1 overflow-x-auto whitespace-pre-wrap rounded bg-surface2 p-2 text-sm">
                {p.raw}
              </pre>
            </details>
          )}
        </Stage>
      ))}

      {trace.vote && (
        <Stage
          title="Vote — choosing the plan"
          meta={trace.vote.latencyS != null ? `${trace.vote.latencyS}s` : undefined}
        >
          <p>
            <strong>Rule fired:</strong> <code>{trace.vote.voteRule}</code>
            {trace.vote.rule === "merge" &&
              " — the proposals agree, so they were merged."}
            {trace.vote.rule === "pick" &&
              trace.vote.picked_model &&
              ` — the proposals disagree; ${modelName(
                trace.vote.picked_model,
              )}'s plan sits closer to the question.`}
            {trace.vote.voteRule.startsWith("single:") &&
              " — only one usable proposal (the other fell back)."}
            {trace.vote.voteRule === "fallback" &&
              " — both proposals fell back; IRIS continues with the original question."}
          </p>
          {trace.vote.pair_sims && trace.vote.pair_sims.length > 0 && (
            <p>
              Pair similarities ({trace.vote.matching} matching):{" "}
              {trace.vote.pair_sims.map((s) => s.toFixed(2)).join(", ")}{" "}
              <span className="text-muted">
                (merge needs both ≥ {trace.vote.threshold})
              </span>
            </p>
          )}
          {trace.vote.mean_sim_to_question && (
            <p>
              Mean similarity to the question:{" "}
              {Object.entries(trace.vote.mean_sim_to_question)
                .map(([m, v]) => `${modelName(m)} ${v.toFixed(2)}`)
                .join(" · ")}
            </p>
          )}
          <div>
            <h4 className="font-semibold">Chosen steps</h4>
            <SubQueryList items={trace.vote.chosen} />
          </div>
        </Stage>
      )}

      {trace.chartDescription && (
        <Stage title="RoSE — chart description">
          <p>{trace.chartDescription}</p>
        </Stage>
      )}

      {trace.paths.length > 0 && (
        <Stage
          title={`RoSE — reasoning paths (${trace.paths.length}${
            trace.pathsExpected ? ` of ${trace.pathsExpected}` : ""
          })`}
          defaultOpen={false}
        >
          {trace.paths.map((p) => (
            <article key={p.index} className="rounded border border-line p-2">
              <h4 className="font-semibold">
                Path {p.index + 1}
                {p.extracted && (
                  <span className="ml-2 font-normal text-muted">
                    → {p.extracted}
                  </span>
                )}
              </h4>
              <pre className="mt-1 overflow-x-auto whitespace-pre-wrap text-sm">
                {p.raw}
              </pre>
            </article>
          ))}
        </Stage>
      )}

      {trace.final && (
        <Stage title="Final — RoSE's weighted vote">
          <p>
            Method: <code>{trace.final.method || "—"}</code>
            {trace.final.uncertainty != null && (
              <> · uncertainty {trace.final.uncertainty.toFixed(2)}</>
            )}
            {trace.final.paramsB != null && (
              <> · {trace.final.paramsB}B parameters engaged</>
            )}
            {trace.final.latencyS != null && <> · {trace.final.latencyS}s total</>}
          </p>
          {trace.final.stageLatencyS &&
            Object.keys(trace.final.stageLatencyS).length > 0 && (
              <p className="text-sm text-muted">
                Per stage:{" "}
                {Object.entries(trace.final.stageLatencyS)
                  .map(([k, v]) => `${k} ${v}s`)
                  .join(" · ")}
              </p>
            )}
        </Stage>
      )}
    </section>
  );
}
