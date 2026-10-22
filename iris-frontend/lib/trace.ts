/** One trace model feeds the UI, whether the data came from a stored result
 *  (replay) or from live SSE events — the Explorer renders both the same. */

import type { StoredResult, VoteDetail } from "./api";

export interface PlannerStage {
  model: string;
  subQueries: string[];
  fallback: boolean;
  raw?: string;
  latencyS?: number;
}

export interface VoteStage extends VoteDetail {
  voteRule: string;
  chosen: string[];
  latencyS?: number;
}

export interface PathStage {
  index: number;
  raw: string;
  extracted: string;
}

export interface FinalStage {
  answer: string;
  normalized: string;
  groundTruth?: string | string[];
  isCorrect?: boolean;
  agreement: number | null;
  uncertainty: number | null;
  method: string;
  latencyS?: number | null;
  stageLatencyS?: Record<string, number>;
  paramsB?: number | null;
}

export interface TraceState {
  source: "stored" | "live" | null;
  planners: PlannerStage[];
  vote: VoteStage | null;
  chartDescription: string;
  paths: PathStage[];
  pathsExpected: number | null;
  final: FinalStage | null;
  error: string | null;
  statusText: string;
  /** 0 idle · 1 planning · 2 voting · 3 reasoning · 4 done */
  stage: number;
}

export const EMPTY_TRACE: TraceState = {
  source: null,
  planners: [],
  vote: null,
  chartDescription: "",
  paths: [],
  pathsExpected: null,
  final: null,
  error: null,
  statusText: "",
  stage: 0,
};

export function traceFromStored(r: StoredResult): TraceState {
  const fallback = r.decomp_fallback;
  const fallbackFor = (model: string): boolean => {
    if (typeof fallback === "object" && fallback !== null)
      return Boolean((fallback as Record<string, boolean>)[model]);
    return Boolean(fallback);
  };
  const planners: PlannerStage[] = Object.entries(r.proposals ?? {}).map(
    ([model, subQueries]) => ({
      model,
      subQueries,
      fallback: fallbackFor(model),
      latencyS: r.stage_latency_s?.[`decompose_${model}`],
    }),
  );
  return {
    source: "stored",
    planners,
    vote: {
      ...r.vote,
      voteRule: r.vote_rule,
      chosen: r.sub_queries ?? [],
      latencyS: r.stage_latency_s?.choose,
    },
    chartDescription: r.chart_description ?? "",
    paths: (r.paths ?? []).map((p) => ({
      index: p.index,
      raw: p.raw,
      extracted: p.extracted,
    })),
    pathsExpected: r.paths?.length ?? null,
    final: {
      answer: r.final_answer,
      normalized: r.prediction_norm,
      groundTruth: r.ground_truth,
      isCorrect: r.is_correct,
      agreement: r.agreement,
      uncertainty: r.uncertainty,
      method: r.method,
      latencyS: r.latency_s,
      stageLatencyS: r.stage_latency_s,
      paramsB: r.params_b_used,
    },
    error: r.error || null,
    statusText: "Showing the stored result.",
    stage: 4,
  };
}

export interface SseEvent {
  event: string;
  data: Record<string, unknown>;
}

/** Fold one SSE event into the trace. Pure, so React state updates stay
 *  predictable and the same reducer can replay a snapshot. */
export function reduceEvent(trace: TraceState, e: SseEvent): TraceState {
  const d = e.data as Record<string, never> & Record<string, unknown>;
  const announce = typeof d.announce === "string" ? d.announce : "";
  const next: TraceState = {
    ...trace,
    source: "live",
    statusText: announce || trace.statusText,
  };
  switch (e.event) {
    case "queued":
      return { ...next, stage: 0 };
    case "started":
      return { ...next, stage: 1 };
    case "stage.decompose":
      return {
        ...next,
        stage: 1,
        planners: [
          ...trace.planners,
          {
            model: String(d.model ?? ""),
            subQueries: (d.sub_queries as string[]) ?? [],
            fallback: Boolean(d.fallback),
            raw: typeof d.raw === "string" ? d.raw : undefined,
            latencyS: d.latency_s as number | undefined,
          },
        ],
      };
    case "stage.vote":
      return {
        ...next,
        stage: 2,
        vote: {
          rule: d.rule as string,
          picked_model: d.picked_model as string | null,
          matching: d.matching as string,
          pair_sims: d.pair_sims as number[],
          mean_sim_to_question: d.mean_sim_to_question as Record<string, number>,
          threshold: d.threshold as number,
          voteRule: String(d.vote_rule ?? d.rule ?? ""),
          chosen: (d.chosen as string[]) ?? [],
          latencyS: d.latency_s as number | undefined,
        },
      };
    case "stage.describe":
      return { ...next, stage: 3, chartDescription: String(d.description ?? "") };
    case "stage.path":
      return {
        ...next,
        stage: 3,
        pathsExpected: (d.total as number) ?? trace.pathsExpected,
        paths: [
          ...trace.paths,
          {
            index: Number(d.index ?? trace.paths.length),
            raw: String(d.raw ?? ""),
            extracted: String(d.extracted ?? ""),
          },
        ],
      };
    case "stage.final":
      return {
        ...next,
        stage: 4,
        final: {
          answer: String(d.answer ?? ""),
          normalized: String(d.prediction_norm ?? ""),
          groundTruth: d.ground_truth as string | string[] | undefined,
          isCorrect: d.is_correct as boolean | undefined,
          agreement: (d.agreement as number) ?? null,
          uncertainty: (d.uncertainty as number) ?? null,
          method: String(d.method ?? ""),
          latencyS: (d.latency_s as number) ?? null,
          stageLatencyS: d.stage_latency_s as Record<string, number>,
          paramsB: (d.params_b_used as number) ?? null,
        },
      };
    case "error":
      return { ...next, error: String(d.message ?? "Unknown error"), stage: 4 };
    case "done":
      return { ...next, stage: 4 };
    default:
      return next;
  }
}

export function agreementLabel(agreement: number | null): {
  text: string;
  tone: "ok" | "mid" | "low";
} {
  if (agreement === null || Number.isNaN(agreement))
    return { text: "agreement unknown", tone: "low" };
  const pct = Math.round(agreement * 100);
  if (agreement >= 0.8) return { text: `high confidence (${pct}% of paths agree)`, tone: "ok" };
  if (agreement >= 0.5) return { text: `medium confidence (${pct}% of paths agree)`, tone: "mid" };
  return { text: `low confidence (${pct}% of paths agree)`, tone: "low" };
}
