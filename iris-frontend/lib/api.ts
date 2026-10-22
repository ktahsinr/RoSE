/** The IRIS backend is the only origin the frontend ever talks to
 *  (zero-cost constraint): everything goes through apiUrl(). */

export const API_URL = (
  process.env.NEXT_PUBLIC_IRIS_API_URL ?? "http://localhost:8000"
).replace(/\/$/, "");

export function apiUrl(path: string): string {
  return `${API_URL}${path}`;
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(apiUrl(path), init);
  } catch {
    throw new ApiError(
      0,
      `Could not reach the IRIS backend at ${API_URL}. Is it running?`,
    );
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export function getJSON<T>(path: string): Promise<T> {
  return request<T>(path);
}

export function postJSON<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function postForm<T>(path: string, form: FormData): Promise<T> {
  return request<T>(path, { method: "POST", body: form });
}

/* ── API types ──────────────────────────────────────────────── */

export interface Health {
  mode: "demo" | "live";
  live_runs_enabled: boolean;
  live_load_error: string | null;
  models_loaded: string[];
  pool_size: number | null;
  dataset: { path: string; count: number };
  replay_algorithms: number[];
  queue_length: number;
}

export interface ModelInfo {
  key: string;
  hf_id: string;
  label: string;
  params_b: number;
  role: "planner" | "final" | "embedder";
  default_final: boolean;
  loaded: boolean;
  revision: string | null;
}

export interface QuestionItem {
  id: string;
  question: string;
  question_type: string;
  ground_truth: string | string[];
  image_url: string | null;
  has_replay: Record<string, boolean>;
}

export interface VoteDetail {
  rule?: string;
  picked_model?: string | null;
  matching?: string;
  pair_sims?: number[];
  mean_sim_to_question?: Record<string, number>;
  threshold?: number;
}

export interface ReasoningPath {
  index: number;
  raw: string;
  extracted: string;
  normalized: string;
}

export interface StoredResult {
  id: string;
  algorithm: number | null;
  question: string;
  ground_truth: string | string[];
  sub_queries: string[];
  proposals: Record<string, string[]>;
  decomp_fallback: Record<string, boolean> | boolean | string;
  vote_rule: string;
  vote: VoteDetail;
  chart_description: string;
  paths: ReasoningPath[];
  agreement: number | null;
  uncertainty: number | null;
  method: string;
  final_answer: string;
  prediction_norm: string;
  is_correct: boolean;
  n_demos: number | null;
  pool_size: number | null;
  latency_s: number | null;
  stage_latency_s: Record<string, number>;
  params_b_used: number | null;
  error: string;
  source: "json" | "csv";
}

export interface QuestionDetail extends QuestionItem {
  replay: Record<string, StoredResult | null>;
}

export interface BenchmarksPayload {
  published: {
    citation: string;
    columns: string[];
    rows: { model: string; group: string; scores: number[]; best: boolean[] }[];
  };
  iris: {
    title: string;
    rows: {
      key: string;
      label: string;
      accuracy_pct: number;
      correct: number;
      n: number;
      ci: [number, number];
      settings: string;
    }[];
    footnote: string;
  };
}

export interface RunStarted {
  run_id: string;
  queue_position: number;
}
