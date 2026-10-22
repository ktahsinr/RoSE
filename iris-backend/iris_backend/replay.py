"""Stored run results (decomp_alg2 / decomp_alg1) for replay mode.

The run JSON (written by decompose.decomp_runner) is preferred because only
it carries the full RoSE paths and the chart description; the per-question
CSV is the fallback with the same per-question summary fields. Either file
may be absent — replay is then simply unavailable for that algorithm.
"""

import csv
import json
from pathlib import Path


def _loads(value, default):
    try:
        return json.loads(value) if value else default
    except (json.JSONDecodeError, TypeError):
        return default


def _from_json_row(r: dict) -> dict:
    raw = r.get("raw_outputs") or []
    extracted = r.get("extracted") or []
    normalized = r.get("normalized") or []
    paths = [{
        "index": i,
        "raw": raw[i] if i < len(raw) else "",
        "extracted": extracted[i] if i < len(extracted) else "",
        "normalized": normalized[i] if i < len(normalized) else "",
    } for i in range(len(raw))]
    return {
        "id": str(r.get("id", "")),
        "algorithm": r.get("algorithm"),
        "question": r.get("question", ""),
        "ground_truth": r.get("ground_truth", ""),
        "sub_queries": r.get("sub_queries", []),
        "sub_answers": r.get("sub_answers", []),
        "sub_confidences": r.get("sub_confidences", []),
        "sub_models": r.get("sub_models", []),
        "proposals": r.get("proposals", {}),
        "decomp_fallback": r.get("decomp_fallback", {}),
        "vote_rule": r.get("vote_rule", ""),
        "vote": r.get("vote", {}),
        "chart_description": r.get("chart_description", ""),
        "paths": paths,
        "agreement": r.get("agreement"),
        "uncertainty": r.get("uncertainty"),
        "method": r.get("method", ""),
        "final_answer": r.get("prediction", ""),
        "prediction_norm": r.get("prediction_norm", ""),
        "is_correct": bool(r.get("is_correct", False)),
        "n_demos": r.get("n_demos"),
        "pool_size": r.get("pool_size"),
        "latency_s": r.get("latency_s"),
        "stage_latency_s": r.get("stage_latency_s", {}),
        "params_b_used": r.get("params_b_used"),
        "error": r.get("error", ""),
        "source": "json",
    }


def _from_csv_row(r: dict) -> dict:
    return {
        "id": str(r.get("id", "")),
        "algorithm": int(r["algorithm"]) if r.get("algorithm") else None,
        "question": r.get("question", ""),
        "ground_truth": r.get("ground_truth", ""),
        "sub_queries": _loads(r.get("sub_queries"), []),
        "sub_answers": _loads(r.get("sub_answers"), []),
        "sub_confidences": _loads(r.get("sub_confidences"), []),
        "sub_models": _loads(r.get("sub_models"), []),
        "proposals": {},                      # JSON-only detail
        "decomp_fallback": _loads(r.get("decomp_fallback"), {}),
        "vote_rule": r.get("vote_rule", ""),
        "vote": {},                           # JSON-only detail
        "chart_description": "",              # JSON-only detail
        "paths": [],                          # JSON-only detail
        "agreement": float(r["agreement"]) if r.get("agreement") else None,
        "uncertainty": float(r["uncertainty"]) if r.get("uncertainty") else None,
        "method": r.get("method", ""),
        "final_answer": r.get("final_answer", ""),
        "prediction_norm": r.get("prediction_norm", ""),
        "is_correct": str(r.get("is_correct", "")).strip().lower() == "true",
        "n_demos": int(r["n_demos"]) if r.get("n_demos") else None,
        "pool_size": int(r["pool_size"]) if r.get("pool_size") else None,
        "latency_s": float(r["latency_s"]) if r.get("latency_s") else None,
        "stage_latency_s": _loads(r.get("stage_latency_s"), {}),
        "params_b_used": float(r["params_b_used"]) if r.get("params_b_used") else None,
        "error": r.get("error", ""),
        "source": "csv",
    }


class ReplayStore:
    """Indexes stored results by algorithm and question id."""

    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.runs = {}        # {algorithm: {id: normalized result}}
        self.meta = {}        # {algorithm: meta json or {}}
        for alg in (1, 2):
            loaded = self._load_alg(alg)
            if loaded:
                self.runs[alg] = loaded
                meta_path = self.results_dir / f"decomp_alg{alg}_meta.json"
                self.meta[alg] = (_loads(meta_path.read_text(), {})
                                  if meta_path.exists() else {})

    def _load_alg(self, alg: int) -> dict:
        json_path = self.results_dir / f"decomp_alg{alg}.json"
        csv_path = self.results_dir / f"decomp_alg{alg}.csv"
        if json_path.exists():
            rows = json.loads(json_path.read_text())
            return {str(r.get("id", "")): _from_json_row(r) for r in rows}
        if csv_path.exists():
            with csv_path.open(newline="", encoding="utf-8") as f:
                return {str(r.get("id", "")): _from_csv_row(r)
                        for r in csv.DictReader(f)}
        return {}

    def algorithms(self) -> list:
        return sorted(self.runs)

    def has(self, question_id: str) -> dict:
        return {f"alg{a}": question_id in self.runs[a] for a in self.runs}

    def get(self, question_id: str, algorithm: int = 2) -> "dict | None":
        return self.runs.get(algorithm, {}).get(question_id)
