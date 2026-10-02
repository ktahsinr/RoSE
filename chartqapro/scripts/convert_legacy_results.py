# =============================================================
# Convert a LEGACY results file to the canonical sweep schema
# =============================================================
# The July notebook (RoSE_ChartQAPro_Notebook.ipynb) wrote rows as
#
#   {"Question": [...], "Answer": [...], "Question Type": "Factoid",
#    "prediction": "...", "_method": "zero_shot"|"rose",
#    "_uncertainty": f, "_complexity": f, "_pool_size": n, "_rationale": "..."}
#
# with no ids, no per-path logs and no latency. Everything written since
# (the Kaggle sweeps, routing/, summarize.py, export_csv.py) speaks the
# canonical schema instead. This one-shot converter bridges the gap:
#
#   id            derived from POSITION: factoid-00000, factoid-00001, …
#                 valid because _pool_size == position+1 in the legacy
#                 file (one pool add per question, in dataset order) and
#                 the sweep notebook numbers the same dataset order.
#   is_correct    recomputed with scoring.py — never trusted.
#   method        zero_shot → zero_shot_cot, rose → rose_few_shot.
#
# What CANNOT be recovered: raw_outputs/extracted/normalized (so no
# offline aggregation ablation for this run) and latency_s (so no 7B
# time-budget numbers from this file).
#
#   python scripts/convert_legacy_results.py checkpoint_factoid.json \
#          --qtype factoid --out results/rose_factoid.json
# =============================================================

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scoring import is_correct          # noqa: E402

_METHOD = {"zero_shot": "zero_shot_cot", "rose": "rose_few_shot"}


def _one(value):
    """Legacy fields are 1-element lists: ['2037-38'] -> '2037-38'."""
    if isinstance(value, (list, tuple)):
        return value[0] if len(value) == 1 else [str(v) for v in value]
    return value


def convert(rows: list, qtype: str) -> list:
    out = []
    for i, r in enumerate(rows):
        question = str(_one(r.get("Question", ""))).strip()
        gold = _one(r.get("Answer", ""))
        prediction = r.get("prediction", "")
        method = _METHOD.get(r.get("_method", ""), r.get("_method", "unknown"))
        out.append({
            "id":              f"{qtype}-{i:05d}",
            "question":        question,
            "question_type":   qtype,
            "choices":         None,
            "ground_truth":    gold,
            "prediction":      prediction,
            "prediction_norm": None,
            "is_correct":      is_correct(prediction, gold, qtype),
            "method":          method,
            "uncertainty":     abs(float(r.get("_uncertainty", 0.0) or 0.0)),
            "complexity":      float(r.get("_complexity", 1.0) or 1.0),
            "pool_size":       int(r.get("_pool_size", i + 1)),
            "best_rationale":  r.get("_rationale", ""),
            "orig_year_flag":  _one(r.get("Year")),
            "legacy":          True,   # marks rows without per-path logs
        })
    return out


def _main():
    ap = argparse.ArgumentParser(description="Legacy results → canonical sweep schema")
    ap.add_argument("legacy", help="legacy checkpoint/results json")
    ap.add_argument("--qtype", default="factoid", choices=["factoid", "mcq", "hypothetical"])
    ap.add_argument("--out", default="results/rose_factoid.json")
    args = ap.parse_args()

    rows = json.loads(Path(args.legacy).read_text())
    for i, r in enumerate(rows):
        if r.get("_pool_size") != i + 1:
            raise SystemExit(
                f"row {i}: _pool_size={r.get('_pool_size')} != {i + 1} — the file is "
                "not in dataset order, so position-derived ids would be wrong. "
                "Convert from the original, unsorted checkpoint.")

    out = convert(rows, args.qtype)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))

    n = len(out)
    acc = sum(r["is_correct"] for r in out) / max(n, 1) * 100
    by = {}
    for r in out:
        by.setdefault(r["method"], []).append(r["is_correct"])
    print(f"✓ {n} rows → {args.out}")
    print(f"  rescored accuracy (scoring.py): {acc:.1f}%")
    for m, hits in sorted(by.items()):
        print(f"    {m:<14} {sum(hits) / len(hits) * 100:5.1f}%  ({sum(hits)}/{len(hits)})")


if __name__ == "__main__":
    _main()
