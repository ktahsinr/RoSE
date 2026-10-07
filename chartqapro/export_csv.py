# =============================================================
# Export a results JSON to CSV
# =============================================================
# CPU-only, stdlib-only. Re-scores each row with scoring.py as it goes,
# so the CSV carries the current normalized verdict rather than whatever
# flag happened to be written during the run.
#
#   python export_csv.py results/factoid_rose_results.json
#   python export_csv.py results/factoid_rose_results.json --per-path
#   python export_csv.py results/factoid_rose_results.json \
#       --verdicts results/verdicts.json          # merge judge verdicts
#
# The raw generations are deliberately NOT in the main CSV — they are
# multi-line and would make it unreadable. They stay in the JSON, and
# --per-path writes a second CSV with one row per reasoning path.
# =============================================================

import argparse
import csv
import json
import sys
from pathlib import Path

try:
    from .scoring import is_correct, normalize_answer, is_unanswerable
except ImportError:
    from scoring import is_correct, normalize_answer, is_unanswerable


COLUMNS = [
    "id", "question_type", "model_key", "question", "ground_truth",
    "prediction", "prediction_norm",
    "is_correct_run", "is_correct_rescored", "judge_correct", "judge_category",
    "gold_unanswerable", "pred_unanswerable",
    "method", "uncertainty", "complexity", "agreement",
    "vote_mode", "n_groups", "paths_voting", "n_demos", "pool_size",
    "latency_s", "routing_scores",
    "extracted_paths", "normalized_paths", "error",
]


def load(path):
    p = Path(path)
    if not p.exists():
        sys.exit(f"error: no such file: {path}")
    with p.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("results") or list(data.values())
    return [r for r in data if isinstance(r, dict)]


def build_row(r, verdicts):
    qtype = str(r.get("question_type", "factoid")).lower().strip()
    gold = r.get("ground_truth", "")
    pred = r.get("prediction", "")
    opts = r.get("options_shown") or r.get("choices")
    is_err = "error" in r or str(pred).strip() == "ERROR"

    rescored = False if is_err else is_correct(
        pred, gold, qtype, choices=opts if qtype == "mcq" else None)

    v = verdicts.get(str(r.get("id", "")), {})

    return {
        "id":                  r.get("id", ""),
        "question_type":       qtype,
        "question":            r.get("question", ""),
        "ground_truth":        gold if not isinstance(gold, (list, tuple))
                               else "|".join(map(str, gold)),
        "prediction":          pred,
        "prediction_norm":     r.get("prediction_norm")
                               or normalize_answer(pred, qtype),
        "is_correct_run":      r.get("is_correct", ""),
        "is_correct_rescored": rescored,
        "judge_correct":       v.get("correct", ""),
        "judge_category":      v.get("category", ""),
        "gold_unanswerable":   is_unanswerable(gold),
        "pred_unanswerable":   is_unanswerable(pred),
        "method":              r.get("method", ""),
        "uncertainty":         r.get("uncertainty", ""),
        "complexity":          r.get("complexity", ""),
        "agreement":           r.get("agreement", ""),
        "vote_mode":           r.get("vote_mode", ""),
        "n_groups":            r.get("n_groups", ""),
        "paths_voting":        r.get("paths_voting", ""),
        "n_demos":             r.get("n_demos", ""),
        "pool_size":           r.get("pool_size", ""),
        "model_key":           r.get("model_key", ""),
        "latency_s":           r.get("latency_s", ""),
        "routing_scores":      json.dumps((r.get("routing") or {}).get("scores"))
                               if r.get("routing") else "",
        "extracted_paths":     " | ".join(map(str, r.get("extracted") or [])),
        "normalized_paths":    " | ".join(map(str, r.get("normalized") or [])),
        "error":               r.get("error", ""),
    }


def write_per_path(rows, out_path):
    """One row per reasoning path — for inspecting self-consistency."""
    fields = ["id", "question_type", "path", "temperature_note",
              "well_formed", "extracted", "normalized", "raw_output"]
    n = 0
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            extracted = r.get("extracted") or []
            normalized = r.get("normalized") or []
            raws = r.get("raw_outputs") or []
            for i in range(max(len(extracted), len(raws))):
                raw = raws[i] if i < len(raws) else ""
                w.writerow({
                    "id": r.get("id", ""),
                    "question_type": r.get("question_type", ""),
                    "path": i,
                    "temperature_note": "greedy" if i == 0 else "sampled",
                    "well_formed": "final answer" in str(raw).lower(),
                    "extracted": extracted[i] if i < len(extracted) else "",
                    "normalized": normalized[i] if i < len(normalized) else "",
                    "raw_output": raw,
                })
                n += 1
    return n


def write_summary(rows, verdicts, out_path):
    """
    One accuracy row per answering model plus an overall row — the
    routed-run report as a CSV. Accuracy is the RESCORED verdict.
    """
    scored = [(r, build_row(r, verdicts)) for r in rows]
    valid = [(r, b) for r, b in scored
             if "error" not in r and str(r.get("prediction", "")).strip() != "ERROR"]
    models = sorted({b["model_key"] for _, b in valid if b["model_key"]})

    def agg(sub, label):
        n_ok = sum(b["is_correct_rescored"] for _, b in sub)
        lats = [r["latency_s"] for r, _ in sub
                if isinstance(r.get("latency_s"), (int, float))]
        return {
            "model":            label,
            "n_answered":       len(sub),
            "n_correct":        n_ok,
            "accuracy_pct":     round(n_ok / max(len(sub), 1) * 100, 2),
            "share_pct":        round(len(sub) / max(len(valid), 1) * 100, 2),
            "mean_latency_s":   round(sum(lats) / len(lats), 2) if lats else "",
        }

    out_rows = [agg([(r, b) for r, b in valid if b["model_key"] == m], m)
                for m in models]
    out_rows.append(agg(valid, "OVERALL"))

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    return out_rows


def main():
    ap = argparse.ArgumentParser(description="Export results JSON to CSV.")
    ap.add_argument("results")
    ap.add_argument("-o", "--out", help="output CSV (default: alongside input)")
    ap.add_argument("--per-path", action="store_true",
                    help="also write a second CSV with one row per path")
    ap.add_argument("--summary", action="store_true",
                    help="also write <out>_summary.csv: accuracy per model + overall")
    ap.add_argument("--verdicts",
                    help="judge verdicts JSON from rescore_judge.py --out")
    args = ap.parse_args()

    rows = load(args.results)

    verdicts = {}
    if args.verdicts:
        for r in load(args.verdicts):
            if r.get("verdict"):
                verdicts[str(r.get("id", ""))] = r["verdict"]
        print(f"  merged {len(verdicts)} judge verdicts")

    out = Path(args.out) if args.out else \
        Path(args.results).with_suffix(".csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(build_row(r, verdicts))

    valid = [r for r in rows if "error" not in r
             and str(r.get("prediction", "")).strip() != "ERROR"]
    n_ok = sum(build_row(r, verdicts)["is_correct_rescored"] for r in valid)
    print(f"✓ {out}  ({len(rows)} rows)")
    print(f"  rescored accuracy: {n_ok}/{len(valid)} = "
          f"{n_ok / max(len(valid), 1) * 100:.1f}%")

    if args.per_path:
        pp = out.with_name(out.stem + "_per_path.csv")
        n = write_per_path(rows, pp)
        print(f"✓ {pp}  ({n} path rows)")

    if args.summary:
        sp = out.with_name(out.stem + "_summary.csv")
        srows = write_summary(rows, verdicts, sp)
        print(f"✓ {sp}")
        for s in srows:
            print(f"  {s['model']:<16} {s['n_correct']:>4}/{s['n_answered']:<4} "
                  f"= {s['accuracy_pct']:5.1f}%   share {s['share_pct']:5.1f}%"
                  + (f"   {s['mean_latency_s']}s/q" if s['mean_latency_s'] != "" else ""))


if __name__ == "__main__":
    main()
