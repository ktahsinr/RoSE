# =============================================================
# Summarize a ChartQAPro results file as JSON
# =============================================================
# CPU-only, stdlib-only. This is the bridge between the offline
# Colab/Kaggle runs and the web frontend: the Node backend shells
# out to this script and serves whatever it prints.
#
# Scoring is NOT reimplemented here — every verdict comes from
# scoring.py and every ablation number from ablate.py, so the
# dashboard, the CSV export and the terminal tables can never
# disagree about the same run.
#
#   python summarize.py results/rose_factoid_mcq_results.json
#   python summarize.py results/rose.json --verdicts results/verdicts.json
#   python summarize.py results/rose.json --rows --offset 0 --limit 50
#   python summarize.py results/rose.json --row-id 42     # one row, full paths
# =============================================================

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .scoring import is_correct, normalize_answer, is_unanswerable
    from .ablate import ablation_table
except ImportError:
    from scoring import is_correct, normalize_answer, is_unanswerable
    from ablate import ablation_table


# ─────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────

def load(path):
    p = Path(path)
    if not p.exists():
        sys.exit(f"error: no such file: {path}")
    with p.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("results") or list(data.values())
    return [r for r in data if isinstance(r, dict)]


def load_meta(results_path):
    """
    The run's provenance. Same candidate list ablate.show_meta() uses — a
    per-type run writes meta_factoid.json / meta_mcq.json rather than a
    single meta.json.
    """
    base = Path(results_path)
    candidates = [
        base.parent / f"meta_{base.stem.split('_')[-1]}.json",
        base.parent / "meta.json",
        base.with_suffix(".meta.json"),
    ]
    meta_path = next((p for p in candidates if p.exists()), None)
    if not meta_path:
        return None
    try:
        with meta_path.open(encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def load_verdicts(path):
    """Judge verdicts keyed by row id (same convention as export_csv.py)."""
    if not path:
        return {}
    out = {}
    for r in load(path):
        if r.get("verdict"):
            out[str(r.get("id", ""))] = r["verdict"]
    return out


def is_error(r):
    return "error" in r or str(r.get("prediction", "")).strip() == "ERROR"


def rescore_row(r):
    """Current normalized verdict for one row, via scoring.py."""
    if is_error(r):
        return False
    qtype = str(r.get("question_type", "factoid")).lower().strip()
    opts = r.get("options_shown") or r.get("choices")
    return bool(is_correct(r.get("prediction", ""), r.get("ground_truth", ""),
                           qtype, choices=opts if qtype == "mcq" else None))


def _acc(correct, total):
    return {"correct": correct, "total": total,
            "acc": round(correct / total * 100, 2) if total else 0.0}


def _num(v):
    """Coerce a logged metric to float, tolerating '' and None."""
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────
# Breakdowns
# ─────────────────────────────────────────────────────────────

def breakdown(valid, key_of, verdict_of):
    tally = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in valid:
        t = tally[key_of(r)]
        t["total"] += 1
        if verdict_of(r):
            t["correct"] += 1
    return [{"key": k, **_acc(v["correct"], v["total"])}
            for k, v in sorted(tally.items())]


def learning_curve(valid, verdict_of, max_points=400):
    """
    Accuracy as the experience pool grows. This is the whole RoSE claim in
    one series, so it is computed over the file's stream order — the order
    the questions were actually answered in.

    Rolling accuracy uses a window that scales with the run so a 60-question
    smoke test and a 2k-question sweep both give a readable line.
    """
    n = len(valid)
    if not n:
        return {"points": [], "window": 0, "phase_switch": None}

    window = max(10, min(100, n // 20))
    verdicts = [1 if verdict_of(r) else 0 for r in valid]

    phase_switch = None
    points = []
    running = 0
    for i, r in enumerate(valid):
        running += verdicts[i]
        lo = max(0, i - window + 1)
        win = verdicts[lo:i + 1]
        method = r.get("method", "")
        if phase_switch is None and method == "rose_few_shot":
            phase_switch = i
        points.append({
            "i":         i,
            "pool":      r.get("pool_size", None),
            "method":    method,
            "correct":   bool(verdicts[i]),
            "cum_acc":   round(running / (i + 1) * 100, 2),
            "roll_acc":  round(sum(win) / len(win) * 100, 2),
        })

    # Downsample for transport — keep the shape, drop the resolution nobody
    # can see on a 900px-wide chart. Always keep the first and last point.
    if len(points) > max_points:
        step = len(points) / max_points
        idx = sorted({int(i * step) for i in range(max_points)} | {len(points) - 1})
        points = [points[i] for i in idx]

    return {"points": points, "window": window, "phase_switch": phase_switch}


def uncertainty_bins(valid, verdict_of, nbins=5):
    """Is the entropy signal actually predictive? Accuracy per uncertainty bin."""
    pairs = [(u, verdict_of(r)) for r in valid
             if (u := _num(r.get("uncertainty"))) is not None]
    if not pairs:
        return []
    lo = min(u for u, _ in pairs)
    hi = max(u for u, _ in pairs)
    if hi - lo < 1e-9:
        return [{"lo": round(lo, 3), "hi": round(hi, 3),
                 **_acc(sum(1 for _, ok in pairs if ok), len(pairs))}]

    width = (hi - lo) / nbins
    tally = defaultdict(lambda: {"correct": 0, "total": 0})
    for u, ok in pairs:
        b = min(int((u - lo) / width), nbins - 1)
        tally[b]["total"] += 1
        if ok:
            tally[b]["correct"] += 1
    return [{"lo": round(lo + b * width, 3), "hi": round(lo + (b + 1) * width, 3),
             **_acc(t["correct"], t["total"])}
            for b, t in sorted(tally.items())]


def metric_stats(valid, field):
    vals = [v for r in valid if (v := _num(r.get(field))) is not None]
    if not vals:
        return None
    vals_sorted = sorted(vals)
    mid = len(vals_sorted) // 2
    median = (vals_sorted[mid] if len(vals_sorted) % 2
              else (vals_sorted[mid - 1] + vals_sorted[mid]) / 2)
    return {"mean": round(sum(vals) / len(vals), 4),
            "median": round(median, 4),
            "min": round(vals_sorted[0], 4),
            "max": round(vals_sorted[-1], 4),
            "n": len(vals)}


def judge_summary(valid, verdicts, verdict_of):
    """
    What the semantic judge changed. Only rows the judge actually saw are
    counted — an audit sample must not be reported as if it were the run.
    """
    seen = [r for r in valid if str(r.get("id", "")) in verdicts]
    if not seen:
        return None

    recovered, still_wrong, agreed = [], [], 0
    for r in seen:
        rule = verdict_of(r)
        j = verdicts[str(r.get("id", ""))]
        jc = bool(j.get("correct"))
        if jc and not rule:
            recovered.append(j)
        elif not jc and rule:
            still_wrong.append(j)
        else:
            agreed += 1

    n_correct = sum(1 for r in seen
                    if verdicts[str(r.get("id", ""))].get("correct"))
    return {
        "judged":       len(seen),
        "coverage":     round(len(seen) / len(valid) * 100, 1),
        "judge_acc":    _acc(n_correct, len(seen)),
        "rule_acc":     _acc(sum(1 for r in seen if verdict_of(r)), len(seen)),
        "agreed":       agreed,
        "recovered":    [{"category": c, "n": n} for c, n in
                         Counter(v.get("category", "?")
                                 for v in recovered).most_common()],
        "still_wrong":  [{"category": c, "n": n} for c, n in
                         Counter(v.get("category", "?")
                                 for v in still_wrong).most_common()],
        "categories":   [{"category": c, "n": n} for c, n in
                         Counter(v.get("category", "?")
                                 for v in verdicts.values()).most_common()],
    }


# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────

def summarize(path, verdicts_path=None, with_ablation=True):
    rows = load(path)
    verdicts = load_verdicts(verdicts_path)
    meta = load_meta(path)

    errors = [r for r in rows if is_error(r)]
    valid = [r for r in rows if not is_error(r)]

    # Rescored is the number to trust: `is_correct` moved after some runs
    # were written, and the file keeps whatever was true at run time.
    rescored = {id(r): rescore_row(r) for r in valid}
    verdict_of = lambda r: rescored[id(r)]              # noqa: E731
    run_of = lambda r: bool(r.get("is_correct", False))  # noqa: E731

    qtype_of = lambda r: str(r.get("question_type", "unknown")).lower().strip()  # noqa: E731
    method_of = lambda r: r.get("method", "unknown")     # noqa: E731
    answerable_of = lambda r: ("unanswerable"            # noqa: E731
                               if is_unanswerable(r.get("ground_truth", ""))
                               else "answerable")

    answerable_said_unans = sum(
        1 for r in valid
        if not is_unanswerable(r.get("ground_truth", ""))
        and is_unanswerable(r.get("prediction", "")))
    unanswerable_said_val = sum(
        1 for r in valid
        if is_unanswerable(r.get("ground_truth", ""))
        and not is_unanswerable(r.get("prediction", "")))

    n_rescored = sum(1 for r in valid if verdict_of(r))
    n_run = sum(1 for r in valid if run_of(r))

    out = {
        "file":        str(path),
        "meta":        meta,
        "extensions":  (meta or {}).get("extensions"),
        "n_rows":      len(rows),
        "n_valid":     len(valid),
        "n_errors":    len(errors),
        "overall": {
            "rescored": _acc(n_rescored, len(valid)),
            "run":      _acc(n_run, len(valid)),
            "drift":    n_rescored - n_run,
        },
        "by_type":       breakdown(valid, qtype_of, verdict_of),
        "by_method":     breakdown(valid, method_of, verdict_of),
        "by_answerable": breakdown(valid, answerable_of, verdict_of),
        # Which aggregation path each answer actually took (exact / cluster /
        # weighted) — only populated by runs new enough to log vote_mode.
        "by_vote_mode":  breakdown([r for r in valid if r.get("vote_mode")],
                                   lambda r: r.get("vote_mode"), verdict_of),
        "confusion": {
            "answerable_said_unanswerable": answerable_said_unans,
            "unanswerable_said_value":      unanswerable_said_val,
        },
        "curve":             learning_curve(valid, verdict_of),
        "uncertainty_bins":  uncertainty_bins(valid, verdict_of),
        "metrics": {
            "uncertainty": metric_stats(valid, "uncertainty"),
            "complexity":  metric_stats(valid, "complexity"),
            "agreement":   metric_stats(valid, "agreement"),
        },
        "judge":     judge_summary(valid, verdicts, verdict_of),
        "has_paths": any(r.get("extracted") for r in valid),
        "errors":    [{"id": r.get("id", ""), "question": r.get("question", ""),
                       "error": r.get("error", "")} for r in errors[:20]],
    }

    # The offline ablation only means anything when per-path answers were
    # logged; without them every configuration returns the same number.
    if with_ablation and out["has_paths"]:
        out["ablation"] = ablation_table(rows, by_type=True)
    else:
        out["ablation"] = None

    return out


# ─────────────────────────────────────────────────────────────
# Row browsing
# ─────────────────────────────────────────────────────────────

def trim_row(r, verdicts):
    """A row without the multi-KB raw generations — those load on demand."""
    v = verdicts.get(str(r.get("id", "")), {})
    gold = r.get("ground_truth", "")
    return {
        "id":            r.get("id", ""),
        "question":      r.get("question", ""),
        "question_type": str(r.get("question_type", "")).lower().strip(),
        "ground_truth":  "|".join(map(str, gold)) if isinstance(gold, (list, tuple))
                         else gold,
        "prediction":    r.get("prediction", ""),
        "prediction_norm": r.get("prediction_norm")
                           or normalize_answer(r.get("prediction", ""),
                                               str(r.get("question_type",
                                                         "factoid")).lower()),
        "is_correct_run":      bool(r.get("is_correct", False)),
        "is_correct":          rescore_row(r),
        "judge_correct":       v.get("correct"),
        "judge_category":      v.get("category"),
        "judge_reason":        v.get("reason"),
        "gold_unanswerable":   is_unanswerable(gold),
        "pred_unanswerable":   is_unanswerable(r.get("prediction", "")),
        "method":       r.get("method", ""),
        "uncertainty":  _num(r.get("uncertainty")),
        "complexity":   _num(r.get("complexity")),
        "agreement":    _num(r.get("agreement")),
        "n_demos":      r.get("n_demos"),
        "pool_size":    r.get("pool_size"),
        "n_paths":      len(r.get("extracted") or []),
        # How the m paths were actually combined for this row: exact-string
        # majority, tolerance clustering, or confidence-weighted.
        "vote_mode":    r.get("vote_mode", ""),
        "n_groups":     r.get("n_groups"),
        "paths_voting": r.get("paths_voting"),
        "error":        r.get("error", ""),
    }


def browse(path, offset=0, limit=50, qtype=None, only=None, q=None,
           verdicts_path=None):
    rows = load(path)
    verdicts = load_verdicts(verdicts_path)
    trimmed = [trim_row(r, verdicts) for r in rows]

    if qtype and qtype != "all":
        trimmed = [r for r in trimmed if r["question_type"] == qtype]
    if only == "wrong":
        trimmed = [r for r in trimmed if not r["is_correct"]]
    elif only == "correct":
        trimmed = [r for r in trimmed if r["is_correct"]]
    elif only == "disagree":
        # Where rescoring or the judge changed the run's own verdict — the
        # rows worth reading by hand.
        trimmed = [r for r in trimmed
                   if r["is_correct"] != r["is_correct_run"]
                   or (r["judge_correct"] is not None
                       and r["judge_correct"] != r["is_correct"])]
    elif only == "errors":
        trimmed = [r for r in trimmed if r["error"]]
    if q:
        needle = q.lower()
        trimmed = [r for r in trimmed
                   if needle in str(r["question"]).lower()
                   or needle in str(r["id"]).lower()]

    total = len(trimmed)
    return {"total": total, "offset": offset, "limit": limit,
            "rows": trimmed[offset:offset + limit]}


def one_row(path, row_id, verdicts_path=None):
    """One row with everything — per-path generations, demos, rationale."""
    verdicts = load_verdicts(verdicts_path)
    for r in load(path):
        if str(r.get("id", "")) == str(row_id):
            raws = r.get("raw_outputs") or []
            extracted = r.get("extracted") or []
            normalized = r.get("normalized") or []
            paths = [{
                "i":          i,
                "decoding":   "greedy" if i == 0 else "sampled",
                "well_formed": "final answer" in str(
                    raws[i] if i < len(raws) else "").lower(),
                "extracted":  extracted[i] if i < len(extracted) else "",
                "normalized": normalized[i] if i < len(normalized) else "",
                "raw":        raws[i] if i < len(raws) else "",
            } for i in range(max(len(extracted), len(raws)))]
            return {
                **trim_row(r, verdicts),
                "choices":       r.get("choices"),
                "options_shown": r.get("options_shown"),
                "option_perms":  r.get("option_perms"),
                "best_rationale": r.get("best_rationale", ""),
                # two_stage_reasoning writes the model's own description of the
                # chart before it answers — worth reading when an answer is wrong.
                "chart_description": r.get("chart_description", ""),
                "paths":         paths,
            }
    sys.exit(f"error: no row with id {row_id}")


def main():
    ap = argparse.ArgumentParser(
        description="Summarize a ChartQAPro results file as JSON.")
    ap.add_argument("results")
    ap.add_argument("--verdicts", help="judge verdicts JSON (rescore_judge --out)")
    ap.add_argument("--no-ablation", action="store_true",
                    help="skip the offline aggregation ablation (faster)")
    ap.add_argument("--rows", action="store_true", help="browse rows instead")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--type", dest="qtype", help="factoid | mcq | all")
    ap.add_argument("--only", choices=["wrong", "correct", "disagree", "errors"])
    ap.add_argument("--q", help="substring filter on question or id")
    ap.add_argument("--row-id", help="dump one row in full, with every path")
    args = ap.parse_args()

    if args.row_id is not None:
        out = one_row(args.results, args.row_id, args.verdicts)
    elif args.rows:
        out = browse(args.results, args.offset, args.limit, args.qtype,
                     args.only, args.q, args.verdicts)
    else:
        out = summarize(args.results, args.verdicts,
                        with_ablation=not args.no_ablation)

    json.dump(out, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
