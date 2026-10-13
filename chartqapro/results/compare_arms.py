# =============================================================
# Compare the arms: no-router 7B vs routed vs decompose Alg 1 / Alg 2
# =============================================================
# CPU-only. Reads the routed run's CSV, both decomposition CSVs and the
# cached no-router 7B results, RESCORES every prediction with scoring.py
# (the single source of truth for correctness — nothing here re-decides
# what "right" means), restricts all arms to the ids they share, and
# prints one table: arm, accuracy, mean params/question, mean
# latency/question — plus, for Algorithm 1, the counts of
# (sub-answer right, final wrong) and (sub-answer wrong, final right).
#
#   cd chartqapro
#   python results/compare_arms.py
#   python results/compare_arms.py --alg1 /path/to/decomp_alg1.csv
#
# Missing files are skipped with a note, so the table grows as runs land.
# =============================================================

import argparse
import csv
import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RESULTS_DIR.parent))

from scoring import is_correct                                      # noqa: E402
from decompose.models import MODELS                                 # noqa: E402


def _load_csv(path):
    with Path(path).open(encoding="utf-8") as f:
        return [row for row in csv.DictReader(f)]


def _is_error(row):
    return bool(row.get("error")) or \
        str(row.get("prediction") or row.get("final_answer") or "").strip() == "ERROR"


def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _norm_rows(rows, pred_key, params_fn):
    """
    One normalized record per question: id, rescored verdict, latency,
    params. `params_fn(row)` supplies the per-question parameter cost.
    """
    out = {}
    for r in rows:
        if _is_error(r):
            continue
        qid = str(r.get("id", ""))
        qtype = str(r.get("question_type", "factoid")).lower().strip()
        out[qid] = {
            "correct":  is_correct(r.get(pred_key, ""), r.get("ground_truth", ""),
                                   qtype),
            "latency":  _float(r.get("latency_s")),
            "params_b": params_fn(r),
            "row":      r,
        }
    return out


def load_arms(args):
    arms = {}

    p = Path(args.baseline)
    if p.exists():
        rows = json.loads(p.read_text())
        arms["no-router-7b"] = _norm_rows(
            rows, "prediction",
            lambda r: MODELS["qwen2.5-vl-7b"]["params_b"])
    else:
        print(f"  (skipping no-router-7b: {p} not found)")

    p = Path(args.routed)
    if p.exists():
        arms["routed"] = _norm_rows(
            _load_csv(p), "prediction",
            lambda r: MODELS.get(r.get("model_key", ""), {}).get("params_b"))
    else:
        print(f"  (skipping routed: {p} not found)")

    for name, path in (("decomp-alg1", args.alg1), ("decomp-alg2", args.alg2)):
        p = Path(path)
        if p.exists():
            arms[name] = _norm_rows(
                _load_csv(p), "final_answer",
                lambda r: _float(r.get("params_b_used")))
        else:
            print(f"  (skipping {name}: {p} not found)")

    return arms


def alg1_confusion(arm, ids):
    """
    Counts over the common window:
      sub_right_final_wrong — at least one sub-answer matches the gold,
                              but RoSE's final answer does not
      sub_wrong_final_right — no sub-answer matches, final answer does
    Sub-answers are scored against the ORIGINAL gold with scoring.py.
    """
    srfw = swfr = 0
    for qid in ids:
        rec = arm[qid]
        row = rec["row"]
        try:
            subs = json.loads(row.get("sub_answers") or "[]")
        except json.JSONDecodeError:
            subs = []
        gold = row.get("ground_truth", "")
        qtype = str(row.get("question_type", "factoid")).lower().strip()
        sub_right = any(is_correct(a, gold, qtype) for a in subs)
        if sub_right and not rec["correct"]:
            srfw += 1
        if not sub_right and rec["correct"]:
            swfr += 1
    return srfw, swfr


def main():
    ap = argparse.ArgumentParser(
        description="Rescore and compare all arms on their common questions")
    ap.add_argument("--routed",
                    default=RESULTS_DIR / "routed" / "routed_chartqapro.csv")
    ap.add_argument("--alg1",
                    default=RESULTS_DIR / "decompose" / "decomp_alg1.csv")
    ap.add_argument("--alg2",
                    default=RESULTS_DIR / "decompose" / "decomp_alg2.csv")
    ap.add_argument("--baseline", default=RESULTS_DIR / "rose_factoid.json",
                    help="no-router 7B results JSON (the cached sweep)")
    args = ap.parse_args()

    arms = load_arms(args)
    if not arms:
        sys.exit("no arm files found — nothing to compare")

    common = set.intersection(*(set(a) for a in arms.values()))
    if not common:
        sys.exit("the arms share no question ids — are these the same window?")
    print(f"\ncommon questions across {len(arms)} arm(s): {len(common)}"
          + "".join(f"\n  {name}: {len(a)} answered"
                    for name, a in arms.items()))

    print(f"\n{'arm':<14} {'n':>5} {'accuracy':>9} {'params/q':>9} {'latency/q':>10}")
    print("-" * 52)
    for name, arm in arms.items():
        recs = [arm[q] for q in sorted(common)]
        acc = sum(r["correct"] for r in recs) / len(recs) * 100
        params = [r["params_b"] for r in recs if r["params_b"] is not None]
        lats = [r["latency"] for r in recs if r["latency"] is not None]
        p_str = f"{sum(params) / len(params):8.2f}B" if params else "        —"
        l_str = f"{sum(lats) / len(lats):8.1f}s" if lats else "         —"
        print(f"{name:<14} {len(recs):>5} {acc:8.1f}% {p_str} {l_str}")

    if "decomp-alg1" in arms:
        srfw, swfr = alg1_confusion(arms["decomp-alg1"], sorted(common))
        print("\nAlgorithm 1 — sub-answer vs final answer (common window):")
        print(f"  sub-answer right, final wrong : {srfw}")
        print(f"  sub-answer wrong, final right : {swfr}")


if __name__ == "__main__":
    main()
