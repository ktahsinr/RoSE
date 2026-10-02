# =============================================================
# Draw the pilot sample — 150 factoid questions, seeded
# =============================================================
# CPU-only, stdlib-only. Samples ids FROM THE CACHED 7B RESULTS FILE,
# not from the dataset directly, for two reasons:
#
#   1. The ids in rose_factoid.json are the ids every later file must
#      agree on; sampling from the same file makes drift impossible.
#   2. The 7B answers for the sample are then guaranteed to exist —
#      only the two 4B models still need a GPU pass over it.
#
# The sample is stratified on answerability (gold Unanswerable vs not),
# so the pilot sees the same punt-rate mix as the full set, and the
# draw is seeded so the file can always be regenerated identically.
#
#   python scripts/make_pilot_sample.py results/rose_factoid.json \
#          --n 150 --seed 499 --out results/pilot/sample_ids.json
# =============================================================

import argparse
import json
import random
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scoring import is_unanswerable          # noqa: E402


def make_sample(results_path, n: int = 150, seed: int = 499) -> dict:
    rows = json.loads(Path(results_path).read_text())
    rows = [r for r in rows if r.get("id") and r.get("method") != "error"]
    if len(rows) < n:
        raise ValueError(f"only {len(rows)} usable rows in {results_path}, need {n}")

    unans = [r["id"] for r in rows if is_unanswerable(str(r.get("ground_truth", "")))]
    ans = [r["id"] for r in rows if r["id"] not in set(unans)]

    rng = random.Random(seed)
    n_unans = round(n * len(unans) / len(rows))
    picked = rng.sample(unans, n_unans) + rng.sample(ans, n - n_unans)
    rng.shuffle(picked)

    return {
        "seed":    seed,
        "n":       len(picked),
        "source":  Path(results_path).name,
        "strata":  {"answerable": n - n_unans, "unanswerable": n_unans},
        "ids":     picked,
    }


def _main():
    ap = argparse.ArgumentParser(description="Draw the seeded pilot sample")
    ap.add_argument("results", help="cached 7B results file (rose_factoid.json)")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=499)
    ap.add_argument("--out", default="results/pilot/sample_ids.json")
    args = ap.parse_args()

    sample = make_sample(args.results, n=args.n, seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(sample, indent=2))
    print(f"✓ {sample['n']} ids (seed {sample['seed']}, "
          f"{sample['strata']['unanswerable']} unanswerable) → {args.out}")


if __name__ == "__main__":
    _main()
