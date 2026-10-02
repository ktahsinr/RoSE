# =============================================================
# Pilot evaluation — every Stage 1 number, from the pool alone
# =============================================================
# CPU-only. Reads the built routing pool and prints/writes:
#
#   per-model pilot accuracy + mean latency per question
#   routed accuracy       (leave-one-out kNN routing)
#   always-7B accuracy    (the baseline routing must beat on cost)
#   oracle accuracy       (upper bound: any model right → right)
#   routing share per model, mean parameter count per query
#
# Leave-one-out means each pilot question is routed with itself removed
# from the pool, so the routed accuracy is an honest estimate rather
# than a self-lookup.
#
#   python -m routing.evaluate_pilot results/pilot/routing_pool.json \
#          --out results/pilot/pilot_report.json
# =============================================================

import argparse
import json
from pathlib import Path

try:
    from .config import ESCALATION_ORDER, K_NEIGHBOURS, MODELS, TAU
    from .pool import load_pool
    from .router import leave_one_out
except ImportError:                 # flat / notebook use
    from config import ESCALATION_ORDER, K_NEIGHBOURS, MODELS, TAU
    from pool import load_pool
    from router import leave_one_out


def evaluate(pool: dict, k: int = K_NEIGHBOURS, tau: float = TAU) -> dict:
    entries = pool["entries"]
    n = len(entries)
    models = [m for m in ESCALATION_ORDER if m in entries[0]["model_hits"]]
    fallback = models[-1]

    # ── per-model accuracy + latency ─────────────────────────
    per_model = {}
    for m in models:
        hits = [e["model_hits"][m] for e in entries]
        lats = [e["latency_s"][m] for e in entries
                if e.get("latency_s", {}).get(m) is not None]
        per_model[m] = {
            "accuracy":       sum(hits) / n,
            "n":              n,
            "mean_latency_s": (sum(lats) / len(lats)) if lats else None,
            "params_b":       MODELS[m]["params_b"],
        }

    # ── leave-one-out routing ────────────────────────────────
    loo = leave_one_out(pool, k=k, tau=tau, order=models)
    share = {m: 0 for m in models}
    routed_correct = 0
    routed_params = 0.0
    per_routed = {m: {"correct": 0, "total": 0} for m in models}
    for row in loo:
        m = row["model"]
        share[m] += 1
        routed_correct += row["hit"]
        routed_params += MODELS[m]["params_b"]
        per_routed[m]["total"] += 1
        per_routed[m]["correct"] += row["hit"]

    # ── oracle: right if ANY model is right; cost = smallest right
    #    model, fallback price when none is ────────────────────
    oracle_correct = 0
    oracle_params = 0.0
    for e in entries:
        winners = [m for m in models if e["model_hits"][m]]
        if winners:
            oracle_correct += 1
            oracle_params += min(MODELS[m]["params_b"] for m in winners)
        else:
            oracle_params += MODELS[fallback]["params_b"]

    return {
        "n": n, "k": k, "tau": tau, "models": models,
        "per_model": per_model,
        "routing": {
            "accuracy":            routed_correct / n,
            "share":               {m: share[m] / n for m in models},
            "mean_params_b":       routed_params / n,
            "accuracy_on_routed":  {
                m: (per_routed[m]["correct"] / per_routed[m]["total"])
                   if per_routed[m]["total"] else None
                for m in models
            },
        },
        "always_7b": {
            "accuracy":      per_model[fallback]["accuracy"],
            "mean_params_b": MODELS[fallback]["params_b"],
        },
        "oracle": {
            "accuracy":      oracle_correct / n,
            "mean_params_b": oracle_params / n,
        },
        "decisions": loo,
    }


def print_report(rep: dict):
    pct = lambda x: f"{x * 100:5.1f}%"       # noqa: E731
    print(f"\n{'=' * 62}\n  PILOT REPORT   n={rep['n']}   "
          f"k={rep['k']}   τ={rep['tau']}\n{'=' * 62}")

    print(f"\n  {'model':<18} {'accuracy':>9} {'latency/q':>10} {'params':>7}")
    for m in rep["models"]:
        s = rep["per_model"][m]
        lat = f"{s['mean_latency_s']:.1f}s" if s["mean_latency_s"] else "—"
        print(f"  {m:<18} {pct(s['accuracy']):>9} {lat:>10} {s['params_b']:>6.1f}B")

    r = rep["routing"]
    print(f"\n  {'policy':<18} {'accuracy':>9} {'params/q':>10}")
    print(f"  {'routed (kNN, LOO)':<18} {pct(r['accuracy']):>9} {r['mean_params_b']:>9.2f}B")
    print(f"  {'always-7B':<18} {pct(rep['always_7b']['accuracy']):>9} "
          f"{rep['always_7b']['mean_params_b']:>9.2f}B")
    print(f"  {'oracle':<18} {pct(rep['oracle']['accuracy']):>9} "
          f"{rep['oracle']['mean_params_b']:>9.2f}B")

    print("\n  routing share (and accuracy on the questions routed there):")
    for m in rep["models"]:
        acc = r["accuracy_on_routed"][m]
        print(f"  {m:<18} {pct(r['share'][m])}   "
              f"acc {pct(acc) if acc is not None else '    —'}")
    print()


def _main():
    ap = argparse.ArgumentParser(description="Evaluate Stage 1 routing on the pilot pool")
    ap.add_argument("pool", help="routing_pool.json from routing/pool.py")
    ap.add_argument("--k", type=int, default=K_NEIGHBOURS)
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--out", default=None, help="write the full report JSON here")
    args = ap.parse_args()

    rep = evaluate(load_pool(args.pool), k=args.k, tau=args.tau)
    print_report(rep)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=2))
        print(f"✓ full report → {args.out}")


if __name__ == "__main__":
    _main()
