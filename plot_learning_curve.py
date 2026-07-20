#!/usr/bin/env python
"""Plot RoSE learning curves.

The central claim of RoSE is that it *self-improves* as its experience pool
grows. To show that, we stream questions through the backend (pool starting
empty, warmup=0) and track accuracy as a function of how many questions have
been answered:

  - cumulative accuracy  — running correct/seen for each method
  - rolling accuracy     — accuracy over a sliding window (local trend)

RoSE should climb as the pool fills, while Zero-Shot-CoT (which ignores the
pool) stays roughly flat — that gap is the learning curve.

Usage:
  .venv/bin/python plot_learning_curve.py CommonsenseQA 30 3
  .venv/bin/python plot_learning_curve.py CommonsenseQA,GSM8K 20 3
  .venv/bin/python plot_learning_curve.py --from-cache CommonsenseQA   # replot saved data
"""
import json
import os
import sys
import urllib.request

API = os.environ.get("ROSE_API", "http://localhost:3001")
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(OUTDIR, exist_ok=True)

METHODS = ["Zero-Shot-CoT", "Auto-CoT", "RoSE"]
COLORS = {"Zero-Shot-CoT": "#94a3b8", "Auto-CoT": "#3b82f6", "RoSE": "#e11d48"}


def run_benchmark(dataset, n, paths):
    """Call the streaming benchmark (warmup=0) and return its per-question log."""
    body = json.dumps({"dataset": dataset, "n": n, "numPaths": paths, "warmup": 0}).encode()
    req = urllib.request.Request(
        f"{API}/api/benchmark", data=body, headers={"Content-Type": "application/json"}
    )
    print(f"  streaming {n} {dataset} questions x {paths} paths (pool grows from empty)…", flush=True)
    with urllib.request.urlopen(req, timeout=7200) as r:
        data = json.load(r)
    if "error" in data:
        raise RuntimeError(data["error"])
    return data


def curves_from_log(log):
    """Compute cumulative + rolling accuracy series per method from the log."""
    series = {m: {"cum": [], "roll": [], "correct": []} for m in METHODS}
    window = max(4, len(log) // 5)
    for m in METHODS:
        correct = [1 if row["predictions"][m]["correct"] else 0 for row in log]
        series[m]["correct"] = correct
        run = 0
        for i, ok in enumerate(correct):
            run += ok
            series[m]["cum"].append(100.0 * run / (i + 1))
            lo = max(0, i - window + 1)
            w = correct[lo : i + 1]
            series[m]["roll"].append(100.0 * sum(w) / len(w))
    return series, window


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.size": 11, "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    nrows = len(results)
    fig, axes = plt.subplots(nrows, 2, figsize=(13, 4.6 * nrows), squeeze=False)
    fig.suptitle("RoSE learning curves — self-improvement as the experience pool grows",
                 fontsize=15, fontweight="bold", y=0.995)

    for r, (dataset, data) in enumerate(results.items()):
        log = data["log"]
        series, window = curves_from_log(log)
        x = list(range(1, len(log) + 1))

        axL, axR = axes[r][0], axes[r][1]
        for m in METHODS:
            axL.plot(x, series[m]["cum"], color=COLORS[m], lw=2.2,
                     marker="o", ms=3, label=m)
            axR.plot(x, series[m]["roll"], color=COLORS[m], lw=2.2, label=m)

        final = {m: series[m]["cum"][-1] for m in METHODS}
        axL.set_title(f"{dataset} — cumulative accuracy   "
                      f"(final: RoSE {final['RoSE']:.0f}% · "
                      f"Auto {final['Auto-CoT']:.0f}% · Zero {final['Zero-Shot-CoT']:.0f}%)",
                      fontsize=11, fontweight="bold")
        axL.set_xlabel("questions answered  (= RoSE pool size)")
        axL.set_ylabel("cumulative accuracy (%)")
        axL.set_ylim(0, 105)
        axL.legend(loc="lower right", framealpha=0.9)

        axR.set_title(f"{dataset} — rolling accuracy (window = {window})",
                      fontsize=11, fontweight="bold")
        axR.set_xlabel("questions answered")
        axR.set_ylabel("windowed accuracy (%)")
        axR.set_ylim(0, 105)
        axR.legend(loc="lower right", framealpha=0.9)

    fig.text(0.5, 0.005,
             "Local llama3.1:8b · small samples are noisy — the signal is RoSE trending "
             "at or above the baselines as its pool fills.",
             ha="center", fontsize=9, color="#666")
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    tag = "_".join(results.keys())
    out = os.path.join(OUTDIR, f"learning_curve_{tag}.png")
    fig.savefig(out, dpi=140)
    print(f"\n  ✓ saved {out}")
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    from_cache = "--from-cache" in sys.argv
    datasets = (args[0] if args else "CommonsenseQA").split(",")
    n = int(args[1]) if len(args) > 1 else 30
    paths = int(args[2]) if len(args) > 2 else 3

    results = {}
    for ds in datasets:
        cache = os.path.join(OUTDIR, f"data_{ds}.json")
        if from_cache:
            print(f"  loading cached {ds}…")
            with open(cache) as f:
                results[ds] = json.load(f)
        else:
            data = run_benchmark(ds, n, paths)
            with open(cache, "w") as f:
                json.dump(data, f)
            acc = data["accuracy"]
            print(f"  {ds} done — " + " · ".join(
                f"{m} {acc[m]['acc']*100:.0f}%" for m in METHODS))
            results[ds] = data

    plot(results)


if __name__ == "__main__":
    main()
