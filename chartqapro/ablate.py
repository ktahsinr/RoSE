# =============================================================
# Offline ablation of the AGGREGATION extensions
# =============================================================
# Recomputes the vote from the per-path answers already logged in a
# results file, under every combination of the two aggregation
# extensions, and re-scores each. Free, instant, CPU-only, no GPU.
#
#   python ablate.py results/rose_factoid_mcq_results.json
#
# WHAT THIS CAN AND CANNOT ABLATE
# --------------------------------
# Two of the six extensions only affect how the m per-path answers are
# combined. Since rose_chartqapro.py logs every path's raw output and
# extracted answer, those two can be ablated offline from ONE run:
#
#     numeric_vote_clustering   ✓ offline
#     drop_malformed_paths      ✓ offline
#
# The other four change what the model is shown or how it decodes, so
# they cannot be recovered from a finished run — each needs its own run:
#
#     mcq_permute_options       ✗ changes the prompts
#     type_aware_retrieval      ✗ changes the demonstrations
#     greedy_first_path         ✗ changes the sampling
#     chart_reading_scaffold    ✗ changes the prompt
#
# For those, set the flag in rose_chartqapro.EXTENSIONS and re-run. Each
# run writes results/meta.json recording its configuration, so results
# files stay self-describing and the ablation table stays honest.
# =============================================================

import argparse
import json
import sys
from pathlib import Path

try:
    from .scoring import (aggregate_answers, has_final_answer, is_correct,
                          normalize_answer, resolve_choice, NUMERIC_TOLERANCE)
except ImportError:
    from scoring import (aggregate_answers, has_final_answer, is_correct,
                         normalize_answer, resolve_choice, NUMERIC_TOLERANCE)


def canonical_mcq(span, opts, perm):
    """Same mapping the pipeline uses: permuted answer -> canonical letter."""
    if not opts or not perm:
        return normalize_answer(span, "mcq")
    shown = [opts[perm[d]] for d in range(len(perm))]
    idx = resolve_choice(span, shown)
    if idx is not None:
        return chr(65 + perm[idx]).lower()
    return normalize_answer(span, "mcq")


def per_path_answers(row):
    """
    Recover the per-path canonical answers from a logged row.
    Returns (canonical_list, raw_list) or (None, None) if the row lacks
    per-path logging (e.g. produced before that was added).
    """
    extracted = row.get("extracted")
    raws = row.get("raw_outputs") or []
    if not extracted:
        return None, None

    qtype = str(row.get("question_type", "factoid")).lower().strip()
    if qtype == "mcq":
        opts = row.get("options_shown") or row.get("choices")
        perms = row.get("option_perms")
        if opts and perms and len(perms) == len(extracted):
            canonical = [canonical_mcq(extracted[i], opts, perms[i])
                         for i in range(len(extracted))]
        else:
            canonical = [normalize_answer(s, "mcq") for s in extracted]
    else:
        canonical = [normalize_answer(s, qtype) for s in extracted]
    return canonical, raws


def rescore(rows, use_clustering: bool, drop_malformed: bool):
    """Recompute the winner for every row under one aggregation config."""
    n_correct = n_total = n_skipped = 0
    for row in rows:
        if "error" in row or row.get("prediction") == "ERROR":
            continue
        qtype = str(row.get("question_type", "factoid")).lower().strip()
        gold = row.get("ground_truth", "")
        opts = row.get("options_shown") or row.get("choices")

        canonical, raws = per_path_answers(row)
        if canonical is None:
            # No per-path log — fall back to the stored prediction so the
            # row still counts, and report how many were affected.
            n_skipped += 1
            n_total += 1
            if is_correct(row.get("prediction", ""), gold, qtype,
                          choices=opts if qtype == "mcq" else None):
                n_correct += 1
            continue

        idx = list(range(len(canonical)))
        if drop_malformed and raws and len(raws) == len(canonical):
            well_formed = [i for i in idx if has_final_answer(raws[i])]
            if well_formed:
                idx = well_formed

        agg = aggregate_answers(
            [canonical[i] for i in idx],
            tolerance=NUMERIC_TOLERANCE,
            use_clustering=(qtype != "mcq") and use_clustering,
        )
        winner = agg["winner"]
        if qtype == "mcq" and len(winner) == 1:
            winner = winner.upper()

        n_total += 1
        if is_correct(winner, gold, qtype,
                      choices=opts if qtype == "mcq" else None):
            n_correct += 1

    return n_correct, n_total, n_skipped


def load(path):
    p = Path(path)
    if not p.exists():
        sys.exit(f"error: no such file: {path}")
    with p.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("results") or list(data.values())
    return [r for r in data if isinstance(r, dict)]


def show_meta(results_path):
    """Print the run's recorded configuration, if meta.json sits beside it."""
    meta_path = Path(results_path).parent / "meta.json"
    if not meta_path.exists():
        print("  (no meta.json beside this results file — configuration "
              "unrecorded, so treat the table below with care)")
        return
    with meta_path.open(encoding="utf-8") as f:
        meta = json.load(f)
    print(f"  run config: model={meta.get('model')}  "
          f"embedder={meta.get('embed_model')}  m={meta.get('m_paths')}  "
          f"paper_faithful={meta.get('paper_faithful')}")
    exts = meta.get("extensions", {})
    if exts:
        on = [k for k, v in exts.items() if v] or ["none"]
        print(f"  extensions at run time: {', '.join(on)}")
        print("  NOTE: prompt/retrieval/decoding extensions above are baked "
              "into this file and cannot be ablated offline.")


def main():
    ap = argparse.ArgumentParser(
        description="Offline ablation of the aggregation extensions, "
                    "recomputed from logged per-path answers. No GPU.")
    ap.add_argument("results", help="results JSON with per-path logging")
    ap.add_argument("--by-type", action="store_true",
                    help="also break each row down by question type")
    args = ap.parse_args()

    rows = load(args.results)
    print(f"\n{'=' * 72}")
    print(f"  AGGREGATION ABLATION — {args.results}")
    print(f"{'=' * 72}")
    show_meta(args.results)

    groups = [("all", rows)]
    if args.by_type:
        for qt in sorted({str(r.get("question_type", "?")).lower().strip()
                          for r in rows}):
            groups.append((qt, [r for r in rows
                                if str(r.get("question_type", "?")).lower()
                                .strip() == qt]))

    configs = [
        ("paper-faithful aggregation (Eq. 1-3)", False, False),
        ("+ drop malformed paths",               False, True),
        ("+ numeric vote clustering",            True,  False),
        ("+ both",                               True,  True),
    ]

    for label, subset in groups:
        if not subset:
            continue
        print(f"\n  --- {label}  (n={len(subset)}) ---")
        print(f"  {'configuration':<40} {'accuracy':>18}")
        print(f"  {'-' * 40} {'-' * 18}")
        baseline = None
        skipped = 0
        for name, clustering, drop in configs:
            correct, total, n_skip = rescore(subset, clustering, drop)
            skipped = max(skipped, n_skip)
            acc = correct / total * 100 if total else 0.0
            if baseline is None:
                baseline = acc
                delta = ""
            else:
                delta = f"  ({acc - baseline:+.1f}pp)"
            print(f"  {name:<40} {acc:5.1f}%  ({correct}/{total}){delta}")
        if skipped:
            print(f"  ⚠  {skipped} rows had no per-path log; their stored "
                  f"prediction was used unchanged in every row above, so the "
                  f"deltas understate the real effect.")

    print(f"\n{'=' * 72}")
    print("  Offline-ablatable: numeric_vote_clustering, drop_malformed_paths")
    print("  Needs its own run: mcq_permute_options, type_aware_retrieval,")
    print("                     greedy_first_path, chart_reading_scaffold")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
