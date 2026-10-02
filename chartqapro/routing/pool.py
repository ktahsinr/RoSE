# =============================================================
# The routing pool — pilot experience the router consults
# =============================================================
# Built ONCE from the pilot result files (one RoSE results JSON per
# model, all over the same sampled questions). Each entry is:
#
#   {
#     "id":         "factoid-00042",
#     "question":   "...",
#     "qtype":      "factoid",
#     "embedding":  [768 floats]          # all-mpnet-base-v2, like RoSE
#     "model_hits": {"gemma3-4b": 1, "qwen3.5-4b": 0, "qwen2.5-vl-7b": 1},
#     "latency_s":  {"gemma3-4b": 41.2, ...}   # absent if a run didn't log it
#   }
#
# Correctness is NOT read off the result files' is_correct flags — it is
# recomputed with scoring.py, the single source of truth for every
# verdict in this repo, so the router can never disagree with the
# dashboard about what "the 4B got it right" means.
#
# CLI:
#   python -m routing.pool \
#       --results gemma3-4b=results/pilot/pilot_gemma3-4b.json \
#                 qwen3.5-4b=results/pilot/pilot_qwen3.5-4b.json \
#                 qwen2.5-vl-7b=results/rose_factoid.json \
#       --sample  results/pilot/sample_ids.json \
#       --out     results/pilot/routing_pool.json
#
# (run from chartqapro/; the 7B file may be the FULL cached run — it is
# restricted to the sampled ids automatically.)
# =============================================================

import argparse
import json
import time
from pathlib import Path

try:
    from .config import EMBED_MODEL, MODELS
except ImportError:                 # flat / notebook use
    from config import EMBED_MODEL, MODELS

try:
    from ..scoring import is_correct
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scoring import is_correct


POOL_VERSION = 1


def _default_embed_fn():
    """
    Lazy sentence-transformers import: the router only ever READS
    embeddings from a built pool, so nothing else in routing/ needs the
    package installed.
    """
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(EMBED_MODEL)

    def embed(texts):
        return embedder.encode(list(texts), convert_to_numpy=True).tolist()

    return embed


def _index_results(path):
    """id -> result row, skipping errored rows."""
    rows = json.loads(Path(path).read_text())
    return {
        str(r["id"]): r
        for r in rows
        if r.get("id") and r.get("method") != "error" and "error" not in r
    }


def _rescore(row) -> int:
    """Recompute the verdict with scoring.py — never trust a stale flag."""
    qtype = row.get("question_type", "factoid")
    return int(is_correct(
        row.get("prediction", ""),
        row.get("ground_truth", ""),
        qtype,
        choices=row.get("options_shown") if qtype == "mcq" else None,
    ))


def build_pool(result_files: dict, sample_ids=None, embed_fn=None) -> dict:
    """
    result_files : {model_key: path} — one RoSE results JSON per model.
    sample_ids   : restrict to these ids (the pilot sample); None = the
                   intersection of ids present in every file.
    embed_fn     : callable(list[str]) -> list[list[float]]. Defaults to
                   all-mpnet-base-v2 via sentence-transformers; injectable
                   so tests run without downloading a model.

    Only questions answered by EVERY model enter the pool — a neighbour
    with unknown hits for some model would silently bias that model's
    score toward 0 or 1 depending on how it was filled in.
    """
    unknown = set(result_files) - set(MODELS)
    if unknown:
        raise ValueError(f"unknown model keys: {sorted(unknown)} "
                         f"(expected keys from routing/config.py: {sorted(MODELS)})")

    indexed = {m: _index_results(p) for m, p in result_files.items()}

    common = set.intersection(*(set(idx) for idx in indexed.values()))
    if sample_ids is not None:
        common &= {str(i) for i in sample_ids}
    if not common:
        raise ValueError("no question ids are present in every result file "
                         "(and the sample, if one was given)")

    # Stable order: as the ids appear in the first file given.
    first = next(iter(indexed.values()))
    ids = [i for i in first if i in common]

    if embed_fn is None:
        embed_fn = _default_embed_fn()
    questions = [first[i].get("question", "") for i in ids]
    embeddings = embed_fn(questions)

    entries = []
    for qid, emb in zip(ids, embeddings):
        row = first[qid]
        entry = {
            "id":         qid,
            "question":   row.get("question", ""),
            "qtype":      row.get("question_type", "factoid"),
            "embedding":  list(map(float, emb)),
            "model_hits": {m: _rescore(idx[qid]) for m, idx in indexed.items()},
        }
        lat = {m: idx[qid].get("latency_s") for m, idx in indexed.items()
               if idx[qid].get("latency_s") is not None}
        if lat:
            entry["latency_s"] = lat
        entries.append(entry)

    return {
        "version":     POOL_VERSION,
        "embed_model": EMBED_MODEL,
        "models":      sorted(result_files),
        "built_at":    time.strftime("%Y-%m-%d %H:%M:%S"),
        "n":           len(entries),
        "entries":     entries,
    }


def save_pool(pool: dict, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(pool))
    return path


def load_pool(path) -> dict:
    pool = json.loads(Path(path).read_text())
    if pool.get("version") != POOL_VERSION:
        raise ValueError(f"routing pool version {pool.get('version')} != {POOL_VERSION} "
                         f"— rebuild it with routing/pool.py from this checkout")
    return pool


# ─────────────────────────────────────────────────────────────

def _main():
    ap = argparse.ArgumentParser(description="Build the kNN routing pool from pilot results")
    ap.add_argument("--results", nargs="+", required=True, metavar="MODEL=PATH",
                    help="one model_key=results.json pair per model")
    ap.add_argument("--sample", default=None,
                    help="sample_ids.json — restrict the pool to the pilot sample")
    ap.add_argument("--out", default="results/pilot/routing_pool.json")
    args = ap.parse_args()

    result_files = {}
    for pair in args.results:
        model, _, path = pair.partition("=")
        if not path:
            ap.error(f"--results entries look like model_key=path, got '{pair}'")
        result_files[model] = path

    sample_ids = None
    if args.sample:
        blob = json.loads(Path(args.sample).read_text())
        sample_ids = blob["ids"] if isinstance(blob, dict) else blob

    pool = build_pool(result_files, sample_ids=sample_ids)
    save_pool(pool, args.out)

    print(f"✓ routing pool: {pool['n']} questions × {len(pool['models'])} models → {args.out}")
    for m in pool["models"]:
        acc = sum(e["model_hits"][m] for e in pool["entries"]) / pool["n"] * 100
        print(f"  {m:<16} pilot accuracy {acc:5.1f}%")


if __name__ == "__main__":
    _main()
