# =============================================================
# Decomposition runner — two algorithms, checkpointed, budgeted
# =============================================================
# The decompose counterpart of routing/routed_runner.py. For every
# question in a dataset JSON it runs ONE of:
#
#   Algorithm 1: the 7B splits the question into two sub-queries;
#     Gemma 3 4B answers sub-query 1 and Qwen3.5-4B answers sub-query 2
#     (3 sampled paths each, agreement ratio as confidence); the original
#     RoSE pipeline then answers the ORIGINAL question on the 7B with the
#     sub-answers prepended to the reasoning prompt as an evidence block.
#
#   Algorithm 2: Gemma and Qwen3.5 each PROPOSE a decomposition; an
#     embedding vote merges agreeing proposals or picks the one closer to
#     the question; the full RoSE pipeline runs on the 7B with the chosen
#     sub-queries inserted as "answer these in order" steps. No small
#     model answers anything.
#
# Both algorithms keep every RoSE hyperparameter unchanged and stream a
# SINGLE experience pool for the 7B, exactly as the pre-routing pipeline
# did. There is NO routing: this file never imports chartqapro.routing
# (tests/test_decompose.py enforces that).
#
# Mirrors routed_runner.py operationally: all three models resident
# (interleave=True), checkpoint after every question, errored rows
# retried on resume, graceful stop_after_s session budget, gated-access
# check up front, per-question CSV + per-stage summary CSV + meta JSON
# with resolved revisions, library versions and the algorithm config.
#
#   python -m decompose.decomp_runner --algorithm 1 \
#       --dataset chartqapro --data /kaggle/temp/chartqapro_factoid.json \
#       --n 250 --out results/decompose/decomp_alg1.json
# =============================================================

import argparse
import contextlib
import csv
import gc
import json
import sys
import time
from pathlib import Path

try:
    from .decomposer import decompose
    from .models import (DECOMPOSER_MODEL, FINAL_MODEL, MODELS, SMALL_MODELS,
                         TOKEN_BUDGET, check_gated_access, env_versions,
                         load_vlm)
    from .subanswer import N_SUB_PATHS, SUB_TEMPERATURE, answer_subquery
    from . import prompts, vote
except ImportError:                 # flat / notebook use
    from decomposer import decompose
    from models import (DECOMPOSER_MODEL, FINAL_MODEL, MODELS, SMALL_MODELS,
                        TOKEN_BUDGET, check_gated_access, env_versions,
                        load_vlm)
    from subanswer import N_SUB_PATHS, SUB_TEMPERATURE, answer_subquery
    import prompts
    import vote

try:
    from .. import rose_chartqapro as rose
    from .. import scoring
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import rose_chartqapro as rose
    import scoring


# Decomposition output is a two-string JSON list — short. 256 tokens
# leaves room for a model that chats before the JSON (the parser copes).
DECOMP_MAX_TOKENS = 256

# Who does what. Algorithm 1 answers sub-query i with SMALL_MODELS[i].
SUB_ASSIGNMENT = list(enumerate(SMALL_MODELS))   # [(0, gemma), (1, qwen3.5)]


# ─────────────────────────────────────────────────────────────
# Prompt injection — the ONLY touch on the RoSE pipeline
# ─────────────────────────────────────────────────────────────

@contextlib.contextmanager
def inject_block(block: str):
    """
    Prepend `block` to every RoSE reasoning prompt for the duration of
    one process_one call, by wrapping rose.zero_shot_prompt /
    rose.few_shot_prompt. Retrieval, the pool, the m paths and the
    confidence-weighted vote are untouched — the final answer still comes
    from RoSE's own vote.
    """
    if not block:
        yield
        return
    zs, fs = rose.zero_shot_prompt, rose.few_shot_prompt
    rose.zero_shot_prompt = lambda *a, **k: block + zs(*a, **k)
    rose.few_shot_prompt = lambda *a, **k: block + fs(*a, **k)
    try:
        yield
    finally:
        rose.zero_shot_prompt, rose.few_shot_prompt = zs, fs


# ─────────────────────────────────────────────────────────────
# Model plumbing
# ─────────────────────────────────────────────────────────────

def load_all_models(quant: str = "nf4"):
    """
    Load the decomposer, both small models and the final 7B, all resident
    (three 4-bit VLMs ≈ 13 GB — needs a T4 x2). Returns
    ({model_key: (model, processor)}, {model_key: resolved_revision}).
    """
    keys = []
    for k in [DECOMPOSER_MODEL, *SMALL_MODELS, FINAL_MODEL]:
        if k not in keys:
            keys.append(k)
    loaded, revisions = {}, {}
    for k in keys:
        print(f"loading {MODELS[k]['hf_id']} (4-bit {quant})…")
        model, processor, revision = load_vlm(
            k, quant=quant,
            min_pixels=rose.MIN_PIXELS, max_pixels=rose.MAX_PIXELS)
        loaded[k] = (model, processor)
        revisions[k] = revision
    return loaded, revisions


def _free_gpu(loaded):
    for model, processor in loaded.values():
        try:
            del model, processor
        except Exception:
            pass
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def _call(loaded, model_key, image, prompt, temperature, max_new_tokens=None):
    """One generation on `model_key`, honouring its token budget."""
    model, processor = loaded[model_key]
    tokens = max_new_tokens if max_new_tokens is not None \
        else TOKEN_BUDGET.get(model_key, rose.MAX_NEW_TOKENS)
    return rose.call_vlm(model, processor, image, prompt,
                         temperature=temperature, max_new_tokens=tokens)


def _params_b(algorithm: int) -> float:
    """
    Parameter-billions engaged per question: the sum over every model
    STAGE the algorithm invokes (a model used twice counts twice) —
    a cost proxy comparable to the routed run's per-question params.
    """
    if algorithm == 1:
        stages = [DECOMPOSER_MODEL, *SMALL_MODELS, FINAL_MODEL]
    else:
        stages = [*SMALL_MODELS, FINAL_MODEL]
    return round(sum(MODELS[m]["params_b"] for m in stages), 2)


# ─────────────────────────────────────────────────────────────
# Per-question logic
# ─────────────────────────────────────────────────────────────

def _rose_with_block(sample, loaded, pool, block, clip_model, clip_processor):
    model, processor = loaded[FINAL_MODEL]
    with inject_block(block):
        return rose.process_one(sample, model, processor, pool,
                                clip_model=clip_model,
                                clip_processor=clip_processor)


def _run_alg1(sample, loaded, pool, clip_model, clip_processor):
    """Decompose (7B) → parallel sub-answers (small models) → evidence → RoSE."""
    question = sample["question"]
    image = rose.load_image(sample.get("image", ""))
    stage_lat = {}

    t0 = time.time()
    sub_queries, fell_back, _raw = decompose(
        question,
        lambda p: _call(loaded, DECOMPOSER_MODEL, image, p, 0.0,
                        max_new_tokens=DECOMP_MAX_TOKENS))
    stage_lat["decompose"] = round(time.time() - t0, 2)

    subs = []
    for idx, model_key in SUB_ASSIGNMENT:
        t0 = time.time()
        res = answer_subquery(
            sub_queries[idx],
            lambda p, temp: _call(loaded, model_key, image, p, temp))
        stage_lat[f"sub_{model_key}"] = round(time.time() - t0, 2)
        subs.append({"sub_query": sub_queries[idx], "model": model_key,
                     "answer": res["answer"], "confidence": res["confidence"]})

    evidence = vote.answer_evidence(subs)

    t0 = time.time()
    r = _rose_with_block(sample, loaded, pool, evidence + "\n\n",
                         clip_model, clip_processor)
    stage_lat["rose_final"] = round(time.time() - t0, 2)

    r.update({
        "algorithm":       1,
        "sub_queries":     sub_queries,
        "sub_answers":     [s["answer"] for s in subs],
        "sub_confidences": [s["confidence"] for s in subs],
        "sub_models":      [s["model"] for s in subs],
        "decomp_fallback": fell_back,
        "vote_rule":       "",            # Algorithm 2 only
        "evidence_block":  evidence,
        "stage_latency_s": stage_lat,
    })
    return r


def _run_alg2(sample, loaded, pool, clip_model, clip_processor, embed_fn):
    """Both small models decompose → embedding vote → RoSE with steps."""
    question = sample["question"]
    image = rose.load_image(sample.get("image", ""))
    stage_lat = {}

    proposals, fallbacks = [], {}
    for model_key in SMALL_MODELS:
        t0 = time.time()
        sq, fell_back, _raw = decompose(
            question,
            lambda p: _call(loaded, model_key, image, p, 0.0,
                            max_new_tokens=DECOMP_MAX_TOKENS))
        stage_lat[f"decompose_{model_key}"] = round(time.time() - t0, 2)
        proposals.append((model_key, sq))
        fallbacks[model_key] = fell_back

    t0 = time.time()
    choice = vote.choose_decomposition(proposals, question, embed_fn)
    stage_lat["choose"] = round(time.time() - t0, 2)

    t0 = time.time()
    r = _rose_with_block(sample, loaded, pool,
                         prompts.subquery_steps_block(choice["sub_queries"]),
                         clip_model, clip_processor)
    stage_lat["rose_final"] = round(time.time() - t0, 2)

    r.update({
        "algorithm":       2,
        "sub_queries":     choice["sub_queries"],
        "sub_answers":     [],              # no small model answers anything
        "sub_confidences": [],
        "sub_models":      [],
        "decomp_fallback": fallbacks,
        "vote_rule":       choice["vote_rule"],
        "proposals":       {k: sq for k, sq in proposals},
        "vote":            {k: choice[k] for k in
                            ("rule", "picked_model", "matching", "pair_sims",
                             "mean_sim_to_question", "threshold")},
        "stage_latency_s": stage_lat,
    })
    return r


# ─────────────────────────────────────────────────────────────
# CSV export — per question and per stage
# ─────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    "id", "question_type", "algorithm", "question", "ground_truth",
    "sub_queries", "sub_answers", "sub_confidences", "sub_models",
    "vote_rule", "decomp_fallback",
    "final_answer", "prediction_norm", "is_correct", "is_correct_run",
    "method", "uncertainty", "complexity", "agreement",
    "n_demos", "pool_size", "latency_s", "params_b_used",
    "stage_latency_s", "error",
]


def _rescore(r) -> bool:
    """The ONLY correctness verdict: scoring.is_correct on the final answer."""
    if r.get("method") == "error":
        return False
    return scoring.is_correct(r.get("prediction", ""), r.get("ground_truth", ""),
                              str(r.get("question_type", "factoid")).lower())


def write_csv(results, out_path: Path):
    """One row per question, rescored with scoring.py as it goes."""
    with Path(out_path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in results:
            gold = r.get("ground_truth", "")
            w.writerow({
                "id":              r.get("id", ""),
                "question_type":   r.get("question_type", ""),
                "algorithm":       r.get("algorithm", ""),
                "question":        r.get("question", ""),
                "ground_truth":    "|".join(map(str, gold))
                                   if isinstance(gold, (list, tuple)) else gold,
                "sub_queries":     json.dumps(r.get("sub_queries", []),
                                              ensure_ascii=False),
                "sub_answers":     json.dumps(r.get("sub_answers", []),
                                              ensure_ascii=False),
                "sub_confidences": json.dumps(r.get("sub_confidences", [])),
                "sub_models":      json.dumps(r.get("sub_models", [])),
                "vote_rule":       r.get("vote_rule", ""),
                "decomp_fallback": json.dumps(r.get("decomp_fallback", "")),
                "final_answer":    r.get("prediction", ""),
                "prediction_norm": r.get("prediction_norm", ""),
                "is_correct":      _rescore(r),
                "is_correct_run":  r.get("is_correct", ""),
                "method":          r.get("method", ""),
                "uncertainty":     r.get("uncertainty", ""),
                "complexity":      r.get("complexity", ""),
                "agreement":       r.get("agreement", ""),
                "n_demos":         r.get("n_demos", ""),
                "pool_size":       r.get("pool_size", ""),
                "latency_s":       r.get("latency_s", ""),
                "params_b_used":   r.get("params_b_used", ""),
                "stage_latency_s": json.dumps(r.get("stage_latency_s", {})),
                "error":           r.get("error", ""),
            })


def write_stage_summary(results, out_path: Path):
    """
    One row per pipeline stage: how long it takes and — where it makes
    sense — how well it does. Sub-answer accuracy is each sub-answer
    scored against the ORIGINAL gold with scoring.py (that is exactly the
    signal the (sub right, final wrong) analysis needs); decompose rows
    report the JSON-fallback rate; rose_final reports final accuracy.
    """
    ok = [r for r in results if r.get("method") != "error"]
    rows = []
    stage_keys = []
    for r in ok:
        for k in (r.get("stage_latency_s") or {}):
            if k not in stage_keys:
                stage_keys.append(k)

    for stage in stage_keys:
        lats = [r["stage_latency_s"][stage] for r in ok
                if stage in (r.get("stage_latency_s") or {})]
        row = {"stage": stage, "n": len(lats),
               "mean_latency_s": round(sum(lats) / max(len(lats), 1), 2),
               "mean_confidence": "", "accuracy_pct": "", "fallback_rate": ""}

        if stage.startswith("sub_"):
            model_key = stage[len("sub_"):]
            confs, hits, n = [], 0, 0
            for r in ok:
                for m, a, c in zip(r.get("sub_models", []),
                                   r.get("sub_answers", []),
                                   r.get("sub_confidences", [])):
                    if m == model_key:
                        confs.append(c)
                        n += 1
                        if scoring.is_correct(a, r.get("ground_truth", ""),
                                              "factoid"):
                            hits += 1
            if confs:
                row["mean_confidence"] = round(sum(confs) / len(confs), 4)
            if n:
                row["accuracy_pct"] = round(hits / n * 100, 2)
        elif stage.startswith("decompose"):
            model_key = stage[len("decompose_"):] if "_" in stage else None
            fb = []
            for r in ok:
                f = r.get("decomp_fallback")
                if isinstance(f, dict):
                    if model_key in f:
                        fb.append(bool(f[model_key]))
                elif f is not None:
                    fb.append(bool(f))
            if fb:
                row["fallback_rate"] = round(sum(fb) / len(fb), 4)
        elif stage == "rose_final":
            row["accuracy_pct"] = round(
                sum(_rescore(r) for r in ok) / max(len(ok), 1) * 100, 2)
        rows.append(row)

    with Path(out_path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stage", "n", "mean_latency_s",
                                          "mean_confidence", "accuracy_pct",
                                          "fallback_rate"])
        w.writeheader()
        w.writerows(rows)
    return rows


# ─────────────────────────────────────────────────────────────
# Main run loop — mirrors routed_runner.run_routed
# ─────────────────────────────────────────────────────────────

def run_decomp(algorithm: int, dataset_path, out_path, n: int = None,
               quant: str = "nf4", embedder=None, loaded=None, revisions=None,
               limit=None, stop_after_s=None, check_access: bool = True,
               interleave: bool = True):
    """
    Run one decomposition algorithm over the first `n` dataset records.

    algorithm    : 1 or 2 (see the module docstring).
    n            : evaluate the FIRST n records in dataset order.
    loaded       : pre-loaded {model_key: (model, processor)} — pass the
                   dict from load_all_models() to share residents between
                   the Alg 1 and Alg 2 runs of one session.
    limit        : dry run — only the first N pending questions.
    stop_after_s : graceful session budget; the run stops BETWEEN
                   questions, writes every output, and returns, so a
                   headless Kaggle version publishes a resumable prefix.
    interleave   : kept for parity with the routed runner; only True is
                   supported — every question needs all three models.
    """
    if algorithm not in (1, 2):
        raise ValueError(f"algorithm must be 1 or 2, got {algorithm!r}")
    if not interleave:
        raise ValueError("interleave=False is not supported: both decompose "
                         "algorithms need all three models per question")

    t_start = time.time()
    out_path = Path(out_path)
    ckpt = out_path.with_name(out_path.stem + "_checkpoint.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = json.loads(Path(dataset_path).read_text())
    if n:
        data = data[:n]

    results = json.loads(ckpt.read_text()) if ckpt.exists() else []
    n_err = sum(1 for r in results if r.get("method") == "error")
    if n_err:
        print(f"↩  dropping {n_err} error row(s) from the checkpoint — will retry")
        results = [r for r in results if r.get("method") != "error"]
    done = {str(r.get("id")) for r in results}
    if done:
        print(f"↩  resuming: {len(done)}/{len(data)} already answered")

    if embedder is None:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(rose.EMBED_MODEL)
    embed_fn = lambda texts: embedder.encode(list(texts),                 # noqa: E731
                                             convert_to_numpy=True)
    clip_model = clip_processor = None
    if rose.ext("visual_hybrid_retrieval"):
        clip_model, clip_processor = rose.load_clip()

    own_models = loaded is None
    if own_models:
        if check_access:
            check_gated_access()
        loaded, revisions = load_all_models(quant)
    revisions = revisions or {}

    # A SINGLE experience pool for the 7B, streamed over the run and
    # rebuilt from the checkpoint — exactly as the pre-routing pipeline.
    pool = rose.ExperiencePool(embedder)
    for r in results:
        pool.add(question=r["question"],
                 rationale=r.get("best_rationale") or r.get("prediction", ""),
                 answer=r.get("prediction", ""),
                 qtype=r.get("question_type", "factoid"),
                 uncertainty=r.get("uncertainty", 0.5),
                 complexity=r.get("complexity", 1.0),
                 clip_emb=None)

    params_b = _params_b(algorithm)
    pending = [s for s in data if str(s.get("id")) not in done]
    if limit:
        pending = pending[:limit]
    out_of_time = False

    print(f"\n━━ Algorithm {algorithm}: {len(pending)} question(s), "
          f"models resident: {', '.join(loaded)} ━━")
    for sample in pending:
        if stop_after_s is not None and time.time() - t_start > stop_after_s:
            print(f"⏱  session budget ({stop_after_s / 3600:.1f} h) reached — "
                  f"stopping cleanly with {len(done)}/{len(data)} answered; "
                  "resume the next session from the checkpoint")
            out_of_time = True
            break

        qid = str(sample.get("id"))
        t0 = time.time()
        try:
            if algorithm == 1:
                r = _run_alg1(sample, loaded, pool, clip_model, clip_processor)
            else:
                r = _run_alg2(sample, loaded, pool, clip_model, clip_processor,
                              embed_fn)
            r["params_b_used"] = params_b
        except Exception as exc:
            import traceback
            traceback.print_exc()
            r = {"id": qid, "question": sample.get("question", ""),
                 "question_type": sample.get("question_type", "factoid"),
                 "ground_truth": sample.get("answer", ""),
                 "prediction": "ERROR", "is_correct": False,
                 "algorithm": algorithm, "error": str(exc), "method": "error"}
        r["latency_s"] = round(time.time() - t0, 2)
        r["model_key"] = FINAL_MODEL
        results.append(r)
        done.add(qid)
        ckpt.write_text(json.dumps(results, indent=2))    # every question

        mark = "✓" if r.get("is_correct") else "✗"
        print(f"[{len(done):03d}/{len(data)}] {mark} alg{algorithm} "
              f"pool={pool.size():<3} {r['latency_s']:6.1f}s "
              f"rule={r.get('vote_rule') or '—':<16} "
              f"pred={str(r.get('prediction'))[:28]}", flush=True)

    if own_models:
        _free_gpu(loaded)

    out_path.write_text(json.dumps(results, indent=2))

    ok = [r for r in results if r.get("method") != "error"]
    acc = sum(_rescore(r) for r in ok) / max(len(ok), 1) * 100
    lat = [r["latency_s"] for r in ok if r.get("latency_s")]
    rules = {}
    for r in ok:
        if r.get("vote_rule"):
            rules[r["vote_rule"]] = rules.get(r["vote_rule"], 0) + 1

    csv_path = out_path.with_suffix(".csv")
    write_csv(results, csv_path)
    stages_path = out_path.with_name(out_path.stem + "_stages.csv")
    write_stage_summary(results, stages_path)

    meta = {
        "algorithm":        algorithm,
        "dataset":          str(dataset_path),
        "n":                len(results),
        "n_total":          len(data),
        "complete":         len(done) >= len(data) and not out_of_time,
        "accuracy_rescored": round(acc, 2),
        "mean_latency_s":   round(sum(lat) / len(lat), 2) if lat else None,
        "params_b_used":    params_b,
        "vote_rules":       rules,
        "quant":            quant,
        "revisions":        {m: revisions.get(m) for m in loaded},
        "token_budget":     TOKEN_BUDGET,
        "algorithm_config": {
            "decomposer_model":  DECOMPOSER_MODEL if algorithm == 1
                                 else SMALL_MODELS,
            "final_model":       FINAL_MODEL,
            "sub_assignment":    {m: i + 1 for i, m in SUB_ASSIGNMENT}
                                 if algorithm == 1 else None,
            "n_sub_paths":       N_SUB_PATHS if algorithm == 1 else None,
            "sub_temperature":   SUB_TEMPERATURE if algorithm == 1 else None,
            "agree_threshold":   vote.AGREE_THRESHOLD if algorithm == 2 else None,
            "decomp_max_tokens": DECOMP_MAX_TOKENS,
            "embed_model":       rose.EMBED_MODEL,
        },
        "rose_config": {
            "m_paths": rose.M_PATHS, "k_demonstrations": rose.K_DEMONSTRATIONS,
            "lambda": rose.LAMBDA, "temperature": rose.TEMPERATURE,
            "max_new_tokens": rose.MAX_NEW_TOKENS,
            "max_pixels": rose.MAX_PIXELS,     # fast notebooks override this
            "extensions": rose.active_extensions(),
        },
        "env":              env_versions(),
        "finished_at":      time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\n✓ {len(results)} results → {out_path}")
    print(f"✓ per-question CSV → {csv_path}")
    print(f"✓ per-stage CSV → {stages_path}")
    print(f"✓ meta → {meta_path}")
    if lat:
        print(f"  rescored accuracy {acc:.1f}%  "
              f"mean latency {sum(lat) / len(lat):.1f}s/q")
    return results


def _main():
    ap = argparse.ArgumentParser(
        description="Evaluate one query-decomposition algorithm with RoSE")
    ap.add_argument("--algorithm", type=int, required=True, choices=[1, 2])
    ap.add_argument("--dataset", default="chartqapro", choices=["chartqapro"],
                    help="dataset key — names the default output files")
    ap.add_argument("--data", required=True,
                    help="dataset JSON (id/question/answer/image rows), e.g. "
                         "the first-250 factoid file the notebook builds")
    ap.add_argument("--n", type=int, default=250,
                    help="first N dataset records (default 250)")
    ap.add_argument("--out", default=None,
                    help="default: results/decompose/decomp_alg<A>.json")
    ap.add_argument("--quant", default="nf4", choices=["nf4", "fp4"])
    ap.add_argument("--limit", type=int, default=None,
                    help="dry run: only the first N pending questions")
    ap.add_argument("--budget-h", type=float, default=None,
                    help="graceful session budget in hours")
    args = ap.parse_args()

    out = args.out or f"results/decompose/decomp_alg{args.algorithm}.json"
    run_decomp(args.algorithm, args.data, out, n=args.n, quant=args.quant,
               limit=args.limit,
               stop_after_s=args.budget_h * 3600 if args.budget_h else None)


if __name__ == "__main__":
    _main()
