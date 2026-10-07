# =============================================================
# Routed runner — route first, then RoSE, one model on GPU at a time
# =============================================================
# The Stage-2 counterpart of pilot_runner.py. For every question in a
# dataset JSON it:
#
#   1. consults the kNN router (routing_pool.json from the pilot) to
#      pick a model — BEFORE any GPU work, so routing is CPU-only and
#      all decisions exist up front in routing_decisions.json;
#   2. runs the repo's real RoSE pipeline (rose_chartqapro.process_one)
#      with the routed model.
#
# By default questions are answered in DATASET ORDER — per question:
# look up the routed model, run RoSE with it, predict — with all routed
# models resident on the GPU (interleave=True; ~13 GB for three 4-bit
# VLMs, fine on a T4 x2). interleave=False instead groups questions by
# model, cheapest first, one VLM on the GPU at a time (for single-GPU
# sessions); predictions are identical either way. Each model streams
# its OWN experience pool, exactly as the pilot and the 7B sweep did.
#
# Checkpoints after every question; on restart answered ids are
# skipped, so a run survives Kaggle's session cap.
#
#   python -m routing.routed_runner \
#       --dataset   /kaggle/temp/figureqa_factoid.json \
#       --pool      results/pilot/routing_pool.json \
#       --decisions results/routed/routing_decisions.json \
#       --out       results/routed/routed_figureqa.json
# =============================================================

import argparse
import gc
import json
import time
from pathlib import Path

try:
    from .config import ESCALATION_ORDER, K_NEIGHBOURS, MODELS, TAU
    from .pool import load_pool
    from .router import Router
    from .pilot_runner import load_vlm, _env_versions
except ImportError:                 # flat / notebook use
    from config import ESCALATION_ORDER, K_NEIGHBOURS, MODELS, TAU
    from pool import load_pool
    from router import Router
    from pilot_runner import load_vlm, _env_versions

try:
    from .. import rose_chartqapro as rose
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import rose_chartqapro as rose


# The pilot showed Qwen3.5-4B's verbose reasoning blows through the
# sweep's 300-token budget (135/150 answers truncated before "Final
# Answer", 80% accuracy on the 15 that fit). Models listed here get a
# larger budget; everything else keeps rose.MAX_NEW_TOKENS.
TOKEN_BUDGET = {"qwen3.5-4b": 768}


def route_dataset(pool_path, dataset_records, k: int = K_NEIGHBOURS,
                  tau: float = TAU, embed_fn=None, out_path=None) -> dict:
    """
    CPU step: one routing decision per dataset record.

    Questions whose id is IN the pool (re-running pilot questions) are
    routed with themselves excluded from the neighbourhood, so the
    decision stays honest (leave-one-out).

    Returns {"decisions": {id: {model, scores, neighbours}}, ...meta}.
    """
    pool = load_pool(pool_path)
    router = Router(pool, k=k, tau=tau, embed_fn=embed_fn)
    pool_ids = {e["id"] for e in pool["entries"]}

    if embed_fn is None:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(pool["embed_model"])
        embed_fn = lambda texts: st.encode(list(texts), convert_to_numpy=True)  # noqa: E731
        router._embed_fn = embed_fn

    questions = [r.get("question", "") for r in dataset_records]
    embs = embed_fn(questions)

    decisions = {}
    for rec, emb in zip(dataset_records, embs):
        qid = str(rec.get("id"))
        d = router.route(emb, exclude_id=qid if qid in pool_ids else None)
        decisions[qid] = d

    share = {m: 0 for m in router.order}
    for d in decisions.values():
        share[d["model"]] += 1

    blob = {
        "pool":      str(pool_path),
        "k": k, "tau": tau,
        "n": len(decisions),
        "share":     {m: share[m] / max(len(decisions), 1) for m in share},
        "decisions": decisions,
    }
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(blob, indent=2))
    for m in router.order:
        print(f"  routed → {m:<16} {share[m]:4d}  ({share[m] / max(len(decisions), 1) * 100:5.1f}%)")
    return blob


def _free_gpu(*objs):
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def run_routed(dataset_path, decisions, out_path,
               quant: str = "nf4", order=None, token_budget=None,
               embedder=None, limit=None, stop_after_s=None,
               interleave: bool = True):
    """
    GPU step: answer every question with its routed model via RoSE.

    decisions    : dict from route_dataset (or a path to its JSON).
    limit        : dry run — only the first N dataset records overall.
    stop_after_s : graceful session budget. After this many seconds the
                   run stops BETWEEN questions, writes its outputs, and
                   returns — so a Kaggle "Save & Run All" version
                   finishes inside the session cap and PUBLISHES the
                   checkpoint instead of being killed with nothing.
                   Resume the next session via the checkpoint.
    interleave   : True (default) answers in DATASET ORDER — per
                   question: look up the routed model, run RoSE with it,
                   predict — with every routed model resident on the GPU
                   (needs ~13 GB for all three 4-bit VLMs; a T4 x2 is
                   fine). A partial checkpoint is then an unbiased
                   prefix of the dataset. False groups questions by
                   model, cheapest first, one VLM on the GPU at a time —
                   for single-16GB-GPU sessions. The predictions are
                   IDENTICAL either way (each model sees its own
                   questions in the same relative order; routing was
                   decided per question before any GPU work).
    """
    t_start = time.time()
    out_path = Path(out_path)
    ckpt = out_path.with_name(out_path.stem + "_checkpoint.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not isinstance(decisions, dict):
        decisions = json.loads(Path(decisions).read_text())
    dec = decisions["decisions"] if "decisions" in decisions else decisions

    data = json.loads(Path(dataset_path).read_text())
    if limit:
        data = data[:limit]
    order = [m for m in (order or ESCALATION_ORDER)]
    token_budget = TOKEN_BUDGET if token_budget is None else token_budget

    missing = [str(d.get("id")) for d in data if str(d.get("id")) not in dec]
    if missing:
        raise ValueError(f"{len(missing)} dataset ids have no routing decision "
                         f"(first: {missing[:3]}) — rerun route_dataset on this dataset")

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
    clip_model = clip_processor = None
    if rose.ext("visual_hybrid_retrieval"):
        clip_model, clip_processor = rose.load_clip()

    base_tokens = rose.MAX_NEW_TOKENS
    revisions = {}
    out_of_time = False

    # One streaming experience pool PER MODEL, rebuilt from the
    # checkpoint — a model only ever learns from its own answers,
    # exactly as the pilot and the 7B sweep did.
    pools = {}

    def _pool_for(model_key):
        if model_key not in pools:
            pool = rose.ExperiencePool(embedder)
            for r in results:
                if r.get("model_key") != model_key or r.get("method") == "error":
                    continue
                pool.add(question=r["question"],
                         rationale=r.get("best_rationale") or r.get("prediction", ""),
                         answer=r.get("prediction", ""),
                         qtype=r.get("question_type", "factoid"),
                         uncertainty=r.get("uncertainty", 0.5),
                         complexity=r.get("complexity", 1.0),
                         clip_emb=None)
            pools[model_key] = pool
        return pools[model_key]

    def _answer(sample, model_key, model, processor):
        qid = str(sample.get("id"))
        pool = _pool_for(model_key)
        rose.MAX_NEW_TOKENS = token_budget.get(model_key, base_tokens)
        t0 = time.time()
        try:
            r = rose.process_one(sample, model, processor, pool,
                                 clip_model=clip_model,
                                 clip_processor=clip_processor)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            r = {"id": qid, "question": sample.get("question", ""),
                 "question_type": sample.get("question_type", "factoid"),
                 "ground_truth": sample.get("answer", ""),
                 "prediction": "ERROR", "is_correct": False,
                 "error": str(exc), "method": "error"}
        finally:
            rose.MAX_NEW_TOKENS = base_tokens
        r["latency_s"] = round(time.time() - t0, 2)
        r["model_key"] = model_key
        r["routing"] = {k: dec[qid][k] for k in ("model", "scores") if k in dec[qid]}
        results.append(r)
        done.add(qid)
        ckpt.write_text(json.dumps(results, indent=2))

        mark = "✓" if r.get("is_correct") else "✗"
        print(f"[{len(done):03d}/{len(data)}] {mark} {model_key} "
              f"pool={pool.size():<3} {r['latency_s']:6.1f}s "
              f"pred={str(r.get('prediction'))[:28]}", flush=True)

    def _budget_up():
        if stop_after_s is not None and time.time() - t_start > stop_after_s:
            print(f"⏱  session budget ({stop_after_s / 3600:.1f} h) reached — "
                  f"stopping cleanly with {len(done)}/{len(data)} answered; "
                  "resume the next session from the checkpoint")
            return True
        return False

    pending = [s for s in data if str(s.get("id")) not in done]
    needed = [m for m in order
              if any(dec[str(s.get("id"))]["model"] == m for s in pending)]

    if interleave:
        # ── dataset order: route → reason → predict, per question ──
        # Every routed model is loaded up front and stays resident
        # (three 4-bit VLMs ≈ 13 GB — fine on a T4 x2's 32 GB, tight on
        # a single 16 GB GPU, where grouped mode is the safer choice).
        try:
            import torch
            if torch.cuda.is_available() and torch.cuda.device_count() < 2 \
                    and len(needed) > 2:
                print("⚠ one GPU visible: three resident 4-bit VLMs are tight "
                      "on 16 GB — if loading OOMs, use interleave=False")
        except Exception:
            pass
        loaded = {}
        for m in needed:
            model, processor, revision = load_vlm(m, quant=quant)
            loaded[m] = (model, processor)
            revisions[m] = revision
        print(f"\n━━ {len(pending)} question(s), dataset order, "
              f"models resident: {', '.join(needed)} ━━")
        for sample in pending:
            if _budget_up():
                out_of_time = True
                break
            m = dec[str(sample.get("id"))]["model"]
            _answer(sample, m, *loaded[m])
        for m, (model, processor) in loaded.items():
            _free_gpu(model, processor)
    else:
        # ── grouped: cheapest model first, one VLM on the GPU at a time ──
        for model_key in order:
            if out_of_time:
                break
            share = [s for s in pending
                     if dec[str(s.get("id"))]["model"] == model_key]
            if not share:
                continue
            print(f"\n━━ {model_key}: {len(share)} question(s) ━━")
            model, processor, revision = load_vlm(model_key, quant=quant)
            revisions[model_key] = revision
            for sample in share:
                if _budget_up():
                    out_of_time = True
                    break
                _answer(sample, model_key, model, processor)
            _free_gpu(model, processor)

    out_path.write_text(json.dumps(results, indent=2))

    ok = [r for r in results if r.get("method") != "error"]
    acc = sum(r.get("is_correct", False) for r in ok) / max(len(ok), 1) * 100
    lat = [r["latency_s"] for r in ok if r.get("latency_s")]
    params = [MODELS[r["model_key"]]["params_b"] for r in ok if r.get("model_key") in MODELS]
    meta = {
        "dataset":        str(dataset_path),
        "n":              len(results),
        "n_total":        len(data),
        "complete":       len(done) >= len(data),
        "interleave":     interleave,
        "accuracy_run":   round(acc, 2),
        "mean_latency_s": round(sum(lat) / len(lat), 2) if lat else None,
        "mean_params_b":  round(sum(params) / len(params), 2) if params else None,
        "quant":          quant,
        "revisions":      {m: revisions.get(m) for m in revisions},
        "token_budget":   token_budget,
        "routing":        {"k": decisions.get("k"), "tau": decisions.get("tau"),
                           "pool": decisions.get("pool"),
                           "share": decisions.get("share")},
        "per_model": {
            m: {
                "n":        sum(1 for r in ok if r.get("model_key") == m),
                "accuracy": round(
                    sum(r.get("is_correct", False) for r in ok if r.get("model_key") == m)
                    / max(sum(1 for r in ok if r.get("model_key") == m), 1) * 100, 2),
            } for m in order
        },
        "rose_config": {
            "m_paths": rose.M_PATHS, "k_demonstrations": rose.K_DEMONSTRATIONS,
            "lambda": rose.LAMBDA, "temperature": rose.TEMPERATURE,
            "extensions": rose.active_extensions(),
        },
        "env":            _env_versions(),
        "finished_at":    time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\n✓ {len(results)} results → {out_path}")
    print(f"✓ meta → {meta_path}")
    return results


def _main():
    ap = argparse.ArgumentParser(description="Route every question, then run RoSE with the routed model")
    ap.add_argument("--dataset", required=True, help="dataset JSON (id/question/answer/image rows)")
    ap.add_argument("--pool", required=True, help="routing_pool.json from the pilot")
    ap.add_argument("--decisions", required=True, help="where to write/read routing_decisions.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--quant", default="nf4", choices=["nf4", "fp4"])
    ap.add_argument("--k", type=int, default=K_NEIGHBOURS)
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    data = json.loads(Path(args.dataset).read_text())
    if Path(args.decisions).exists():
        decisions = json.loads(Path(args.decisions).read_text())
        print(f"↩  reusing {args.decisions}")
    else:
        decisions = route_dataset(args.pool, data, k=args.k, tau=args.tau,
                                  out_path=args.decisions)
    run_routed(args.dataset, decisions, args.out,
               quant=args.quant, limit=args.limit)


if __name__ == "__main__":
    _main()
