# =============================================================
# Pilot runner — ONE model over the pilot sample  (GPU)
# =============================================================
# The only GPU file in routing/. Runs the repo's real pipeline
# (rose_chartqapro.process_one — prompts, pool, m paths, voting,
# scoring) with a 4-bit VLM chosen from routing/config.py, over the
# questions named in sample_ids.json, and writes a results file in the
# exact shape the sweep notebooks produce — plus, per question:
#
#   latency_s   wall-clock seconds for the whole question
#               (description pass + m paths), which Stage 2's time
#               budget is computed from.
#
# Its meta file records the RESOLVED model revision and the installed
# library versions, so "which exact versions did the pilot use?" is
# always answerable from the outputs alone.
#
#   python -m routing.pilot_runner --model gemma3-4b \
#       --dataset /kaggle/temp/chartqapro_factoid.json \
#       --sample  results/pilot/sample_ids.json \
#       --out     results/pilot/pilot_gemma3-4b.json
#
# Run it once per model (the T4 fits one 4-bit VLM at a time). Each
# model streams its own experience pool over the sample, exactly as the
# 7B sweep did — so hits are comparable across models.
# =============================================================

import argparse
import json
import time
from pathlib import Path

try:
    from .config import MODELS
except ImportError:                 # flat / notebook use
    from config import MODELS

try:
    from .. import rose_chartqapro as rose
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import rose_chartqapro as rose


def load_vlm(model_key: str, quant: str = "nf4"):
    """
    Load any of the registry's VLMs 4-bit on the GPU, through the generic
    transformers auto-classes so Gemma 3 / Qwen3.5 / Qwen2.5-VL all take
    the same path. Returns (model, processor, resolved_revision).

    NOTE on quantization: the final-week 7B sweep used rose_chartqapro's
    BitsAndBytesConfig, which does not set bnb_4bit_quant_type and so ran
    fp4 (the bitsandbytes default). The pilot spec says NF4 for all three
    models — that is this function's default. To reproduce the 7B sweep's
    exact numbers instead, pass --quant fp4.
    """
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig

    spec = MODELS[model_key]
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    common = dict(revision=spec["revision"], trust_remote_code=True)

    processor = AutoProcessor.from_pretrained(
        spec["hf_id"],
        min_pixels=rose.MIN_PIXELS, max_pixels=rose.MAX_PIXELS,
        **common,
    )

    try:
        from transformers import AutoModelForImageTextToText as _Auto
    except ImportError:             # older transformers
        from transformers import AutoModelForVision2Seq as _Auto
    model = _Auto.from_pretrained(
        spec["hf_id"], quantization_config=bnb, device_map="auto", **common,
    )
    model.eval()

    revision = getattr(model.config, "_commit_hash", None) or spec["revision"]
    return model, processor, revision


def _env_versions():
    """Exact installed versions, for the meta file (and the pinned requirements)."""
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for pkg in ("torch", "transformers", "bitsandbytes", "accelerate",
                "sentence-transformers", "datasets"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def run_pilot(model_key: str, dataset_path, sample_path, out_path,
              checkpoint_path=None, quant: str = "nf4",
              model=None, processor=None, embedder=None,
              clip_model=None, clip_processor=None, limit=None):
    """
    Stream the sampled questions through RoSE with the given model.
    Checkpoints after EVERY question (pilot questions are expensive);
    on restart, ids already answered are skipped.

    Pass pre-loaded model/processor/embedder to avoid reloads between
    calls (e.g. a dry run followed by the real one in a notebook).
    """
    spec = MODELS[model_key]
    out_path = Path(out_path)
    ckpt = Path(checkpoint_path) if checkpoint_path else \
        out_path.with_name(out_path.stem + "_checkpoint.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    blob = json.loads(Path(sample_path).read_text())
    sample_ids = [str(i) for i in (blob["ids"] if isinstance(blob, dict) else blob)]
    data_all = json.loads(Path(dataset_path).read_text())
    by_id = {str(d.get("id")): d for d in data_all}
    missing = [i for i in sample_ids if i not in by_id]
    if missing:
        raise ValueError(
            f"{len(missing)} sample ids not in {dataset_path} (first: {missing[:3]}). "
            "The dataset json must be built the same way (same split, same order) "
            "as the one the sample was drawn from.")
    data = [by_id[i] for i in sample_ids]
    if limit:
        data = data[:limit]

    if model is None:
        print(f"loading {spec['hf_id']} (4-bit {quant})…")
        model, processor, revision = load_vlm(model_key, quant=quant)
    else:
        revision = getattr(model.config, "_commit_hash", None) or spec["revision"]
    if embedder is None:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(rose.EMBED_MODEL)
    if clip_model is None and rose.ext("visual_hybrid_retrieval"):
        clip_model, clip_processor = rose.load_clip()

    results = json.loads(ckpt.read_text()) if ckpt.exists() else []
    done = {str(r.get("id")) for r in results}
    if done:
        print(f"↩  resuming: {len(done)}/{len(data)} already answered")

    # Rebuild this model's streaming pool from what is already answered,
    # exactly as the sweep notebooks do on resume.
    pool = rose.ExperiencePool(embedder)
    for r in results:
        if r.get("method") == "error":
            continue
        pool.add(question=r["question"],
                 rationale=r.get("best_rationale") or r.get("prediction", ""),
                 answer=r.get("prediction", ""),
                 qtype=r.get("question_type", "factoid"),
                 uncertainty=r.get("uncertainty", 0.5),
                 complexity=r.get("complexity", 1.0),
                 clip_emb=None)

    for i, sample in enumerate(data):
        if str(sample.get("id")) in done:
            continue
        t0 = time.time()
        try:
            r = rose.process_one(sample, model, processor, pool,
                                 clip_model=clip_model, clip_processor=clip_processor)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            r = {"id": sample.get("id", ""), "question": sample.get("question", ""),
                 "question_type": sample.get("question_type", "factoid"),
                 "ground_truth": sample.get("answer", ""),
                 "prediction": "ERROR", "is_correct": False,
                 "error": str(exc), "method": "error"}
        r["latency_s"] = round(time.time() - t0, 2)
        r["model_key"] = model_key
        results.append(r)
        ckpt.write_text(json.dumps(results, indent=2))   # every question

        mark = "✓" if r.get("is_correct") else "✗"
        print(f"[{i + 1:03d}/{len(data)}] {mark} {model_key} "
              f"pool={pool.size():<3} {r['latency_s']:6.1f}s "
              f"pred={str(r.get('prediction'))[:28]}", flush=True)

    out_path.write_text(json.dumps(results, indent=2))

    ok = [r for r in results if r.get("method") != "error"]
    acc = sum(r.get("is_correct", False) for r in ok) / max(len(ok), 1) * 100
    lat = [r["latency_s"] for r in ok if r.get("latency_s")]
    meta = {
        "model_key":      model_key,
        "hf_id":          spec["hf_id"],
        "revision":       revision,
        "quant":          quant,
        "n":              len(results),
        "accuracy_run":   round(acc, 2),
        "mean_latency_s": round(sum(lat) / len(lat), 2) if lat else None,
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
    print(f"✓ meta (revision={str(revision)[:12]}, versions) → {meta_path}")
    return results


def _main():
    ap = argparse.ArgumentParser(description="Run one model over the pilot sample")
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--dataset", required=True,
                    help="chartqapro_factoid.json as built by the sweep notebook")
    ap.add_argument("--sample", required=True, help="sample_ids.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--quant", default="nf4", choices=["nf4", "fp4"])
    ap.add_argument("--limit", type=int, default=None,
                    help="dry run: only the first N sampled questions")
    args = ap.parse_args()
    run_pilot(args.model, args.dataset, args.sample, args.out,
              checkpoint_path=args.checkpoint, quant=args.quant, limit=args.limit)


if __name__ == "__main__":
    _main()
