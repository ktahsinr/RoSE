# =============================================================
# Model registry + loader for the decompose arms — NO routing import
# =============================================================
# This deliberately DUPLICATES the registry in routing/config.py and the
# loader in routing/pilot_runner.py instead of importing them: the
# decompose arms are specified as routing-free (no kNN router, no
# routing pool, no escalation), and tests/test_decompose.py asserts that
# importing this package never pulls in chartqapro.routing. If a value
# here drifts from routing/config.py, every run's meta JSON records the
# RESOLVED revision and versions, so finished runs stay self-describing.
# =============================================================

MODELS = {
    "gemma3-4b": {
        "hf_id":    "google/gemma-3-4b-it",
        "revision": "093f9f388b31de276ce2de164bdc2081324b9767",
        "params_b": 4.3,
        "label":    "Gemma 3 4B-it",
        # Gemma 3 activations overflow float16 (NaN logits → CUDA assert
        # under sampling); it computes in bf16 where supported, else fp32.
        "fp16_safe": False,
    },
    "qwen3.5-4b": {
        "hf_id":    "Qwen/Qwen3.5-4B",
        "revision": None,
        "params_b": 4.0,
        "label":    "Qwen3.5-4B (VL)",
    },
    "qwen2.5-vl-7b": {
        "hf_id":    "Qwen/Qwen2.5-VL-7B-Instruct",
        "revision": None,
        "params_b": 8.3,
        "label":    "Qwen2.5-VL-7B-Instruct",
    },
}

# Role assignment (fixed — there is no routing in these arms).
DECOMPOSER_MODEL = "qwen2.5-vl-7b"   # Alg 1: splits the question
FINAL_MODEL      = "qwen2.5-vl-7b"   # both: runs the original RoSE pipeline
SMALL_MODELS     = ["gemma3-4b", "qwen3.5-4b"]   # Alg 1: answer sub-query 1/2
                                                  # Alg 2: propose decompositions

# Qwen3.5-4B's verbose reasoning blows through the sweep's 300-token
# budget (the pilot measured 135/150 answers truncated before "Final
# Answer"); it gets a larger budget, everything else keeps the RoSE
# default. Same values as the routed runner, duplicated for isolation.
TOKEN_BUDGET = {"qwen3.5-4b": 768}


def load_vlm(model_key: str, quant: str = "nf4", min_pixels=None, max_pixels=None):
    """
    Load one of the registry's VLMs 4-bit on the GPU through the generic
    transformers auto-classes. Returns (model, processor, resolved_revision).
    Mirrors routing/pilot_runner.load_vlm (see the duplication note above).
    """
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig

    spec = MODELS[model_key]
    if spec.get("fp16_safe", True):
        dtype = torch.float16
    elif torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
    common = dict(revision=spec["revision"], trust_remote_code=True)

    proc_kwargs = {}
    if min_pixels is not None:
        proc_kwargs["min_pixels"] = min_pixels
    if max_pixels is not None:
        proc_kwargs["max_pixels"] = max_pixels
    processor = AutoProcessor.from_pretrained(spec["hf_id"], **proc_kwargs, **common)

    try:
        from transformers import AutoModelForImageTextToText as _Auto
    except ImportError:             # older transformers
        from transformers import AutoModelForVision2Seq as _Auto
    model = _Auto.from_pretrained(
        spec["hf_id"], quantization_config=bnb, device_map="auto",
        torch_dtype=dtype, **common,
    )
    model.eval()
    print(f"{model_key}: compute dtype {dtype}")

    revision = getattr(model.config, "_commit_hash", None) or spec["revision"]
    return model, processor, revision


def env_versions():
    """Exact installed versions, for the meta file."""
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for pkg in ("torch", "transformers", "bitsandbytes", "accelerate",
                "sentence-transformers", "datasets"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def check_gated_access(model_keys=None):
    """
    Verify HF access to every model BEFORE any GPU work, by downloading
    each config.json — the exact request the model load makes.
    (model_info() is NOT enough: a gated repo's metadata is public, so it
    "succeeds" with no token and the run still 401s hours later.)
    Raises RuntimeError with a fix-it message on the first failure.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    for mk in (model_keys or list(MODELS)):
        spec = MODELS[mk]
        try:
            hf_hub_download(spec["hf_id"], "config.json", revision=spec["revision"])
            print(f"✓ access to {spec['hf_id']} verified (config.json downloaded)")
        except GatedRepoError:
            raise RuntimeError(
                f"NO ACCESS to the gated model {spec['hf_id']}: log in with a "
                "READ token from the account that accepted its license "
                f"(https://huggingface.co/{spec['hf_id']}) and retry.")
        except RepositoryNotFoundError:
            raise RuntimeError(
                f"{spec['hf_id']} not found — with a gated repo this can also "
                "mean the token has no access; check token + license.")
