# =============================================================
# Model registry + routing hyperparameters
# =============================================================
# One place for everything the router and the pilot runner need to
# know about the three VLMs. `revision` pins the exact HuggingFace
# commit a run used; None means "whatever main resolves to", and the
# pilot runner records the RESOLVED sha into its meta output so a
# finished run is always reproducible even if the pin was left open.
# =============================================================

# Keys are the short names used everywhere: result filenames, the
# routing pool, routing_decisions.json, report tables.
MODELS = {
    "gemma3-4b": {
        "hf_id":    "google/gemma-3-4b-it",
        "revision": None,          # TODO: pin to the sha your pilot run used
        "params_b": 4.3,           # nominal, for the mean-params-per-query metric
        "label":    "Gemma 3 4B-it",
    },
    "qwen3.5-4b": {
        "hf_id":    "Qwen/Qwen3.5-4B",     # unified VL model — no separate -VL variant
        "revision": None,          # TODO: pin to the sha your pilot run used
        "params_b": 4.0,
        "label":    "Qwen3.5-4B (VL)",
    },
    "qwen2.5-vl-7b": {
        "hf_id":    "Qwen/Qwen2.5-VL-7B-Instruct",
        "revision": None,          # TODO: pin to the sha of the final-week 7B run
        "params_b": 8.3,           # 7B LLM + ~0.7B vision tower
        "label":    "Qwen2.5-VL-7B-Instruct",
    },
}

# Cheapest-first. The router walks this list and picks the FIRST model
# whose neighbourhood hit-rate clears TAU; the last entry is the
# unconditional fallback, so a question no small model looks safe on
# always escalates to 7B.
ESCALATION_ORDER = ["gemma3-4b", "qwen3.5-4b", "qwen2.5-vl-7b"]

# ── routing hyperparameters ──────────────────────────────────
K_NEIGHBOURS = 5     # pilot neighbours consulted per question
TAU          = 0.6   # min similarity-weighted hit rate to trust a small model

# Same embedder RoSE itself retrieves demonstrations with — one more
# reason the routing pool and the experience pool stay conceptually
# interchangeable.
EMBED_MODEL = "all-mpnet-base-v2"
