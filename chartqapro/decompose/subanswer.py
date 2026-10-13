# =============================================================
# Sub-answers (Algorithm 1) — 3-path self-consistency per sub-query
# =============================================================
# A small model answers ONE sub-query against the chart with
# N_SUB_PATHS sampled reasoning paths; the majority answer wins and the
# AGREEMENT RATIO (majority size / paths sampled) is the confidence that
# travels into the evidence block. Extraction, normalization and the
# vote all come from scoring.py — nothing here re-invents correctness.
# =============================================================

import sys
from pathlib import Path

try:
    from . import prompts
except ImportError:                 # flat / notebook use
    import prompts

try:
    from ..scoring import aggregate_answers, extract_answer, normalize_answer
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scoring import aggregate_answers, extract_answer, normalize_answer

N_SUB_PATHS     = 3     # sampled paths per sub-query (spec: 3)
SUB_TEMPERATURE = 0.8   # same sampling temperature as the RoSE paths


def answer_subquery(sub_query: str, call_fn,
                    n_paths: int = N_SUB_PATHS,
                    temperature: float = SUB_TEMPERATURE) -> dict:
    """
    Answer one sub-query with self-consistency.

    call_fn(prompt, temperature) -> raw generation string. The caller owns
    the model, the chart image and the token budget, so this stays
    GPU-free and directly testable.

    Returns:
      answer      str   — the majority answer as the model wrote it
      answer_norm str   — its canonical form (scoring.normalize_answer)
      confidence  float — agreement ratio: majority size / n_paths
      raw_outputs list  — the n_paths generations, for the logs
    """
    prompt = prompts.sub_answer_prompt(sub_query)
    raws = [str(call_fn(prompt, temperature) or "") for _ in range(n_paths)]
    extracted = [extract_answer(r, "factoid") for r in raws]
    normalized = [normalize_answer(e, "factoid") for e in extracted]

    # Plain majority (no clustering, no weights): the confidence must be a
    # readable agreement ratio, not a weighted share.
    agg = aggregate_answers(normalized, use_clustering=False, weights=None)
    winner = agg["winner"]
    n_agree = sum(1 for a in normalized if a and a == winner)
    members = agg["members"]
    display = extracted[members[0]] if members else (extracted[0] if extracted else "")

    return {
        "answer":      display,
        "answer_norm": winner,
        "confidence":  round(n_agree / max(n_paths, 1), 4),
        "raw_outputs": raws,
    }
