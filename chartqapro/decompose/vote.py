# =============================================================
# Voting — evidence block (Alg 1) and decomposition vote (Alg 2)
# =============================================================
# CPU-only. The Algorithm 2 vote embeds sub-queries (the runner passes
# the same all-mpnet-base-v2 encoder RoSE retrieves demonstrations with)
# and either MERGES two agreeing proposals or PICKS the proposal whose
# sub-queries sit closer to the original question. The rule that fired is
# always returned, so every run logs its own vote behaviour.
# =============================================================

import numpy as np

try:
    from . import prompts
except ImportError:                 # flat / notebook use
    import prompts

# Two proposals "agree" when, after best matching, BOTH matched pairs
# have cosine similarity at or above this.
AGREE_THRESHOLD = 0.8


# ── Algorithm 1: evidence block ──────────────────────────────

def answer_evidence(items: list) -> str:
    """
    Format the sub-answers as the evidence block prepended to the RoSE
    reasoning prompt. `items` is a list of dicts with keys
    sub_query / answer / model / confidence, in sub-query order.
    """
    lines = [prompts.EVIDENCE_HEADER]
    for i, it in enumerate(items, 1):
        lines.append(f"[Evidence {i}] sub-query: {it['sub_query']}")
        lines.append(
            f"  sub-answer: {it['answer']}  "
            f"(model: {it['model']}, confidence: {float(it['confidence']):.2f})"
        )
    return "\n".join(lines)


# ── Algorithm 2: vote on the decomposition ───────────────────

def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    return v / (np.linalg.norm(v) + 1e-12)


def choose_decomposition(proposals: list, question: str, embed_fn,
                         threshold: float = AGREE_THRESHOLD,
                         fallbacks: "dict | None" = None) -> dict:
    """
    Pick one decomposition out of (up to) two proposals.

    proposals : [(model_key, [sq1, sq2]), (model_key, [sq1, sq2])]
    embed_fn  : list[str] -> array-like of embeddings (all-mpnet-base-v2
                in production; anything with the same shape in tests)
    fallbacks : {model_key: bool} — True when that proposal is the parser's
                [question, question] fallback. A fallback proposal is
                EXCLUDED from the vote: its two sub-queries are the original
                question verbatim, so its pairwise cosine to itself is 1.0
                and its mean similarity to the question is 1.0 — left in,
                it would always win both rules. None keeps the historical
                behaviour (no exclusion).

    Rule, in order:
      fallback — every proposal fell back: return the original question
              (vote_rule "fallback"); the pipeline behaves like plain RoSE.
      single — exactly one usable proposal (the other fell back, or only
              one planner was asked): take it, vote_rule "single:<model>".
      merge — the proposals agree: after best matching of the 2x2 pairs
              (straight a1–b1/a2–b2 vs crossed a1–b2/a2–b1, whichever has
              the higher mean cosine), BOTH matched pairs clear
              `threshold`. Each matched pair keeps the sub-query closer
              to the original question, in proposal-A order.
      pick  — otherwise take the whole proposal whose sub-queries have
              the higher MEAN cosine similarity to the original question.

    Returns a dict with sub_queries, rule ("fallback"/"single"/"merge"/
    "pick"), vote_rule (the loggable label, e.g. "merge" or
    "pick:gemma3-4b"), picked_model, matching, pair_sims and
    mean_sim_to_question.
    """
    fallbacks = fallbacks or {}
    usable = [(k, sq) for k, sq in proposals if not fallbacks.get(k)]

    if not usable:
        return {
            "sub_queries": [question],
            "rule": "fallback", "vote_rule": "fallback",
            "picked_model": None, "matching": None, "pair_sims": [],
            "mean_sim_to_question": {}, "threshold": threshold,
        }
    if len(usable) == 1:
        key, prop = usable[0]
        return {
            "sub_queries": list(prop),
            "rule": "single", "vote_rule": f"single:{key}",
            "picked_model": key, "matching": None, "pair_sims": [],
            "mean_sim_to_question": {}, "threshold": threshold,
        }

    (key_a, prop_a), (key_b, prop_b) = usable
    embs = np.asarray(embed_fn([question] + list(prop_a) + list(prop_b)))
    q, a0, a1, b0, b1 = (_unit(e) for e in embs)

    def sim(x, y):
        return float(np.dot(x, y))

    # Best matching between the two 2-element proposals.
    straight = [sim(a0, b0), sim(a1, b1)]        # a1–b1, a2–b2
    crossed  = [sim(a0, b1), sim(a1, b0)]        # a1–b2, a2–b1
    if sum(straight) >= sum(crossed):
        matching, pair_sims = "straight", straight
        pairs = [((prop_a[0], a0), (prop_b[0], b0)),
                 ((prop_a[1], a1), (prop_b[1], b1))]
    else:
        matching, pair_sims = "crossed", crossed
        pairs = [((prop_a[0], a0), (prop_b[0], b1)),
                 ((prop_a[1], a1), (prop_b[1], b0))]

    mean_to_q = {
        key_a: (sim(a0, q) + sim(a1, q)) / 2.0,
        key_b: (sim(b0, q) + sim(b1, q)) / 2.0,
    }

    if min(pair_sims) >= threshold:
        # Agreement: merge, keeping per pair the sub-query closer to the
        # original question (ties go to proposal A, arbitrary but stable).
        sub_queries = [
            text_a if sim(emb_a, q) >= sim(emb_b, q) else text_b
            for (text_a, emb_a), (text_b, emb_b) in pairs
        ]
        rule, picked = "merge", None
    else:
        picked = key_a if mean_to_q[key_a] >= mean_to_q[key_b] else key_b
        sub_queries = list(prop_a if picked == key_a else prop_b)
        rule = "pick"

    return {
        "sub_queries":          sub_queries,
        "rule":                 rule,
        "vote_rule":            rule if rule == "merge" else f"pick:{picked}",
        "picked_model":         picked,
        "matching":             matching,
        "pair_sims":            [round(s, 4) for s in pair_sims],
        "mean_sim_to_question": {k: round(v, 4) for k, v in mean_to_q.items()},
        "threshold":            threshold,
    }
