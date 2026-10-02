# =============================================================
# The kNN router
# =============================================================
# For a new question:
#
#   1. Embed it (same all-mpnet-base-v2 as RoSE retrieval).
#   2. Find its K nearest pilot questions by cosine similarity.
#   3. For each model, score = similarity-weighted mean of that model's
#      hits over those neighbours ("how often did this model get
#      questions LIKE this one right?").
#   4. Walk ESCALATION_ORDER (cheapest first) and pick the first model
#      whose score ≥ TAU. Nothing clears the bar → the last entry (7B).
#
# The router is deterministic, CPU-only, and needs nothing beyond numpy
# once the pool is built — embedding is injected or lazily loaded, so
# tests and offline analysis never download a model.
# =============================================================

import json
from pathlib import Path

import numpy as np

try:
    from .config import ESCALATION_ORDER, K_NEIGHBOURS, TAU
except ImportError:                 # flat / notebook use
    from config import ESCALATION_ORDER, K_NEIGHBOURS, TAU


class Router:
    def __init__(self, pool: dict, k: int = K_NEIGHBOURS, tau: float = TAU,
                 order=None, embed_fn=None):
        """
        pool     : dict from routing.pool (build_pool/load_pool)
        k        : neighbours consulted per question
        tau      : min weighted hit rate to trust a model
        order    : escalation order; defaults to config.ESCALATION_ORDER
                   restricted to models the pool actually knows about
        embed_fn : callable(list[str]) -> array-like, for route_question()
        """
        self.entries = pool["entries"]
        if not self.entries:
            raise ValueError("routing pool is empty")
        self.k = max(1, min(k, len(self.entries)))
        self.tau = tau
        self.order = [m for m in (order or ESCALATION_ORDER)
                      if m in self.entries[0]["model_hits"]]
        if not self.order:
            raise ValueError("no model in ESCALATION_ORDER appears in the pool")
        self._embed_fn = embed_fn

        self._ids = [e["id"] for e in self.entries]
        emb = np.asarray([e["embedding"] for e in self.entries], dtype=np.float32)
        self._emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        self._hits = {
            m: np.asarray([e["model_hits"][m] for e in self.entries], dtype=np.float32)
            for m in self.entries[0]["model_hits"]
        }

    # ── core ─────────────────────────────────────────────────

    def route(self, embedding, exclude_id: str = None) -> dict:
        """
        Route one question given its embedding.

        exclude_id drops that pilot question from its own neighbourhood —
        this is what makes leave-one-out honest.

        Returns {model, scores, neighbours: [{id, sim}], k, tau}.
        """
        q = np.asarray(embedding, dtype=np.float32).reshape(-1)
        q = q / (np.linalg.norm(q) + 1e-12)
        sims = self._emb @ q

        mask = np.ones(len(sims), dtype=bool)
        if exclude_id is not None and exclude_id in self._ids:
            mask[self._ids.index(exclude_id)] = False

        cand = np.flatnonzero(mask)
        k = min(self.k, len(cand))
        top = cand[np.argsort(sims[cand])[-k:][::-1]]

        # Negative cosines would flip a hit into an anti-vote; clip to 0
        # so a dissimilar neighbour merely counts for nothing.
        w = np.clip(sims[top], 0.0, None)
        wsum = float(w.sum())

        scores = {}
        for m, hits in self._hits.items():
            scores[m] = float((w * hits[top]).sum() / wsum) if wsum > 0 else 0.0

        chosen = self.order[-1]
        for m in self.order[:-1]:
            if scores[m] >= self.tau:
                chosen = m
                break

        return {
            "model":      chosen,
            "scores":     {m: round(s, 4) for m, s in scores.items()},
            "neighbours": [{"id": self._ids[i], "sim": round(float(sims[i]), 4)}
                           for i in top],
            "k":   k,
            "tau": self.tau,
        }

    def route_question(self, question: str, exclude_id: str = None) -> dict:
        """Embed a raw question, then route it."""
        if self._embed_fn is None:
            try:
                from .pool import _default_embed_fn
            except ImportError:
                from pool import _default_embed_fn
            self._embed_fn = _default_embed_fn()
        [emb] = self._embed_fn([question])
        return self.route(emb, exclude_id=exclude_id)

    def route_many(self, items, exclude_self: bool = False) -> dict:
        """
        items: [{id, embedding}] → {id: decision}. With exclude_self=True
        each item is routed with itself removed from the pool (LOO).
        """
        return {
            str(it["id"]): self.route(
                it["embedding"],
                exclude_id=str(it["id"]) if exclude_self else None)
            for it in items
        }


# ─────────────────────────────────────────────────────────────
# Leave-one-out over the pilot
# ─────────────────────────────────────────────────────────────

def leave_one_out(pool: dict, k: int = K_NEIGHBOURS, tau: float = TAU,
                  order=None) -> list:
    """
    Route every pilot question with itself held out, and read the routed
    verdict straight off the pool's recorded hits — an unbiased estimate
    of routed accuracy that costs zero GPU seconds.

    Returns one row per pilot question:
      {id, model, hit, scores, neighbours}
    """
    router = Router(pool, k=k, tau=tau, order=order)
    rows = []
    for e in pool["entries"]:
        d = router.route(e["embedding"], exclude_id=e["id"])
        rows.append({
            "id":         e["id"],
            "model":      d["model"],
            "hit":        e["model_hits"][d["model"]],
            "scores":     d["scores"],
            "neighbours": [n["id"] for n in d["neighbours"]],
        })
    return rows


# ─────────────────────────────────────────────────────────────
# Routing a full question file (Stage 2, GPU-free step)
# ─────────────────────────────────────────────────────────────

def route_file(pool_path, questions_path, out_path,
               k: int = K_NEIGHBOURS, tau: float = TAU,
               skip_ids=None) -> dict:
    """
    Route every question in a dataset JSON (the chartqapro_factoid.json a
    sweep notebook builds) and write routing_decisions.json:
        {id: {model, scores, neighbours}}
    skip_ids (e.g. the pilot ids) are left out — their answers already exist.
    """
    try:
        from .pool import load_pool, _default_embed_fn
    except ImportError:
        from pool import load_pool, _default_embed_fn

    pool = load_pool(pool_path)
    questions = json.loads(Path(questions_path).read_text())
    skip = {str(i) for i in (skip_ids or [])}
    todo = [q for q in questions if str(q.get("id")) not in skip]

    embed = _default_embed_fn()
    embeddings = embed([q.get("question", "") for q in todo])

    router = Router(pool, k=k, tau=tau)
    decisions = {
        str(q["id"]): router.route(emb)
        for q, emb in zip(todo, embeddings)
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(decisions, indent=2))

    share = {}
    for d in decisions.values():
        share[d["model"]] = share.get(d["model"], 0) + 1
    print(f"✓ routed {len(decisions)} questions → {out_path}")
    for m, n in sorted(share.items()):
        print(f"  {m:<16} {n:4d}  ({n / max(len(decisions), 1) * 100:.1f}%)")
    return decisions
