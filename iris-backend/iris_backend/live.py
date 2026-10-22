"""Live runs: Algorithm 2 with user-selected models, streamed stage by stage.

Design decisions (approved in the design review):
- The chart description is ALWAYS generated (once per chart, cached), even in
  fast mode where the two-stage reasoning pass is off — the accessible viewer
  needs the text alternative. When two-stage IS on, rose.get_chart_description
  is wrapped to return the cached text so it is not generated twice.
- Per-path streaming taps rose.call_vlm at runtime for the duration of the
  RoSE stage (the same pattern decomp_runner.inject_block uses on the prompt
  builders); rose_chartqapro.py's source is not touched.
- Fast mode mirrors the fast notebooks at runtime: M_PATHS=3 and
  two_stage_reasoning off; spec-exact restores the loaded configuration.
- The experience pool rebuilds from IRIS_POOL_PATH (a results JSON, exactly
  as decomp_runner resumes from a checkpoint) and stays read-only during
  runs unless IRIS_POOL_STREAM=true.
"""

import contextlib
import json
import time
from pathlib import Path

from .config import Settings


class LiveEngine:
    def __init__(self, settings: Settings, by_id: "dict | None" = None):
        self.settings = settings
        self.by_id = by_id if by_id is not None else {}
        self.loaded = {}
        self.revisions = {}
        self.rose = None
        self.embedder = None
        self.embed_fn = None
        self.pool = None
        self.description_cache = {}

    # ── startup ──────────────────────────────────────────────

    def load(self):
        import rose_chartqapro as rose
        from decompose import decomp_runner
        from decompose.models import check_gated_access
        from sentence_transformers import SentenceTransformer

        self.rose = rose
        check_gated_access()
        self.loaded, self.revisions = decomp_runner.load_all_models("nf4")
        self.embedder = SentenceTransformer(rose.EMBED_MODEL)
        self.embed_fn = lambda texts: self.embedder.encode(
            list(texts), convert_to_numpy=True)
        self.pool = rose.ExperiencePool(self.embedder)
        self._seed_pool()

    def _seed_pool(self):
        path = self.settings.pool_path
        if not path or not Path(path).exists():
            return
        rows = json.loads(Path(path).read_text())
        for r in rows:
            if r.get("method") == "error":
                continue
            self.pool.add(
                question=r.get("question", ""),
                rationale=r.get("best_rationale") or r.get("prediction", ""),
                answer=r.get("prediction", ""),
                qtype=r.get("question_type", "factoid"),
                uncertainty=r.get("uncertainty", 0.5),
                complexity=r.get("complexity", 1.0),
                clip_emb=None,
            )
        print(f"experience pool seeded: {self.pool.size()} entries from {path}")

    # ── runtime wraps (no chartqapro source changes) ─────────

    @contextlib.contextmanager
    def _fast_overrides(self, fast: bool):
        """Mirror the fast notebooks: 3 paths, no two-stage pass."""
        rose = self.rose
        saved_m, saved_ext = rose.M_PATHS, dict(rose.EXTENSIONS)
        if fast:
            rose.M_PATHS = 3
            rose.EXTENSIONS["two_stage_reasoning"] = False
        try:
            yield
        finally:
            rose.M_PATHS = saved_m
            rose.EXTENSIONS.clear()
            rose.EXTENSIONS.update(saved_ext)

    @contextlib.contextmanager
    def _tap_paths(self, emit, expected: int):
        """Emit a stage.path event for every generation RoSE samples."""
        import scoring
        rose = self.rose
        orig = rose.call_vlm
        state = {"i": 0}

        def tapped(model, processor, image, prompt, temperature=None,
                   max_new_tokens=None):
            kwargs = {}
            if temperature is not None:
                kwargs["temperature"] = temperature
            if max_new_tokens is not None:
                kwargs["max_new_tokens"] = max_new_tokens
            out = orig(model, processor, image, prompt, **kwargs)
            i = state["i"]
            state["i"] += 1
            extracted = scoring.extract_answer(out)
            emit("stage.path", {
                "index": i, "total": expected, "raw": out,
                "extracted": extracted,
                "announce": f"Reasoning path {i + 1} of {expected}: "
                            f"{extracted or 'no final answer'}",
            })
            return out

        rose.call_vlm = tapped
        try:
            yield
        finally:
            rose.call_vlm = orig

    @contextlib.contextmanager
    def _serve_cached_description(self, description: str):
        """When two-stage is on, hand RoSE the already-generated description."""
        rose = self.rose
        if not rose.ext("two_stage_reasoning"):
            yield
            return
        orig = rose.get_chart_description
        rose.get_chart_description = lambda *a, **k: description
        try:
            yield
        finally:
            rose.get_chart_description = orig

    # ── pieces ───────────────────────────────────────────────

    def _resolve_image_path(self, request: dict) -> Path:
        if request.get("question_id"):
            rec = self.by_id.get(request["question_id"])
            if rec is None:
                raise ValueError(f"no question {request['question_id']!r}")
            if not self.settings.image_dir:
                raise ValueError("IRIS_IMAGE_DIR is not configured")
            return Path(self.settings.image_dir) / rec["image_name"]
        path = Path(self.settings.upload_dir) / f"{request['image_id']}.png"
        if not path.exists():
            raise ValueError(f"no uploaded image {request['image_id']!r}")
        return path

    def _build_sample(self, request: dict) -> "tuple[dict, Path, bool]":
        image_path = self._resolve_image_path(request)
        if request.get("question_id"):
            rec = self.by_id[request["question_id"]]
            sample = {"id": rec["id"], "question": rec["question"],
                      "question_type": rec["question_type"],
                      "choices": rec.get("choices"),
                      "answer": rec["ground_truth"],
                      "image": str(image_path)}
            return sample, image_path, True
        sample = {"id": f"adhoc-{int(time.time())}",
                  "question": str(request.get("question", "")).strip(),
                  "question_type": "factoid", "choices": None,
                  "answer": "", "image": str(image_path)}
        return sample, image_path, False

    def _describe(self, image_path: Path, image, final_model: str) -> str:
        """The chart description — always generated, cached per chart."""
        key = str(image_path)
        if key not in self.description_cache:
            rose = self.rose
            model, processor = self.loaded[final_model]
            self.description_cache[key] = rose.call_vlm(
                model, processor, image, rose._DESCRIPTION_PROMPT,
                temperature=0.0, max_new_tokens=rose.DESC_MAX_TOKENS)
        return self.description_cache[key]

    def _truncate_pool(self, n: int):
        """Undo pool growth after a run (read-only demo pool)."""
        import numpy as np
        self.pool.experiences = self.pool.experiences[:n]
        self.pool._embeddings = (
            np.stack([e["embedding"] for e in self.pool.experiences])
            if self.pool.experiences else None)

    # ── the run ──────────────────────────────────────────────

    def execute(self, run_id: str, request: dict, events, loop, info: dict):
        """Runs in a worker thread; publishes events through the app loop."""
        import scoring
        from decompose import decomp_runner, prompts, vote
        from decompose.decomposer import decompose

        def emit(name, data):
            events.publish_threadsafe(loop, name, data)

        rose = self.rose
        planners = list(request.get("planners") or [])
        final_model = request.get("final_model") or "qwen2.5-vl-7b"
        fast = bool(request.get("fast_mode", True))
        for key in [*planners, final_model]:
            if key not in self.loaded:
                raise ValueError(f"model {key!r} is not loaded")

        sample, image_path, has_gold = self._build_sample(request)
        question = sample["question"]
        emit("started", {
            "planners": planners, "final_model": final_model,
            "fast_mode": fast, "question": question,
            "announce": f"Run started with {len(planners)} planner"
                        f"{'s' if len(planners) != 1 else ''}.",
        })

        image = rose.load_image(sample["image"])
        stage_lat = {}
        t_run = time.time()

        with self._fast_overrides(fast):
            # 1 — planners propose sub-queries
            proposals, fallbacks = [], {}
            names = {"gemma3-4b": "Gemma", "qwen3.5-4b": "Qwen3.5",
                     "qwen2.5-vl-7b": "Qwen2.5-VL"}
            for mk in planners:
                t0 = time.time()
                sq, fell_back, raw = decompose(
                    question,
                    lambda p: decomp_runner._call(
                        self.loaded, mk, image, p, 0.0,
                        max_new_tokens=decomp_runner.DECOMP_MAX_TOKENS))
                lat = round(time.time() - t0, 2)
                stage_lat[f"decompose_{mk}"] = lat
                proposals.append((mk, sq))
                fallbacks[mk] = fell_back
                emit("stage.decompose", {
                    "model": mk, "raw": raw, "sub_queries": sq,
                    "fallback": fell_back, "latency_s": lat,
                    "announce": (f"{names.get(mk, mk)}'s proposal could not be "
                                 "parsed and falls back to the question."
                                 if fell_back else
                                 f"{names.get(mk, mk)} proposed two sub-queries."),
                })
                if info.get("cancelled"):
                    raise RuntimeError("cancelled")

            # 2 — vote
            t0 = time.time()
            choice = vote.choose_decomposition(
                proposals, question, self.embed_fn, fallbacks=fallbacks)
            lat = round(time.time() - t0, 2)
            stage_lat["choose"] = lat
            rule = choice["vote_rule"]
            if rule == "merge":
                vote_announce = (
                    "Vote merged the proposals (similarities "
                    + " and ".join(f"{s:.2f}" for s in choice["pair_sims"]) + ").")
            elif rule.startswith("pick:"):
                vote_announce = (f"Vote chose {names.get(choice['picked_model'], choice['picked_model'])}'s "
                                 "plan — it sits closer to the question.")
            elif rule.startswith("single:"):
                vote_announce = (f"Only {names.get(choice['picked_model'], choice['picked_model'])}'s "
                                 "proposal was usable; using it.")
            else:
                vote_announce = ("Both proposals fell back; continuing with "
                                 "the original question.")
            emit("stage.vote", {
                **{k: choice[k] for k in ("rule", "picked_model", "matching",
                                          "pair_sims", "mean_sim_to_question",
                                          "threshold")},
                "vote_rule": rule, "chosen": choice["sub_queries"],
                "latency_s": lat, "announce": vote_announce,
            })

            # 3 — chart description (always, for the accessible viewer)
            t0 = time.time()
            description = self._describe(image_path, image, final_model)
            stage_lat["describe"] = round(time.time() - t0, 2)
            emit("stage.describe", {
                "description": description,
                "latency_s": stage_lat["describe"],
                "announce": "IRIS described the chart.",
            })
            if info.get("cancelled"):
                raise RuntimeError("cancelled")

            # 4 — RoSE on the final model, paths streamed
            block = ("" if rule == "fallback"
                     else prompts.subquery_steps_block(choice["sub_queries"]))
            pool_before = self.pool.size()
            t0 = time.time()
            with self._serve_cached_description(description), \
                    self._tap_paths(emit, expected=rose.M_PATHS):
                r = decomp_runner._rose_with_block(
                    sample, self.loaded, self.pool, block, None, None,
                    final_model)
            stage_lat["rose_final"] = round(time.time() - t0, 2)
            if not self.settings.pool_stream:
                self._truncate_pool(pool_before)

        # 5 — final
        is_correct = None
        if has_gold:
            is_correct = scoring.is_correct(
                r.get("prediction", ""), sample["answer"],
                sample["question_type"])
        from decompose.models import MODELS
        params_b = round(sum(
            MODELS.get(k, {}).get("params_b", 0.0)
            for k in [*planners, final_model]), 2)
        answer = r.get("prediction", "")
        announce = f"Final answer: {answer}."
        if is_correct is not None:
            announce += (" It matches the expected answer."
                         if is_correct else
                         f" The expected answer was {sample['answer']}.")
        emit("stage.final", {
            "answer": answer,
            "prediction_norm": r.get("prediction_norm", ""),
            "ground_truth": sample["answer"] if has_gold else None,
            "is_correct": is_correct,
            "agreement": r.get("agreement"),
            "uncertainty": r.get("uncertainty"),
            "method": r.get("method", ""),
            "vote_rule": rule,
            "sub_queries": choice["sub_queries"],
            "latency_s": round(time.time() - t_run, 2),
            "stage_latency_s": stage_lat,
            "params_b_used": params_b,
            "pool_size": self.pool.size(),
            "announce": announce,
        })
