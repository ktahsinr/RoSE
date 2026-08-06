# =============================================================
# RoSE on ChartQAPro — Paper-Faithful Implementation
# Paper: "Making LLMs Better Reasoners with Orchestrated
#         Streaming Experiences" (EMNLP 2024)
# Target: ChartQAPro Factoid + MCQ questions only
# Model:  Qwen2.5-VL-7B-Instruct, 4-bit (fits a free T4 GPU)
# =============================================================
#
# EXACT PAPER MAPPINGS:
#   m = 20 reasoning paths per question (reduced to 3 for free GPU)
#   λ = 1.2 × min_uncertainty per bucket (Eq. 6-7)
#   k = number of demonstrations = number of buckets (default 3)
#   Uncertainty = Shannon entropy (Eq. 1-3)
#   Complexity  = avg CountSteps of majority-answer paths (Eq. 4)
#   Stored path = argmax CountSteps (Eq. 5)
#   Selection   = argmax complexity per bucket (Eq. 8)
#   Inference   = LLM(q1,r1,a1,...,qk,rk,ak,qt) (Eq. 9-10)
#
# Answer extraction, normalization, and scoring live in scoring.py
# (pure Python, no GPU deps) so results can be re-scored offline.
# =============================================================

import json, os, math, re, time
from pathlib import Path
from collections import Counter, defaultdict

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

try:                                # package-relative (preferred)
    from .scoring import (
        extract_answer, normalize_answer, is_correct, is_unanswerable,
        resolve_choice, aggregate_answers, has_final_answer,
        NUMERIC_TOLERANCE,
    )
except ImportError:                 # flat / notebook use
    from scoring import (
        extract_answer, normalize_answer, is_correct, is_unanswerable,
        resolve_choice, aggregate_answers, has_final_answer,
        NUMERIC_TOLERANCE,
    )

# Qwen2.5-VL has its own model class in recent transformers; the older
# Qwen2VL class silently mis-handles some 2.5 checkpoints.
try:
    from transformers import Qwen2_5_VLForConditionalGeneration as _VLModel
except ImportError:
    from transformers import Qwen2VLForConditionalGeneration as _VLModel


# ─────────────────────────────────────────────────────────────
# CONFIG  — adjust only these values if needed
# ─────────────────────────────────────────────────────────────
MODEL_NAME       = "Qwen/Qwen2.5-VL-7B-Instruct"   # 7B, NOT 3B — see README
EMBED_MODEL      = "all-mpnet-base-v2"   # the paper's embedder (was MiniLM-L6)

DATASET_PATH     = "ChartQAPro/data/test.json"
IMAGES_DIR       = "ChartQAPro/data/images"     # folder containing chart PNGs
RESULTS_DIR      = "results"
CHECKPOINT_FILE  = "results/checkpoint.json"
FINAL_FILE       = "results/rose_factoid_mcq_results.json"

# Paper hyperparameters
M_PATHS          = 3     # paper uses 20; reduce to 3 for free T4 GPU
K_DEMONSTRATIONS = 3     # number of few-shot examples (= number of buckets)
LAMBDA           = 1.2   # uncertainty threshold multiplier (paper §3.2)
TEMPERATURE      = 0.7   # paper uses 1.0; 0.7 works well for smaller models
MAX_NEW_TOKENS   = 384
CHECKPOINT_EVERY = 25    # save progress every N samples

# Vision resolution. Charts are resolution-bound: too few pixels and the
# model cannot read axis labels at all. 28*28 is Qwen's patch area.
MIN_PIXELS       = 256 * 28 * 28
MAX_PIXELS       = 1280 * 28 * 28
MAX_IMAGE_SIDE   = 1600  # resize guard — keeps huge charts from OOMing the T4

# Offer "Unanswerable" as an allowed answer (a large slice of the gold set)
OFFER_UNANSWERABLE = True


# ─────────────────────────────────────────────────────────────
# EXTENSIONS — changes that are NOT in the paper
# ─────────────────────────────────────────────────────────────
# Everything below deviates from RoSE as published. Each is separately
# switchable so you can report a faithful baseline AND the improved
# system, with an ablation between them.
#
# Set PAPER_FAITHFUL = True to disable all of them at once and run RoSE
# exactly as specified (with the implementation bugs fixed — those are
# corrections, not extensions).
#
# From a Colab cell:
#     rose_chartqapro.PAPER_FAITHFUL = True                     # faithful run
#     rose_chartqapro.EXTENSIONS["greedy_first_path"] = False    # one at a time
#
PAPER_FAITHFUL = False

EXTENSIONS = {
    # Rotate MCQ option order across the m paths we already pay for, so
    # option-position bias cancels in the vote. Not in Eq. 9.
    "mcq_permute_options": True,

    # Group numeric answers within tolerance before voting, and take the
    # median of the largest group. Changes Eq. 1-3.
    "numeric_vote_clustering": True,

    # Retrieve demonstrations only from the same question type, so a
    # factoid question is not shown MCQ examples. Changes Algorithm 1.
    "type_aware_retrieval": True,

    # Exclude paths that never emitted a 'Final Answer:' line from the
    # vote (they are being read by a noisy last-line fallback).
    "drop_malformed_paths": True,

    # Decode path 0 greedily (T=0) instead of sampling. The paper samples
    # every path at T=1.0.
    "greedy_first_path": True,

    # Add an axis-and-units-first reading scaffold to the prompt.
    "chart_reading_scaffold": True,
}


def ext(name: str) -> bool:
    """True if extension `name` is active."""
    if PAPER_FAITHFUL:
        return False
    return bool(EXTENSIONS.get(name, False))


def active_extensions() -> dict:
    """Provenance — written into results/meta.json so a run is self-describing."""
    return {k: ext(k) for k in EXTENSIONS}

# Filter: only run on these question types
TARGET_TYPES     = {"factoid", "mcq"}   # lowercase


# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD MODELS
# ─────────────────────────────────────────────────────────────

def load_models():
    print(f"\n[1/3] Loading {MODEL_NAME} (4-bit quantized for T4 GPU)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    model = _VLModel.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    mem = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"   ✓ VLM loaded  ({mem:.1f}/{total:.1f} GB used)")

    print("[2/3] Loading sentence embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL)
    print("   ✓ Embedder loaded")
    return model, processor, embedder


# ─────────────────────────────────────────────────────────────
# STEP 2 — IMAGE LOADER
# ─────────────────────────────────────────────────────────────

def load_image(image_field) -> Image.Image:
    """
    Find and open the chart image.
    ChartQAPro stores images under data/images/<source>/<filename>.

    Always converts to RGB (some charts are palette/RGBA PNGs, which the
    processor mishandles) and downscales oversized charts so a single
    high-resolution image cannot exhaust T4 memory.
    """
    if isinstance(image_field, Image.Image):
        img = image_field
    else:
        candidates = [
            Path(str(image_field)),
            Path(IMAGES_DIR) / str(image_field),
            Path(IMAGES_DIR) / Path(str(image_field)).name,
            Path("ChartQAPro") / str(image_field),
        ]
        img = None
        for p in candidates:
            if p.exists():
                img = Image.open(p)
                break
        if img is None:
            raise FileNotFoundError(
                f"Image not found. Tried: {[str(c) for c in candidates]}"
            )

    img = img.convert("RGB")

    # Resize guard — preserve aspect ratio, only ever shrink.
    longest = max(img.size)
    if longest > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / longest
        new_size = (max(1, int(img.width * scale)),
                    max(1, int(img.height * scale)))
        img = img.resize(new_size, Image.LANCZOS)

    return img


# ─────────────────────────────────────────────────────────────
# STEP 3 — VLM CALL  (single forward pass)
# ─────────────────────────────────────────────────────────────

def call_vlm(model, processor, image: Image.Image,
             prompt: str, temperature: float = TEMPERATURE) -> str:
    """One forward pass through Qwen2.5-VL. Returns raw output string."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text":  prompt},
        ],
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=[image], return_tensors="pt"
    ).to(model.device)

    gen_kwargs = dict(max_new_tokens=MAX_NEW_TOKENS)
    if temperature and temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
    else:
        gen_kwargs.update(do_sample=False)

    with torch.no_grad():
        out_ids = model.generate(**inputs, **gen_kwargs)

    new_tokens = out_ids[0][inputs["input_ids"].shape[1]:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────────────────────
# STEP 4 — PROMPTS  (zero-shot CoT and few-shot CoT)
# ─────────────────────────────────────────────────────────────

# Strict output contract. This does double duty: it makes extraction
# reliable AND stops the model emitting '[85]' or '9%' where the gold
# answer is a bare value.
_FACTOID_FORMAT = (
    "Answer with a single bare value.\n"
    "Rules for the final line:\n"
    "  - no brackets, quotes, commas or thousands separators\n"
    "  - no units, currency symbols or percent signs\n"
    "  - no sentences or explanation\n"
    "  - if the chart does not contain enough information to answer, "
    "write exactly: Unanswerable\n\n"
    "End your response with exactly this line:\n"
    "Final Answer: <value>"
)

# Unanswerable calibration. Offering the option is what fixed MCQ, but
# offering it without a bar invites over-punting: the model takes the
# escape hatch instead of reading a value off the chart. This raises the
# bar without removing the option.
# EXTENSION chart_reading_scaffold: the dominant factoid failure mode is
# misreading the axis or the units, not faulty arithmetic. Naming that step
# explicitly costs a handful of prompt tokens.
_SCAFFOLD = (
    "Before answering: read the axis labels and their units, identify which "
    "series or category the question is about, then read the value.\n\n"
)


def _scaffold() -> str:
    return _SCAFFOLD if ext("chart_reading_scaffold") else ""


_MCQ_FORMAT = (
    "Choose Unanswerable ONLY if the chart genuinely does not contain the "
    "information needed. If the chart does show the information — even if "
    "you have to read or estimate a value from an axis or a bar — choose "
    "the closest option instead.\n\n"
    "End your response with exactly this line:\n"
    "Final Answer: (X)\n"
    "where X is the letter of the correct option."
)


def build_options(choices: list, perm: list = None):
    """
    Render the option block, appending an Unanswerable option when
    configured, and optionally displaying them in a permuted order.

    Returns (rendered_text, options, perm) where `options` is the
    canonical (unpermuted) list and perm[displayed_index] = canonical_index.
    """
    opts = list(choices)
    if OFFER_UNANSWERABLE and not any(is_unanswerable(c) for c in opts):
        opts.append("Unanswerable")
    if perm is None:
        perm = list(range(len(opts)))
    rendered = "\n".join(
        f"({chr(65 + d)}) {opts[perm[d]]}" for d in range(len(perm))
    )
    return rendered, opts, perm


def option_permutations(n_options: int, m: int) -> list:
    """
    m option orderings, as rotations. perm[displayed] = canonical.

    This is the free MCQ accuracy lever. We already pay for m sampled
    paths per question; running them all against the SAME option order
    means their errors are correlated by option position, which is a
    well-known VLM bias. Rotating the order across the same m paths costs
    nothing extra and lets position bias cancel in the vote.

    The Unanswerable option is appended before permuting, so it moves
    around too — otherwise it would always sit last and carry its own
    position bias.

    Rotation 0 is the identity, so path 0 always sees the dataset's own
    ordering. Counter.most_common breaks ties by insertion order, which
    means a 3-way tie falls back to that unpermuted path.
    """
    if n_options <= 1:
        return [[0]] * m
    return [
        [(d + r) % n_options for d in range(n_options)]
        for r in (i % n_options for i in range(m))
    ]


def canonical_mcq_answer(span: str, opts: list, perm: list) -> str:
    """
    Map a model answer given in PERMUTED option space to a canonical
    option letter, so votes across different orderings are comparable.
    """
    if not opts:
        return normalize_answer(span, "mcq")

    displayed_idx = resolve_choice(span, [opts[perm[d]] for d in range(len(perm))])
    if displayed_idx is not None:
        return chr(65 + perm[displayed_idx]).lower()

    # Unresolvable — keep the normalized text so it still votes as itself.
    return normalize_answer(span, "mcq")


def zero_shot_prompt(question: str, qtype: str,
                     choices: list = None, perm: list = None) -> str:
    """
    Zero-Shot-CoT prompt — paper §3.3, used when the pool is empty.
    """
    header = ("Look at the chart carefully and answer the question.\n\n"
              + _scaffold())
    if qtype == "mcq" and choices:
        opts, _, _ = build_options(choices, perm)
        return (
            f"{header}Question: {question}\n\nOptions:\n{opts}\n\n"
            f"Let's think step by step.\n\n{_MCQ_FORMAT}"
        )
    return (
        f"{header}Question: {question}\n\n"
        f"Let's think step by step.\n\n{_FACTOID_FORMAT}"
    )


def few_shot_prompt(question: str, qtype: str,
                    demonstrations: list,
                    choices: list = None, perm: list = None) -> str:
    """
    Few-Shot-CoT prompt — paper Equation 9:
    LLM(q1, r1, a1, ..., qk, rk, ak, qt)

    Each demonstration is a triplet: (question, rationale, answer).
    The chart image is passed separately to the VLM.
    """
    header = (
        "You are an expert at reading and understanding charts. "
        "Here are some examples of chart questions with step-by-step reasoning:\n\n"
    ) + _scaffold()
    examples = ""
    for i, d in enumerate(demonstrations, 1):
        examples += (
            f"[Example {i}]\n"
            f"Q: {d['question']}\n"
            f"Reasoning: {d['rationale']}\n"
            f"A: {d['answer']}\n\n"
        )

    if qtype == "mcq" and choices:
        opts, _, _ = build_options(choices, perm)
        task = (
            "Now answer the question about the chart above:\n\n"
            f"Q: {question}\n\nOptions:\n{opts}\n\n"
            f"Let's think step by step.\n\n{_MCQ_FORMAT}"
        )
    else:
        task = (
            "Now answer the question about the chart above:\n\n"
            f"Q: {question}\n\n"
            f"Let's think step by step.\n\n{_FACTOID_FORMAT}"
        )

    return header + examples + task


# ─────────────────────────────────────────────────────────────
# STEP 6 — COUNT STEPS  (paper Equation 4–5)
# ─────────────────────────────────────────────────────────────

# The old indicator list contained bare "-", "+", "/" and "so", which
# matched almost every line and made complexity a proxy for line count —
# meaning Eq. 8 was really "pick the longest rationale". These are matched
# on word boundaries, and real arithmetic is counted separately.
_STEP_INDICATORS = [
    "step", "first", "second", "third", "next", "finally",
    "looking at", "from the chart", "the chart shows", "according to",
    "we can see", "because", "since", "therefore", "thus",
    "note that", "observing", "calculate", "difference", "compare",
]

_STEP_RE = re.compile(
    r"\b(?:%s)\b" % "|".join(re.escape(w) for w in _STEP_INDICATORS),
    re.IGNORECASE,
)
_ARITHMETIC_RE = re.compile(r"\d\s*[-+*/=×÷]\s*\d")


def count_steps(text: str) -> int:
    """
    Count reasoning steps in a rationale.
    Paper: 'we see a line as one reasoning step' (§3.1).
    A line counts when it contains a reasoning cue or actual arithmetic.
    """
    count = 0
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if _STEP_RE.search(line) or _ARITHMETIC_RE.search(line):
            count += 1
    return max(count, 1)  # at least 1 to avoid zero-complexity entries


# ─────────────────────────────────────────────────────────────
# STEP 7 — SELF-CONSISTENCY + UNCERTAINTY (paper Eq. 1–3)
# ─────────────────────────────────────────────────────────────

def run_m_paths(model, processor, image: Image.Image,
                prompt_for_path, qtype: str,
                m: int = M_PATHS,
                opts: list = None, perms: list = None) -> dict:
    """
    Generate m reasoning paths and aggregate them.

    `prompt_for_path(i)` returns the prompt for path i. For MCQ this
    varies the option ORDER per path (see option_permutations) at no
    extra cost; for factoid every path gets the same prompt.

    Voting happens on CANONICAL answers:
      - factoid: the normalized answer. Voting on raw strings made '85',
        '85%' and '[85]' three distinct answers, so paths that agreed
        registered as maximum disagreement — corrupting the majority
        answer AND inflating the entropy Eq. 6-7 filters the pool on.
      - MCQ: the canonical option letter, mapped back out of whatever
        permuted order that path was shown. Without this remap the votes
        from different orderings would be meaningless.

    Returns a dict:
      answer          str   — majority answer, canonical form
      answer_display  str   — majority answer as the model wrote it
      raw_outputs     list  — every generation (logged for re-scoring)
      extracted       list  — per-path extracted span
      normalized      list  — per-path canonical answer
      rationale       str   — path with most steps (Eq. 5)
      uncertainty     float — Shannon entropy, normalized to [0,1] (Eq. 1-3)
      complexity      float — avg steps of majority paths (Eq. 4)
      agreement       float — share of paths backing the majority
    """
    raw_outputs, extracted, normalized = [], [], []
    is_mcq = qtype == "mcq" and opts and perms

    for i in range(m):
        # EXTENSION greedy_first_path: decode path 0 deterministically.
        # The paper samples every path (T=1.0); T=0 usually gives the
        # single strongest path, and for MCQ the option rotation already
        # supplies the diversity self-consistency needs.
        temp = 0.0 if (i == 0 and ext("greedy_first_path")) else TEMPERATURE
        raw = call_vlm(model, processor, image, prompt_for_path(i), temp)
        span = extract_answer(raw, qtype)
        raw_outputs.append(raw)
        extracted.append(span)
        if is_mcq:
            normalized.append(canonical_mcq_answer(span, opts, perms[i]))
        else:
            normalized.append(normalize_answer(span, qtype))

    # EXTENSION drop_malformed_paths: a path that never emitted a
    # 'Final Answer:' line is being read by the last-line fallback, which
    # is noise voting at full weight. Exclude those — unless every path
    # was malformed, in which case we have nothing else to go on.
    voting_idx = list(range(m))
    if ext("drop_malformed_paths"):
        well_formed = [i for i in range(m) if has_final_answer(raw_outputs[i])]
        if well_formed:
            voting_idx = well_formed

    # Aggregate (Eq. 1–3, or the clustering extension). Lives in
    # scoring.py so ablate.py can recompute this offline with no GPU.
    agg = aggregate_answers(
        [normalized[i] for i in voting_idx],
        tolerance=NUMERIC_TOLERANCE,
        use_clustering=(not is_mcq) and ext("numeric_vote_clustering"),
    )
    majority = agg["winner"]

    # Map member positions back to original path indices.
    majority_idx = [voting_idx[j] for j in agg["members"]] or list(range(m))
    majority_paths = [raw_outputs[i] for i in majority_idx]

    step_counts = [count_steps(r) for r in majority_paths]
    complexity = sum(step_counts) / len(step_counts) if step_counts else 0.0

    best_rationale = max(majority_paths, key=count_steps) if majority_paths \
        else (raw_outputs[0] if raw_outputs else "")

    # For MCQ the per-path extracted span is a letter in that path's
    # PERMUTED space, so reporting it verbatim would be wrong. Report the
    # canonical letter instead. For factoid, show it as the model wrote it —
    # except under clustering, where the winner is a computed median that
    # no single path necessarily produced.
    if is_mcq:
        display = majority.upper() if len(majority) == 1 else majority
    elif agg["mode"] == "cluster":
        display = majority
    else:
        display = extracted[majority_idx[0]] if majority_idx else majority

    return {
        "answer":         majority,
        "answer_display": display,
        "raw_outputs":    raw_outputs,
        "extracted":      extracted,
        "normalized":     normalized,
        "rationale":      best_rationale,
        "uncertainty":    agg["uncertainty"],
        "complexity":     complexity,
        "agreement":      agg["agreement"],
        "vote_mode":      agg["mode"],
        "n_groups":       agg["n_groups"],
        "paths_voting":   len(voting_idx),
    }


# ─────────────────────────────────────────────────────────────
# STEP 8 — EXPERIENCE POOL  (paper §3.1 + Algorithm 1)
# ─────────────────────────────────────────────────────────────

class ExperiencePool:
    """
    Streaming experience pool.
    Stores triplets (question, rationale, answer) with
    uncertainty and complexity attributes.
    """

    def __init__(self, embedder: SentenceTransformer):
        self.embedder     = embedder
        self.experiences  = []   # list of dicts
        self._embeddings  = None # np.ndarray shape (N, D)

    def _encode(self, question: str) -> np.ndarray:
        """Encode to a flat (D,) vector — some backends return (1, D)."""
        emb = self.embedder.encode(question, convert_to_numpy=True)
        return np.asarray(emb).reshape(-1)

    def add(self, question: str, rationale: str, answer: str,
            qtype: str, uncertainty: float, complexity: float):
        emb = self._encode(question)
        self.experiences.append({
            "question":    question,
            "rationale":   rationale,
            "answer":      answer,
            "qtype":       qtype,
            "uncertainty": uncertainty,
            "complexity":  complexity,
            "embedding":   emb,
        })
        self._embeddings = np.stack(
            [e["embedding"] for e in self.experiences]
        )

    def size(self) -> int:
        return len(self.experiences)

    # ── Algorithm 1: Partition ──────────────────────────────

    def _partition(self, sim_sorted_indices: list, k: int) -> list:
        """
        Algorithm 1 from paper:
        1. Sort by similarity (already done — sim_sorted_indices is low→high).
        2. Uniformly divide into k buckets.
        3. If any bucket is empty, split the largest bucket in 2.
        4. Return list of buckets (each bucket = list of pool indices).
        """
        n = len(sim_sorted_indices)
        if n == 0:
            return []

        # Initial uniform partition
        bucket_size = max(1, n // k)
        buckets = []
        for b in range(k):
            start = b * bucket_size
            end   = start + bucket_size if b < k - 1 else n
            buckets.append(list(sim_sorted_indices[start:end]))

        # Remove truly empty buckets, then split the largest until len==k
        buckets = [b for b in buckets if b]
        while len(buckets) < k and any(len(b) > 1 for b in buckets):
            largest_idx = max(range(len(buckets)),
                              key=lambda i: len(buckets[i]))
            largest = buckets.pop(largest_idx)
            mid = len(largest) // 2
            buckets.append(largest[:mid])
            buckets.append(largest[mid:])
            buckets = [b for b in buckets if b]

        return buckets

    # ── Uncertainty-based filtering (Eq. 6–7) ──────────────

    def _filter_by_uncertainty(self, bucket: list) -> list:
        """
        Keep only experiences with uncertainty ≤ λ × min_uncertainty_in_bucket.
        Always keep at least one experience (the one with min uncertainty).
        """
        uncertainties = [self.experiences[i]["uncertainty"] for i in bucket]
        u_min  = min(uncertainties)
        thresh = LAMBDA * u_min
        filtered = [i for i in bucket
                    if self.experiences[i]["uncertainty"] <= thresh]
        if not filtered:
            best = min(bucket, key=lambda i: self.experiences[i]["uncertainty"])
            filtered = [best]
        return filtered

    # ── Complexity-based selection (Eq. 8) ─────────────────

    def _select_from_bucket(self, bucket: list) -> dict:
        """Pick experience with highest complexity from bucket."""
        best_idx = max(bucket,
                       key=lambda i: self.experiences[i]["complexity"])
        return self.experiences[best_idx]

    # ── Main orchestration method ───────────────────────────

    def orchestrate(self, question: str,
                    k: int = K_DEMONSTRATIONS, qtype: str = None) -> list:
        """
        Full RoSE orchestration for a new question:
        relevance → diversity → uncertainty filtering → complexity selection.
        Returns a list of up to k experience dicts.
        """
        if self.size() == 0:
            return []

        q_emb = self._encode(question)
        sims = self._embeddings.dot(q_emb) / (
            np.linalg.norm(self._embeddings, axis=1) *
            np.linalg.norm(q_emb) + 1e-12
        )

        # EXTENSION type_aware_retrieval: restrict candidates to the same
        # question type. The paper's pool is single-task; ours holds both
        # factoid and MCQ, so a factoid question can otherwise be shown MCQ
        # demonstrations that teach the wrong output format. Falls back to
        # the whole pool while there are too few same-type experiences.
        eligible = np.arange(self.size())
        if qtype and ext("type_aware_retrieval"):
            same = np.array([i for i in range(self.size())
                             if self.experiences[i].get("qtype") == qtype],
                            dtype=int)
            if len(same) >= k:
                eligible = same

        n = min(len(eligible), 3 * k)   # candidate pool: top 3k MOST similar

        # Take the n most similar experiences as candidates, then order them
        # low→high similarity so the buckets span a diversity range.
        #
        # The previous code did `np.argsort(sims)[:n]`, which is ascending —
        # it selected the n LEAST similar experiences in the whole pool, so
        # the demonstrations were the most irrelevant ones available. That
        # was invisible while the pool was small (n == pool size) and got
        # steadily worse as the stream grew.
        order = eligible[np.argsort(sims[eligible])]   # ascending similarity
        sorted_indices = order[-n:].tolist()           # n most similar, asc

        buckets = self._partition(sorted_indices, k)

        demonstrations = []
        for bucket in buckets[:k]:
            filtered = self._filter_by_uncertainty(bucket)   # Eq. 6-7
            selected = self._select_from_bucket(filtered)     # Eq. 8
            demonstrations.append(selected)

        return demonstrations[:k]


# ─────────────────────────────────────────────────────────────
# STEP 10 — PROCESS ONE SAMPLE  (the main per-question logic)
# ─────────────────────────────────────────────────────────────

def process_one(sample: dict, model, processor,
                pool: ExperiencePool) -> dict:
    """
    Run RoSE on a single ChartQAPro question.

    Phase A (pool < k): zero-shot CoT, add to pool.
    Phase B (pool ≥ k): orchestrate demonstrations, few-shot CoT, add to pool.
    """
    question = sample["question"]
    truth    = sample.get("answer", "")
    qtype    = sample.get("question_type", "factoid").lower().strip()
    choices  = sample.get("choices", None)

    image = load_image(sample.get("image", ""))

    if pool.size() < K_DEMONSTRATIONS:
        method = "zero_shot_cot"
        demos  = []
        build = lambda perm: zero_shot_prompt(          # noqa: E731
            question, qtype, choices, perm)
    else:
        demos  = pool.orchestrate(question, k=K_DEMONSTRATIONS, qtype=qtype)
        method = "rose_few_shot"
        build = lambda perm: few_shot_prompt(           # noqa: E731
            question, qtype, demos, choices, perm)

    # MCQ: rotate the option order across the m paths we already pay for,
    # so option-position bias cancels in the vote. Costs nothing extra.
    if qtype == "mcq" and choices and ext("mcq_permute_options"):
        _, opts, _ = build_options(choices)
        perms = option_permutations(len(opts), M_PATHS)
        prompt_for_path = lambda i: build(perms[i])      # noqa: E731
    else:
        _, opts, _ = (build_options(choices) if (qtype == "mcq" and choices)
                      else (None, None, None))
        perms = None
        base_prompt = build(None)
        prompt_for_path = lambda i: base_prompt          # noqa: E731

    out = run_m_paths(model, processor, image, prompt_for_path, qtype,
                      m=M_PATHS, opts=opts, perms=perms)

    # What this answer looks like as a DEMONSTRATION for a later question.
    # A bare 'B' is useless in a few-shot example, because the next
    # question has different options and the letter is unanchored. Store
    # '(B) 45%' so the example conveys both the required output shape and
    # the actual value.
    demo_answer = out["answer_display"]
    if qtype == "mcq" and opts:
        idx = resolve_choice(out["answer_display"], opts)
        if idx is not None:
            demo_answer = f"({chr(65 + idx)}) {opts[idx]}"

    # Add to pool (always — RoSE grows its pool on every question)
    pool.add(
        question    = question,
        rationale   = out["rationale"],
        answer      = demo_answer,
        qtype       = qtype,
        uncertainty = out["uncertainty"],
        complexity  = out["complexity"],
    )

    # Pass the option list so gold-as-letter and gold-as-text are
    # equivalent, and so an answer written as the option's text still counts.
    correct = is_correct(out["answer_display"], truth, qtype,
                         choices=opts if qtype == "mcq" else None)

    # Everything needed to re-score offline is logged here: the raw
    # generations, the extracted span, and the normalized form. No future
    # re-scoring should ever require re-running the GPU.
    return {
        "id":             sample.get("id", ""),
        "question":       question,
        "question_type":  qtype,
        "choices":        choices,
        "options_shown":  opts,          # includes the appended Unanswerable
        "option_perms":   perms,         # per-path display order, for auditing
        "ground_truth":   truth,
        "prediction":     out["answer_display"],
        "prediction_norm": out["answer"],
        "is_correct":     correct,
        "method":         method,
        "uncertainty":    round(out["uncertainty"], 4),
        "complexity":     round(out["complexity"], 4),
        "agreement":      round(out["agreement"], 4),
        "n_demos":        len(demos),
        "pool_size":      pool.size(),
        "best_rationale": out["rationale"],
        "raw_outputs":    out["raw_outputs"],
        "extracted":      out["extracted"],
        "normalized":     out["normalized"],
    }


# ─────────────────────────────────────────────────────────────
# STEP 11 — LOAD + FILTER DATASET
# ─────────────────────────────────────────────────────────────

def load_factoid_mcq(path: str) -> list:
    """Load ChartQAPro and keep only the TARGET_TYPES questions."""
    with open(path) as f:
        data = json.load(f)

    filtered = [
        d for d in data
        if d.get("question_type", "").lower().strip() in TARGET_TYPES
    ]

    print(f"\n[3/3] Dataset loaded")
    print(f"   Total questions : {len(data)}")
    print(f"   Target types    : {len(filtered)}")

    by_type = Counter(d["question_type"].lower() for d in filtered)
    for qt, n in by_type.items():
        print(f"   └─ {qt:<12} : {n}")

    n_unans = sum(1 for d in filtered if is_unanswerable(d.get("answer", "")))
    print(f"   └─ unanswerable gold : {n_unans} "
          f"({n_unans / max(len(filtered), 1) * 100:.1f}%)")

    return filtered


# ─────────────────────────────────────────────────────────────
# STEP 12 — MAIN RUN LOOP
# ─────────────────────────────────────────────────────────────

def run(model=None, processor=None, embedder=None):
    """
    Entry point. Call run() directly in Colab after loading models,
    or call run() with no arguments to load models internally.
    """
    Path(RESULTS_DIR).mkdir(exist_ok=True, parents=True)

    if model is None:
        model, processor, embedder = load_models()

    data = load_factoid_mcq(DATASET_PATH)

    # Resume from checkpoint if it exists
    results   = []
    start_idx = 0
    if Path(CHECKPOINT_FILE).exists():
        with open(CHECKPOINT_FILE) as f:
            results = json.load(f)
        start_idx = len(results)
        print(f"\n↩  Resuming from checkpoint — sample {start_idx}/{len(data)}")

    pool = ExperiencePool(embedder)

    # Pre-fill pool from already-processed results (warm restart).
    # process_one now stores best_rationale, so a resumed run rebuilds the
    # pool with real rationales instead of falling back to the answer text.
    for r in results:
        if r.get("method") == "error":
            continue
        pool.add(
            question    = r["question"],
            rationale   = r.get("best_rationale") or r.get("prediction", ""),
            answer      = r.get("prediction", ""),
            qtype       = r.get("question_type", "factoid"),
            uncertainty = r.get("uncertainty", 0.5),
            complexity  = r.get("complexity", 1.0),
        )

    # ── Main loop ──────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  RoSE on ChartQAPro  |  {'/'.join(sorted(TARGET_TYPES))}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Paths per Q: {M_PATHS}   |   k demonstrations: {K_DEMONSTRATIONS}")
    print(f"  λ = {LAMBDA}   |   tolerance = {NUMERIC_TOLERANCE:.0%}")
    print(f"  Questions: {len(data) - start_idx} remaining")
    print(f"{'=' * 60}")

    for i, sample in enumerate(data[start_idx:], start=start_idx):

        qtype  = sample.get("question_type", "?")
        method = "RoSE" if pool.size() >= K_DEMONSTRATIONS else "ZeroShot"
        print(f"\n[{i + 1:04d}/{len(data)}]  type={qtype}  "
              f"pool={pool.size()}  {method}")

        try:
            result = process_one(sample, model, processor, pool)
            results.append(result)

            mark = "✓" if result["is_correct"] else "✗"
            print(f"  {mark}  pred='{str(result['prediction'])[:50]}'"
                  f"  (norm='{result['prediction_norm']}')")
            print(f"     truth='{result['ground_truth']}'")
            print(f"     u={result['uncertainty']:.3f}  "
                  f"c={result['complexity']:.2f}  "
                  f"agree={result['agreement']:.2f}")

        except Exception as exc:
            print(f"  ⚠  Error: {exc}")
            results.append({
                "id":            sample.get("id", ""),
                "question":      sample.get("question", ""),
                "question_type": sample.get("question_type", ""),
                "ground_truth":  sample.get("answer", ""),
                "prediction":    "ERROR",
                "is_correct":    False,
                "error":         str(exc),
                "method":        "error",
            })

        if (i + 1) % CHECKPOINT_EVERY == 0:
            with open(CHECKPOINT_FILE, "w") as f:
                json.dump(results, f, indent=2)
            valid = [r for r in results if "error" not in r]
            acc = (sum(r.get("is_correct", False) for r in valid)
                   / max(len(valid), 1)) * 100
            print(f"\n  💾 Checkpoint saved ({i + 1} done, "
                  f"running acc={acc:.1f}%)")

    with open(FINAL_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓  Results saved → {FINAL_FILE}")

    # Provenance. A results file must say which configuration produced it,
    # or an ablation table cannot be trusted. Also the file the resume glob
    # (/kaggle/input/**/results/meta.json) looks for.
    meta = {
        "model":            MODEL_NAME,
        "embed_model":      EMBED_MODEL,
        "m_paths":          M_PATHS,
        "k_demonstrations": K_DEMONSTRATIONS,
        "lambda":           LAMBDA,
        "temperature":      TEMPERATURE,
        "max_new_tokens":   MAX_NEW_TOKENS,
        "max_pixels":       MAX_PIXELS,
        "numeric_tolerance": NUMERIC_TOLERANCE,
        "offer_unanswerable": OFFER_UNANSWERABLE,
        "target_types":     sorted(TARGET_TYPES),
        "paper_faithful":   PAPER_FAITHFUL,
        "extensions":       active_extensions(),
        "n_results":        len(results),
        "finished_at":      time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = Path(RESULTS_DIR) / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✓  Run config saved → {meta_path}")
    label = "PAPER-FAITHFUL" if PAPER_FAITHFUL else ", ".join(
        k for k, v in active_extensions().items() if v) or "none"
    print(f"   extensions active: {label}")

    print_results_table(results)
    return results


# ─────────────────────────────────────────────────────────────
# STEP 13 — RESULTS TABLE  (what you show your professor)
# ─────────────────────────────────────────────────────────────

def print_results_table(results: list):
    """
    Accuracy by question type, by method, and by answerable/unanswerable,
    plus the answerable↔unanswerable confusion counts.
    """
    valid = [r for r in results if "error" not in r]
    if not valid:
        print("\n(no valid results to report)")
        return

    by_type   = defaultdict(lambda: {"correct": 0, "total": 0})
    by_method = defaultdict(lambda: {"correct": 0, "total": 0})
    by_ans    = defaultdict(lambda: {"correct": 0, "total": 0})

    # Confusion between "has an answer" and "Unanswerable"
    answerable_said_unans = 0
    unanswerable_said_val = 0

    for r in valid:
        qt      = r.get("question_type", "unknown")
        method  = r.get("method", "unknown")
        ok      = r.get("is_correct", False)
        gold_un = is_unanswerable(r.get("ground_truth", ""))
        pred_un = is_unanswerable(r.get("prediction", ""))
        bucket  = "unanswerable" if gold_un else "answerable"

        for table, key in ((by_type, qt), (by_method, method), (by_ans, bucket)):
            table[key]["total"] += 1
            if ok:
                table[key]["correct"] += 1

        if not gold_un and pred_un:
            answerable_said_unans += 1
        if gold_un and not pred_un:
            unanswerable_said_val += 1

    def _block(title, table):
        print("\n" + "=" * 55)
        print(f"  {title}")
        print("=" * 55)
        for key, s in sorted(table.items()):
            acc = s["correct"] / s["total"] * 100 if s["total"] else 0
            print(f"  {key:<22}  {acc:5.1f}%  ({s['correct']}/{s['total']})")

    _block("ACCURACY BY QUESTION TYPE", by_type)
    total_correct = sum(s["correct"] for s in by_type.values())
    total_all     = sum(s["total"] for s in by_type.values())
    overall = total_correct / total_all * 100 if total_all else 0
    print(f"  {'OVERALL':<22}  {overall:5.1f}%  ({total_correct}/{total_all})")

    _block("ACCURACY BY METHOD  (zero-shot vs RoSE)", by_method)
    _block("ACCURACY BY ANSWERABILITY", by_ans)

    # This is the MCQ diagnostic: a high unanswerable score paired with a
    # low answerable score usually means the model over-uses the escape
    # hatch once it is offered as an option.
    print("\n" + "=" * 55)
    print("  ANSWERABILITY CONFUSION")
    print("=" * 55)
    n_answerable   = by_ans["answerable"]["total"]
    n_unanswerable = by_ans["unanswerable"]["total"]
    print(f"  answerable gold, predicted Unanswerable : "
          f"{answerable_said_unans}/{n_answerable}"
          f"  ({answerable_said_unans / max(n_answerable, 1) * 100:.1f}%)")
    print(f"  unanswerable gold, predicted a value    : "
          f"{unanswerable_said_val}/{n_unanswerable}"
          f"  ({unanswerable_said_val / max(n_unanswerable, 1) * 100:.1f}%)")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    run()
