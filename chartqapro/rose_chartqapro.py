# =============================================================
# RoSE on ChartQAPro — Improved Implementation
# Paper: "Making LLMs Better Reasoners with Orchestrated
#         Streaming Experiences" (EMNLP 2024)
# Target: ChartQAPro Factoid + MCQ + Hypothetical questions
# Model:  Qwen2.5-VL-7B-Instruct, 4-bit (fits a free T4 GPU)
# =============================================================
#
# EXACT PAPER MAPPINGS (unchanged from paper):
#   m = 20 reasoning paths per question (reduced to 5 for free GPU)
#   λ = 1.2 × min_uncertainty per bucket (Eq. 6-7)
#   k = number of demonstrations = number of buckets (default 3)
#   Uncertainty = Shannon entropy (Eq. 1-3)
#   Complexity  = avg CountSteps of majority-answer paths (Eq. 4)
#   Stored path = argmax CountSteps (Eq. 5)
#   Selection   = argmax complexity per bucket (Eq. 8)
#   Inference   = LLM(q1,r1,a1,...,qk,rk,ak,qt) (Eq. 9-10)
#
# IMPROVEMENTS OVER PREVIOUS VERSION:
#   1. MCQ demo anchor fix   — stores option TEXT not "(B) text" in pool
#   2. M_PATHS 3→5           — better self-consistency and entropy estimates
#   3. Two-stage reasoning   — pre-extracts chart structure before QA (NEW)
#   4. CLIP hybrid retrieval — visual+text similarity for pool search (NEW)
#   5. Confidence-weighted vote — upweights greedy + well-formed paths (NEW)
#   6. Chart-type-aware scaffolds — type-specific reading instructions (NEW)
#   7. Hypothetical support  — dedicated prompt template for hypothetical Qs
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
MODEL_NAME       = "Qwen/Qwen2.5-VL-7B-Instruct"
EMBED_MODEL      = "all-mpnet-base-v2"

DATASET_PATH     = "ChartQAPro/data/test.json"
IMAGES_DIR       = "ChartQAPro/data/images"
RESULTS_DIR      = "results"
CHECKPOINT_FILE  = "results/checkpoint.json"
FINAL_FILE       = "results/rose_results.json"

# Paper hyperparameters
# M_PATHS raised from 3→5: with only 3 paths, entropy can take very few
# discrete values, making uncertainty estimates unreliable for pool filtering.
# 5 paths gives meaningfully more stable entropy at manageable GPU cost.
M_PATHS          = 5     # paper uses 20; 5 balances quality vs T4 budget
K_DEMONSTRATIONS = 3     # number of few-shot examples (= number of buckets)
LAMBDA           = 1.2   # uncertainty threshold multiplier (paper §3.2)
TEMPERATURE      = 0.8   # slightly higher than before for path diversity
MAX_NEW_TOKENS   = 300   # reduced from 384; chart answers are concise
DESC_MAX_TOKENS  = 200   # token limit for the two-stage description pass
CHECKPOINT_EVERY = 20    # save progress every N samples

# CLIP visual retrieval: blend weight between text and visual similarity.
# 0.0 = text only (disables visual component), 1.0 = CLIP only.
VISUAL_ALPHA     = 0.35

# Vision resolution
MIN_PIXELS       = 256 * 28 * 28
MAX_PIXELS       = 1280 * 28 * 28
MAX_IMAGE_SIDE   = 1600

# Offer "Unanswerable" as an allowed answer
OFFER_UNANSWERABLE = True

# Filter: only run on these question types.
# Override from a notebook cell: rose_chartqapro.TARGET_TYPES = {"factoid"}
TARGET_TYPES     = {"factoid", "mcq", "hypothetical"}


# ─────────────────────────────────────────────────────────────
# EXTENSIONS — changes that are NOT in the paper
# ─────────────────────────────────────────────────────────────
# Set PAPER_FAITHFUL = True to disable all extensions at once.
# Override a single extension: rose_chartqapro.EXTENSIONS["two_stage_reasoning"] = False
#
PAPER_FAITHFUL = False

EXTENSIONS = {
    # ── FROM ORIGINAL VERSION ──────────────────────────────────
    # Rotate MCQ option order across the m paths (position bias cancels).
    "mcq_permute_options":      True,

    # Group numeric answers within tolerance before voting; take median.
    "numeric_vote_clustering":  True,

    # Retrieve demonstrations only from the same question type.
    "type_aware_retrieval":     True,

    # Exclude paths that never emitted "Final Answer:" from the vote.
    "drop_malformed_paths":     True,

    # Decode path 0 greedily (T=0); sample the rest.
    "greedy_first_path":        True,

    # Add axis-and-units reading scaffold to the prompt.
    "chart_reading_scaffold":   True,

    # ── NEW IMPROVEMENTS ───────────────────────────────────────
    # CRITICAL MCQ FIX: store only the option TEXT (e.g. "2019"), not
    # "(B) 2019", in the pool. The letter "B" is anchored to a specific
    # question's option ordering and actively misleads later questions
    # where option B is a completely different value.
    "mcq_demo_anchor_fix":      True,

    # Two-stage reasoning: run a fast zero-temperature pass to extract
    # chart structure (type, axes, key values) BEFORE the QA paths.
    # That structured description is injected into every inference path,
    # forcing systematic chart reading rather than relying on the model
    # to read and reason in one unchecked pass.
    "two_stage_reasoning":      True,

    # CLIP hybrid retrieval: blend CLIP visual similarity with text
    # similarity when searching the experience pool. Finds demonstrations
    # that look visually similar (same chart type, layout) not just
    # semantically similar in question text.
    "visual_hybrid_retrieval":  True,

    # Confidence-weighted vote: upweight the greedy path (T=0) and
    # well-formed paths (that produced "Final Answer:") in the majority vote.
    "confidence_weighted_vote": True,
}


def ext(name: str) -> bool:
    """True if extension `name` is active."""
    if PAPER_FAITHFUL:
        return False
    return bool(EXTENSIONS.get(name, False))


def active_extensions() -> dict:
    """Written into meta.json so every run is self-describing."""
    return {k: ext(k) for k in EXTENSIONS}


# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD MODELS
# ─────────────────────────────────────────────────────────────

def load_models():
    """Load VLM (4-bit, GPU) and sentence embedder (CPU)."""
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

    mem   = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"   ✓ VLM loaded  ({mem:.1f}/{total:.1f} GB used)")

    print("[2/3] Loading sentence embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL)
    print("   ✓ Embedder loaded")

    return model, processor, embedder


def load_clip():
    """
    NEW — Load CLIP for visual hybrid retrieval.
    CLIP is lightweight (ViT-B/32, ~350 MB) and runs on CPU so it needs
    no extra VRAM. Returns (None, None) if the extension is disabled or
    the package is unavailable, so the rest of the code degrades gracefully.
    """
    if not ext("visual_hybrid_retrieval"):
        return None, None
    try:
        from transformers import CLIPModel, CLIPProcessor
        print("[3/3] Loading CLIP for visual retrieval...")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model.eval()   # CPU, no VRAM used
        print("   ✓ CLIP loaded (CPU)")
        return clip_model, clip_proc
    except Exception as e:
        print(f"   ⚠  CLIP load failed ({e}). Visual retrieval disabled.")
        return None, None


# ─────────────────────────────────────────────────────────────
# STEP 2 — IMAGE LOADER
# ─────────────────────────────────────────────────────────────

def load_image(image_field) -> Image.Image:
    """
    Find and open the chart image.
    Always converts to RGB and downscales oversized charts so a single
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

    longest = max(img.size)
    if longest > MAX_IMAGE_SIDE:
        scale    = MAX_IMAGE_SIDE / longest
        new_size = (max(1, int(img.width * scale)),
                    max(1, int(img.height * scale)))
        img = img.resize(new_size, Image.LANCZOS)

    return img


# ─────────────────────────────────────────────────────────────
# STEP 3 — VLM CALL  (single forward pass)
# ─────────────────────────────────────────────────────────────

def call_vlm(model, processor, image: Image.Image,
             prompt: str, temperature: float = TEMPERATURE,
             max_new_tokens: int = None) -> str:
    """
    One forward pass through Qwen2.5-VL. Returns raw output string.
    max_new_tokens defaults to the global MAX_NEW_TOKENS but can be
    overridden (the description pass uses DESC_MAX_TOKENS).
    """
    tokens = max_new_tokens if max_new_tokens is not None else MAX_NEW_TOKENS

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

    gen_kwargs = dict(max_new_tokens=tokens)
    if temperature and temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
    else:
        gen_kwargs.update(do_sample=False)

    with torch.no_grad():
        out_ids = model.generate(**inputs, **gen_kwargs)

    new_tokens = out_ids[0][inputs["input_ids"].shape[1]:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────────────────────
# STEP 3b — TWO-STAGE CHART READING  (NEW EXTENSION)
# ─────────────────────────────────────────────────────────────

# First-stage prompt: extract chart structure at T=0 (deterministic).
# This is called ONCE per question and the result is injected into
# ALL m inference paths, so the model commits to what it sees before
# reasoning rather than reading and reasoning in one unchecked pass.
_DESCRIPTION_PROMPT = (
    "Analyze this chart and extract the following information:\n\n"
    "CHART_TYPE: (bar / line / pie / scatter / table / other)\n"
    "X_AXIS: axis label and representative values "
    "(e.g. 'Year: 2018, 2019, 2020')\n"
    "Y_AXIS: axis label and units "
    "(e.g. 'Revenue (USD billions)')\n"
    "SERIES: list of series or category names "
    "(e.g. 'Region A, Region B, Total')\n"
    "KEY_VALUES: 3-5 notable data points "
    "(e.g. 'Region A 2020: 4.5B; max value: 12.1B')\n\n"
    "Be precise and concise. "
    "If you cannot read a label clearly, write UNCLEAR. "
    "Do NOT answer any question — only describe the chart structure."
)

# Type-specific reading scaffolds: different chart types require
# different reading strategies. Detected from the first-stage description.
_CHART_TYPE_SCAFFOLDS = {
    "bar":     ("Compare bar heights carefully. "
                "Note whether bars are grouped or stacked."),
    "line":    ("Follow each line across its full range. "
                "Note peaks, troughs, and crossover points."),
    "pie":     ("Identify each slice's label and approximate proportion. "
                "Sum of all slices = 100%."),
    "scatter": ("Locate individual data points using both axes. "
                "Note axis ranges and any visible clusters."),
    "table":   ("Read row headers, then column headers, "
                "then find the intersection cell."),
    "other":   ("Read all axis labels and units carefully "
                "before extracting any value."),
}

# Fallback scaffold (used when two_stage_reasoning is off)
_SCAFFOLD = (
    "Before answering: read the axis labels and their units, identify which "
    "series or category the question is about, then read the value.\n\n"
)


def get_chart_description(model, processor,
                           image: Image.Image) -> str:
    """
    EXTENSION two_stage_reasoning:
    First-stage pass at T=0 to extract structured chart metadata.
    Called ONCE per question. Takes ~25-30 s on a T4.
    Returns empty string when the extension is disabled.
    """
    if not ext("two_stage_reasoning"):
        return ""
    try:
        return call_vlm(
            model, processor, image,
            _DESCRIPTION_PROMPT,
            temperature=0.0,
            max_new_tokens=DESC_MAX_TOKENS,
        )
    except Exception as e:
        print(f"   ⚠  Chart description failed: {e}")
        return ""


def _chart_scaffold_from_desc(chart_desc: str) -> str:
    """
    Choose a chart-type-specific reading scaffold based on the
    CHART_TYPE line of the first-stage description.
    Falls back to the generic scaffold when type is unclear.
    """
    if chart_desc and ext("chart_reading_scaffold"):
        lower = chart_desc.lower()
        for chart_type, scaffold in _CHART_TYPE_SCAFFOLDS.items():
            # Only check the first ~120 chars = the CHART_TYPE line
            if chart_type in lower[:120]:
                return scaffold + "\n\n"
    # Generic scaffold (original behaviour)
    return _SCAFFOLD if ext("chart_reading_scaffold") else ""


# ─────────────────────────────────────────────────────────────
# STEP 3c — CLIP VISUAL EMBEDDING  (NEW EXTENSION)
# ─────────────────────────────────────────────────────────────

def compute_clip_embedding(image: Image.Image,
                           clip_model=None,
                           clip_processor=None) -> "np.ndarray | None":
    """
    EXTENSION visual_hybrid_retrieval:
    Return an L2-normalised CLIP (512-d) visual embedding. CPU only.
    Returns None when the extension is off or CLIP is unavailable.
    """
    if not ext("visual_hybrid_retrieval"):
        return None
    if clip_model is None or clip_processor is None:
        return None
    try:
        inputs = clip_processor(
            images=image, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            feats = clip_model.get_image_features(**inputs)  # (1, 512)
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
        return feats.squeeze().cpu().numpy()
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# STEP 4 — PROMPTS  (zero-shot CoT and few-shot CoT)
# ─────────────────────────────────────────────────────────────

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

_MCQ_FORMAT = (
    "Choose Unanswerable ONLY if the chart genuinely does not contain the "
    "information needed. If the chart does show the information — even if "
    "you have to read or estimate a value from an axis or a bar — choose "
    "the closest option instead.\n\n"
    "Note: the example answers show the selected VALUE (not just a letter), "
    "so you can see what type of answer is expected.\n\n"
    "End your response with exactly this line:\n"
    "Final Answer: (X)\n"
    "where X is the letter of the correct option."
)

# Hypothetical questions require reading real values from the chart
# THEN applying a specified change and computing the result.
_HYPOTHETICAL_FORMAT = (
    "This question describes a hypothetical scenario.\n"
    "Step 1: Read the ACTUAL values from the chart that are relevant.\n"
    "Step 2: Apply the hypothetical change described in the question.\n"
    "Step 3: Calculate the result clearly.\n\n"
    "Answer with a single bare value (no units, no commas).\n"
    "If the question cannot be answered from the chart, write: Unanswerable\n\n"
    "End your response with exactly this line:\n"
    "Final Answer: <value>"
)


def build_options(choices: list, perm: list = None):
    """
    Render the option block with optional permuted order.
    Returns (rendered_text, canonical_options_list, perm).
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
    m rotational permutations so option-position bias cancels in the vote.
    perm[displayed_index] = canonical_index.
    Rotation 0 is the identity (dataset's own ordering).
    """
    if n_options <= 1:
        return [[0]] * m
    return [
        [(d + r) % n_options for d in range(n_options)]
        for r in (i % n_options for i in range(m))
    ]


def canonical_mcq_answer(span: str, opts: list, perm: list) -> str:
    """
    Map a model answer in PERMUTED option space to a canonical option letter
    so votes from different orderings are directly comparable.
    """
    if not opts:
        return normalize_answer(span, "mcq")
    displayed_idx = resolve_choice(span, [opts[perm[d]] for d in range(len(perm))])
    if displayed_idx is not None:
        return chr(65 + perm[displayed_idx]).lower()
    return normalize_answer(span, "mcq")


def _answer_format(qtype: str) -> str:
    """Return the correct output-format block for each question type."""
    if qtype == "mcq":
        return _MCQ_FORMAT
    if qtype == "hypothetical":
        return _HYPOTHETICAL_FORMAT
    return _FACTOID_FORMAT


def zero_shot_prompt(question: str, qtype: str,
                     choices: list = None, perm: list = None,
                     chart_desc: str = "") -> str:
    """
    Zero-Shot-CoT prompt used when the pool is empty (paper §3.3).
    IMPROVED: injects chart_desc from the two-stage pass when available,
    and routes to a chart-type-specific scaffold.
    """
    scaffold = _chart_scaffold_from_desc(chart_desc)
    ctx = (f"Chart Structure:\n{chart_desc}\n\n") if chart_desc else ""
    header = f"Look at the chart carefully and answer the question.\n\n{ctx}{scaffold}"

    if qtype == "mcq" and choices:
        opts_text, _, _ = build_options(choices, perm)
        return (
            f"{header}Question: {question}\n\nOptions:\n{opts_text}\n\n"
            f"Let's think step by step.\n\n{_MCQ_FORMAT}"
        )
    return (
        f"{header}Question: {question}\n\n"
        f"Let's think step by step.\n\n{_answer_format(qtype)}"
    )


def few_shot_prompt(question: str, qtype: str,
                    demonstrations: list,
                    choices: list = None, perm: list = None,
                    chart_desc: str = "") -> str:
    """
    Few-Shot-CoT prompt — paper Eq. 9: LLM(q1,r1,a1,...,qk,rk,ak,qt).
    IMPROVED:
      - Injects chart_desc from the two-stage pass when available.
      - Adds a note that MCQ demo answers show the VALUE not just a letter
        (reinforces the mcq_demo_anchor_fix: demos store text, not "(B) text").
      - Routes to chart-type-specific scaffold.
    """
    scaffold = _chart_scaffold_from_desc(chart_desc)
    ctx = (f"Chart Structure:\n{chart_desc}\n\n") if chart_desc else ""
    header = (
        "You are an expert at reading and understanding charts. "
        "Here are some examples of chart questions with step-by-step reasoning:\n\n"
    ) + ctx + scaffold

    examples = ""
    for i, d in enumerate(demonstrations, 1):
        examples += (
            f"[Example {i}]\n"
            f"Q: {d['question']}\n"
            f"Reasoning: {d['rationale']}\n"
            f"A: {d['answer']}\n\n"
        )

    if qtype == "mcq" and choices:
        opts_text, _, _ = build_options(choices, perm)
        task = (
            "Now answer the question about the chart above:\n\n"
            f"Q: {question}\n\nOptions:\n{opts_text}\n\n"
            f"Let's think step by step.\n\n{_MCQ_FORMAT}"
        )
    else:
        task = (
            "Now answer the question about the chart above:\n\n"
            f"Q: {question}\n\n"
            f"Let's think step by step.\n\n{_answer_format(qtype)}"
        )

    return header + examples + task


# ─────────────────────────────────────────────────────────────
# STEP 6 — COUNT STEPS  (paper Eq. 4–5)
# ─────────────────────────────────────────────────────────────

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
    A line counts when it contains a reasoning cue or arithmetic.
    """
    count = 0
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if _STEP_RE.search(line) or _ARITHMETIC_RE.search(line):
            count += 1
    return max(count, 1)


# ─────────────────────────────────────────────────────────────
# STEP 7 — SELF-CONSISTENCY + UNCERTAINTY (paper Eq. 1–3)
# ─────────────────────────────────────────────────────────────

def run_m_paths(model, processor, image: Image.Image,
                prompt_for_path, qtype: str,
                m: int = M_PATHS,
                opts: list = None, perms: list = None) -> dict:
    """
    Generate m reasoning paths and aggregate them.

    IMPROVEMENTS over original:
      - confidence_weighted_vote: greedy path (i=0) and well-formed paths
        get a higher vote weight, so they outweigh noisy sampled paths.
      - drop_malformed_paths: unchanged (exclude no-Final-Answer paths).
      - aggregate_answers receives weights when confidence_weighted_vote is on.
    """
    raw_outputs, extracted, normalized = [], [], []
    is_mcq = qtype == "mcq" and opts and perms

    for i in range(m):
        # EXTENSION greedy_first_path: decode path 0 deterministically.
        temp = 0.0 if (i == 0 and ext("greedy_first_path")) else TEMPERATURE
        raw  = call_vlm(model, processor, image, prompt_for_path(i), temp)
        span = extract_answer(raw, qtype)
        raw_outputs.append(raw)
        extracted.append(span)
        if is_mcq:
            normalized.append(canonical_mcq_answer(span, opts, perms[i]))
        else:
            normalized.append(normalize_answer(span, qtype))

    # EXTENSION drop_malformed_paths
    voting_idx = list(range(m))
    if ext("drop_malformed_paths"):
        well_formed = [i for i in range(m) if has_final_answer(raw_outputs[i])]
        if well_formed:
            voting_idx = well_formed

    # EXTENSION confidence_weighted_vote (NEW):
    # Assign a weight to each voting path.
    #   - Greedy path (i=0): 1.5x weight — usually the strongest single answer
    #   - Well-formed path (has "Final Answer:"): 1.3x multiplier
    #   - Malformed path: 0.7x multiplier (already filtered if drop_malformed on)
    # Weights are passed to aggregate_answers; entropy is still computed from
    # unweighted counts to preserve the paper formula.
    path_weights = None
    if ext("confidence_weighted_vote"):
        path_weights = []
        for i in voting_idx:
            w = 1.5 if (i == 0 and ext("greedy_first_path")) else 1.0
            w *= 1.3 if has_final_answer(raw_outputs[i]) else 0.7
            path_weights.append(w)

    agg = aggregate_answers(
        [normalized[i] for i in voting_idx],
        tolerance=NUMERIC_TOLERANCE,
        use_clustering=(not is_mcq) and ext("numeric_vote_clustering"),
        weights=path_weights,   # None → standard majority (paper faithful)
    )
    majority = agg["winner"]

    majority_idx   = [voting_idx[j] for j in agg["members"]] or list(range(m))
    majority_paths = [raw_outputs[i] for i in majority_idx]

    step_counts = [count_steps(r) for r in majority_paths]
    complexity  = sum(step_counts) / len(step_counts) if step_counts else 0.0

    best_rationale = (max(majority_paths, key=count_steps)
                      if majority_paths else (raw_outputs[0] if raw_outputs else ""))

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
    Streaming experience pool with optional CLIP visual hybrid retrieval.

    IMPROVEMENTS over original:
      - add() now accepts clip_emb (optional CLIP visual embedding).
      - orchestrate() accepts q_clip_emb and uses hybrid similarity
        when visual_hybrid_retrieval is active.
      - _hybrid_sims() blends text cosine with CLIP cosine.
    """

    def __init__(self, embedder: SentenceTransformer):
        self.embedder    = embedder
        self.experiences = []          # list of dicts
        self._embeddings = None        # np.ndarray (N, D_text)

    def _encode(self, question: str) -> np.ndarray:
        emb = self.embedder.encode(question, convert_to_numpy=True)
        return np.asarray(emb).reshape(-1)

    def add(self, question: str, rationale: str, answer: str,
            qtype: str, uncertainty: float, complexity: float,
            clip_emb: "np.ndarray | None" = None):
        """
        Add one experience to the pool.
        clip_emb: CLIP visual embedding (512-d, L2-normalised), or None.
        Passing None for clip_emb is always safe; the hybrid similarity
        will use a zero vector for that entry and degrade gracefully.
        """
        emb = self._encode(question)
        self.experiences.append({
            "question":    question,
            "rationale":   rationale,
            "answer":      answer,
            "qtype":       qtype,
            "uncertainty": uncertainty,
            "complexity":  complexity,
            "embedding":   emb,
            "clip_emb":    clip_emb,   # may be None
        })
        self._embeddings = np.stack(
            [e["embedding"] for e in self.experiences]
        )

    def size(self) -> int:
        return len(self.experiences)

    def _hybrid_sims(self, q_text_emb: np.ndarray,
                     q_clip_emb: "np.ndarray | None") -> np.ndarray:
        """
        EXTENSION visual_hybrid_retrieval (NEW):
        Blend text cosine similarity with CLIP visual cosine similarity.
          hybrid = (1 - VISUAL_ALPHA) × text_sim + VISUAL_ALPHA × visual_sim
        Falls back to text-only when the extension is off or clip_emb is None.
        Pool entries without a clip_emb (None) are treated as zero vectors,
        meaning they contribute no visual similarity — they still participate
        via their text similarity.
        """
        text_sims = self._embeddings.dot(q_text_emb) / (
            np.linalg.norm(self._embeddings, axis=1) *
            np.linalg.norm(q_text_emb) + 1e-12
        )

        if not ext("visual_hybrid_retrieval") or q_clip_emb is None:
            return text_sims

        # Stack CLIP embeddings; fill missing ones with zeros.
        clip_embs = np.array([
            e["clip_emb"] if e["clip_emb"] is not None
            else np.zeros(512, dtype=np.float32)
            for e in self.experiences
        ])
        q_norm   = np.linalg.norm(q_clip_emb) + 1e-12
        row_norm = np.linalg.norm(clip_embs, axis=1, keepdims=True) + 1e-12
        visual_sims = (clip_embs / row_norm) @ (q_clip_emb / q_norm)

        return (1.0 - VISUAL_ALPHA) * text_sims + VISUAL_ALPHA * visual_sims

    # ── Algorithm 1: Partition ──────────────────────────────

    def _partition(self, sim_sorted_indices: list, k: int) -> list:
        """Algorithm 1: uniform bucket partition (low→high similarity)."""
        n = len(sim_sorted_indices)
        if n == 0:
            return []
        bucket_size = max(1, n // k)
        buckets = []
        for b in range(k):
            start = b * bucket_size
            end   = start + bucket_size if b < k - 1 else n
            buckets.append(list(sim_sorted_indices[start:end]))
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

    # ── Eq. 6-7: Uncertainty filtering ─────────────────────

    def _filter_by_uncertainty(self, bucket: list) -> list:
        """Keep only experiences with u ≤ λ × min_u in bucket."""
        u_min  = min(self.experiences[i]["uncertainty"] for i in bucket)
        thresh = LAMBDA * u_min
        filtered = [i for i in bucket
                    if self.experiences[i]["uncertainty"] <= thresh]
        if not filtered:
            best = min(bucket, key=lambda i: self.experiences[i]["uncertainty"])
            filtered = [best]
        return filtered

    # ── Eq. 8: Complexity selection ─────────────────────────

    def _select_from_bucket(self, bucket: list) -> dict:
        """Pick experience with highest complexity from bucket."""
        best_idx = max(bucket,
                       key=lambda i: self.experiences[i]["complexity"])
        return self.experiences[best_idx]

    # ── Main orchestration method ───────────────────────────

    def orchestrate(self, question: str,
                    k: int = K_DEMONSTRATIONS,
                    qtype: str = None,
                    q_clip_emb: "np.ndarray | None" = None) -> list:
        """
        Full RoSE orchestration: relevance → diversity → uncertainty → complexity.

        IMPROVEMENT: q_clip_emb enables visual hybrid similarity when
        visual_hybrid_retrieval is active.
        """
        if self.size() == 0:
            return []

        q_emb = self._encode(question)
        sims  = self._hybrid_sims(q_emb, q_clip_emb)   # text or hybrid

        # EXTENSION type_aware_retrieval (unchanged)
        eligible = np.arange(self.size())
        if qtype and ext("type_aware_retrieval"):
            same = np.array([i for i in range(self.size())
                             if self.experiences[i].get("qtype") == qtype],
                            dtype=int)
            if len(same) >= k:
                eligible = same

        n     = min(len(eligible), 3 * k)
        order = eligible[np.argsort(sims[eligible])]   # ascending
        sorted_indices = order[-n:].tolist()            # n most similar, asc

        buckets = self._partition(sorted_indices, k)

        demonstrations = []
        for bucket in buckets[:k]:
            filtered = self._filter_by_uncertainty(bucket)
            selected = self._select_from_bucket(filtered)
            demonstrations.append(selected)

        return demonstrations[:k]


# ─────────────────────────────────────────────────────────────
# STEP 10 — PROCESS ONE SAMPLE  (main per-question logic)
# ─────────────────────────────────────────────────────────────

def process_one(sample: dict, model, processor,
                pool: ExperiencePool,
                clip_model=None, clip_processor=None) -> dict:
    """
    Run RoSE on a single ChartQAPro question.

    Phase A (pool < k): zero-shot CoT, add to pool.
    Phase B (pool ≥ k): orchestrate demonstrations, few-shot CoT, add to pool.

    IMPROVEMENTS over original:
      1. Calls get_chart_description() once (two_stage_reasoning).
      2. Computes CLIP embedding once (visual_hybrid_retrieval).
      3. Passes chart_desc to all prompt builders.
      4. Passes q_clip_emb to pool.orchestrate().
      5. Applies MCQ demo anchor fix when storing to pool.
      6. Stores clip_emb in pool.add().
      7. Returns chart_description + vote_mode + n_groups + paths_voting.
    """
    question = sample["question"]
    truth    = sample.get("answer", "")
    qtype    = sample.get("question_type", "factoid").lower().strip()
    choices  = sample.get("choices", None)

    image = load_image(sample.get("image", ""))

    # ── IMPROVEMENT 1: Two-stage — extract chart structure once ──
    chart_desc = get_chart_description(model, processor, image)

    # ── IMPROVEMENT 2: CLIP embedding for hybrid retrieval ────────
    clip_emb = compute_clip_embedding(image, clip_model, clip_processor)

    # ── Build prompt function ─────────────────────────────────────
    if pool.size() < K_DEMONSTRATIONS:
        method = "zero_shot_cot"
        demos  = []
        def build(perm):                                        # noqa: E731
            return zero_shot_prompt(question, qtype, choices, perm, chart_desc)
    else:
        demos  = pool.orchestrate(
            question, k=K_DEMONSTRATIONS,
            qtype=qtype, q_clip_emb=clip_emb          # IMPROVEMENT 4
        )
        method = "rose_few_shot"
        def build(perm):                                        # noqa: E731
            return few_shot_prompt(
                question, qtype, demos, choices, perm, chart_desc
            )

    # ── MCQ: rotate options across m paths ───────────────────────
    if qtype == "mcq" and choices and ext("mcq_permute_options"):
        _, opts, _ = build_options(choices)
        perms = option_permutations(len(opts), M_PATHS)
        prompt_for_path = lambda i: build(perms[i])             # noqa: E731
    else:
        _, opts, _ = (build_options(choices) if (qtype == "mcq" and choices)
                      else (None, None, None))
        perms     = None
        base_prompt = build(None)
        prompt_for_path = lambda i: base_prompt                 # noqa: E731

    out = run_m_paths(model, processor, image, prompt_for_path, qtype,
                      m=M_PATHS, opts=opts, perms=perms)

    # ── IMPROVEMENT 5: MCQ demo anchor fix ────────────────────────
    # ORIGINAL (broken): stores "(B) 2019". The letter "B" is specific to
    # THIS question's option ordering. A later question where option B is
    # "15%" will be misled into picking B because the demo said "(B) ...".
    #
    # FIX: store only the option TEXT ("2019"). The demo now teaches
    # "what kind of value to produce" without anchoring to a position letter.
    demo_answer = out["answer_display"]
    if qtype == "mcq" and opts:
        idx = resolve_choice(out["answer_display"], opts)
        if idx is not None:
            if ext("mcq_demo_anchor_fix"):
                demo_answer = opts[idx]                 # TEXT only — the fix
            else:
                demo_answer = f"({chr(65 + idx)}) {opts[idx]}"   # old behaviour

    # ── Add to pool ───────────────────────────────────────────────
    pool.add(
        question    = question,
        rationale   = out["rationale"],
        answer      = demo_answer,
        qtype       = qtype,
        uncertainty = out["uncertainty"],
        complexity  = out["complexity"],
        clip_emb    = clip_emb,            # IMPROVEMENT 6
    )

    correct = is_correct(out["answer_display"], truth, qtype,
                         choices=opts if qtype == "mcq" else None)

    return {
        "id":              sample.get("id", ""),
        "question":        question,
        "question_type":   qtype,
        "choices":         choices,
        "options_shown":   opts,
        "option_perms":    perms,
        "ground_truth":    truth,
        "prediction":      out["answer_display"],
        "prediction_norm": out["answer"],
        "is_correct":      correct,
        "method":          method,
        "uncertainty":     round(out["uncertainty"], 4),
        "complexity":      round(out["complexity"], 4),
        "agreement":       round(out["agreement"], 4),
        "vote_mode":       out["vote_mode"],      # NEW: "exact"/"cluster"/"weighted"
        "n_groups":        out["n_groups"],        # NEW
        "paths_voting":    out["paths_voting"],    # NEW
        "n_demos":         len(demos),
        "pool_size":       pool.size(),
        "chart_description": chart_desc,          # NEW: for ablation/analysis
        "best_rationale":  out["rationale"],
        "raw_outputs":     out["raw_outputs"],
        "extracted":       out["extracted"],
        "normalized":      out["normalized"],
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
        print(f"   └─ {qt:<14} : {n}")

    n_unans = sum(1 for d in filtered if is_unanswerable(d.get("answer", "")))
    print(f"   └─ unanswerable : {n_unans} "
          f"({n_unans / max(len(filtered), 1) * 100:.1f}%)")

    return filtered


# ─────────────────────────────────────────────────────────────
# STEP 12 — MAIN RUN LOOP
# ─────────────────────────────────────────────────────────────

def run(model=None, processor=None, embedder=None,
        clip_model=None, clip_processor=None):
    """
    Entry point. Call run() with no args to load everything internally,
    or pass pre-loaded models to avoid reloading between calls.

    IMPROVEMENT: now loads CLIP and passes clip_model/clip_processor through
    to process_one() and pool.add().
    """
    Path(RESULTS_DIR).mkdir(exist_ok=True, parents=True)

    if model is None:
        model, processor, embedder = load_models()

    # Load CLIP separately so it can be disabled without changing load_models()
    if clip_model is None:
        clip_model, clip_processor = load_clip()

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
    # clip_emb is not stored in the checkpoint JSON (too large), so we
    # pass None. The pool entry will use text-only similarity for retrieval,
    # which degrades gracefully — it is identical to the original behaviour.
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
            clip_emb    = None,   # not stored in checkpoint; new entries get it
        )

    # ── Banner ─────────────────────────────────────────────────
    active_exts = [k for k, v in active_extensions().items() if v]
    print(f"\n{'=' * 62}")
    print(f"  RoSE on ChartQAPro  |  types: {sorted(TARGET_TYPES)}")
    print(f"  Model   : {MODEL_NAME}")
    print(f"  Paths   : {M_PATHS}   k : {K_DEMONSTRATIONS}   "
          f"λ : {LAMBDA}   T : {TEMPERATURE}")
    print(f"  CLIP    : {'enabled  α=' + str(VISUAL_ALPHA) if clip_model else 'disabled'}")
    print(f"  Exts on : {active_exts or ['none (paper faithful)']}")
    print(f"  Remaining: {len(data) - start_idx} questions")
    print(f"{'=' * 62}")

    for i, sample in enumerate(data[start_idx:], start=start_idx):

        qtype  = sample.get("question_type", "?")
        method = "RoSE" if pool.size() >= K_DEMONSTRATIONS else "ZeroShot"
        print(f"\n[{i + 1:04d}/{len(data)}]  type={qtype}  "
              f"pool={pool.size()}  {method}")

        try:
            result = process_one(
                sample, model, processor, pool,
                clip_model=clip_model, clip_processor=clip_processor
            )
            results.append(result)

            mark = "✓" if result["is_correct"] else "✗"
            print(f"  {mark}  pred='{str(result['prediction'])[:50]}'"
                  f"  (norm='{result['prediction_norm']}')")
            print(f"     truth='{result['ground_truth']}'")
            print(f"     u={result['uncertainty']:.3f}  "
                  f"c={result['complexity']:.2f}  "
                  f"agree={result['agreement']:.2f}  "
                  f"paths_voting={result['paths_voting']}")
            if result.get("chart_description"):
                # Show only the CHART_TYPE line for quick sanity-check
                first_line = result["chart_description"].split("\n")[0]
                print(f"     {first_line}")

        except Exception as exc:
            import traceback
            print(f"  ⚠  Error: {exc}")
            traceback.print_exc()
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

        if (i + 1) % CHECKPOINT_EVERY == 0 or (i + 1) == len(data):
            with open(CHECKPOINT_FILE, "w") as f:
                json.dump(results, f, indent=2)
            valid = [r for r in results
                     if "error" not in r and r.get("method") != "error"]
            acc = (sum(r.get("is_correct", False) for r in valid)
                   / max(len(valid), 1)) * 100
            print(f"\n  💾 Checkpoint saved  [{i + 1}/{len(data)}]  "
                  f"running acc = {acc:.1f}%")

    with open(FINAL_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓  Results saved → {FINAL_FILE}")

    meta = {
        "model":              MODEL_NAME,
        "embed_model":        EMBED_MODEL,
        "clip_model":         "openai/clip-vit-base-patch32"
                              if clip_model else None,
        "m_paths":            M_PATHS,
        "k_demonstrations":   K_DEMONSTRATIONS,
        "lambda":             LAMBDA,
        "temperature":        TEMPERATURE,
        "visual_alpha":       VISUAL_ALPHA,
        "max_new_tokens":     MAX_NEW_TOKENS,
        "desc_max_tokens":    DESC_MAX_TOKENS,
        "max_pixels":         MAX_PIXELS,
        "numeric_tolerance":  NUMERIC_TOLERANCE,
        "offer_unanswerable": OFFER_UNANSWERABLE,
        "target_types":       sorted(TARGET_TYPES),
        "paper_faithful":     PAPER_FAITHFUL,
        "extensions":         active_extensions(),
        "n_results":          len(results),
        "finished_at":        time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = Path(RESULTS_DIR) / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✓  Run config saved → {meta_path}")

    label = ("PAPER-FAITHFUL" if PAPER_FAITHFUL
             else ", ".join(k for k, v in active_extensions().items() if v)
             or "none")
    print(f"   extensions active: {label}")

    print_results_table(results)
    return results


# ─────────────────────────────────────────────────────────────
# STEP 13 — RESULTS TABLE
# ─────────────────────────────────────────────────────────────

def print_results_table(results: list):
    """
    Print accuracy by question type, by method (zero-shot vs RoSE),
    and by answerability, plus the confusion counts.
    """
    valid = [r for r in results
             if "error" not in r and r.get("method") != "error"]
    if not valid:
        print("\n(no valid results to report)")
        return

    by_type   = defaultdict(lambda: {"correct": 0, "total": 0})
    by_method = defaultdict(lambda: {"correct": 0, "total": 0})
    by_ans    = defaultdict(lambda: {"correct": 0, "total": 0})

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
        print("\n" + "=" * 58)
        print(f"  {title}")
        print("=" * 58)
        for key, s in sorted(table.items()):
            acc = s["correct"] / s["total"] * 100 if s["total"] else 0
            print(f"  {key:<26}  {acc:5.1f}%  ({s['correct']}/{s['total']})")

    _block("ACCURACY BY QUESTION TYPE", by_type)
    total_correct = sum(s["correct"] for s in by_type.values())
    total_all     = sum(s["total"]   for s in by_type.values())
    overall = total_correct / total_all * 100 if total_all else 0
    print(f"  {'OVERALL':<26}  {overall:5.1f}%  ({total_correct}/{total_all})")

    _block("ACCURACY BY METHOD  (zero-shot vs RoSE)", by_method)
    _block("ACCURACY BY ANSWERABILITY", by_ans)

    print("\n" + "=" * 58)
    print("  ANSWERABILITY CONFUSION")
    print("=" * 58)
    n_ans  = by_ans["answerable"]["total"]
    n_un   = by_ans["unanswerable"]["total"]
    print(f"  answerable gold, predicted Unanswerable : "
          f"{answerable_said_unans}/{n_ans}"
          f"  ({answerable_said_unans / max(n_ans, 1) * 100:.1f}%)")
    print(f"  unanswerable gold, predicted a value    : "
          f"{unanswerable_said_val}/{n_un}"
          f"  ({unanswerable_said_val / max(n_un, 1) * 100:.1f}%)")
    print("=" * 58 + "\n")


if __name__ == "__main__":
    run()