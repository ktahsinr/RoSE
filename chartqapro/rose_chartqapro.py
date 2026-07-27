# =============================================================
# RoSE on ChartQAPro — Paper-Faithful Implementation
# Paper: "Making LLMs Better Reasoners with Orchestrated
#         Streaming Experiences" (EMNLP 2024)
# Target: ChartQAPro Factoid + MCQ questions only
# Model:  Qwen2.5-VL-3B-Instruct (vision-language, free T4 GPU)
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
# =============================================================

import json, os, math, time
from pathlib import Path
from collections import Counter, defaultdict

import torch
import numpy as np
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    BitsAndBytesConfig,
)
from sentence_transformers import SentenceTransformer


# ─────────────────────────────────────────────────────────────
# CONFIG  — adjust only these values if needed
# ─────────────────────────────────────────────────────────────
MODEL_NAME       = "Qwen/Qwen2.5-VL-3B-Instruct"
EMBED_MODEL      = "all-MiniLM-L6-v2"          # paper uses all-mpnet-base-v2

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
MAX_NEW_TOKENS   = 300
CHECKPOINT_EVERY = 25    # save progress every N samples

# Filter: only run on these question types
TARGET_TYPES     = {"factoid", "mcq"}   # lowercase


# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD MODELS
# ─────────────────────────────────────────────────────────────

def load_models():
    print("\n[1/3] Loading Qwen2.5-VL-3B (4-bit quantized for T4 GPU)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME, trust_remote_code=True
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
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

def load_image(image_field: str) -> Image.Image:
    """
    Tries several path patterns to find the chart image.
    ChartQAPro stores images under data/images/<source>/<filename>.
    """
    candidates = [
        Path(image_field),
        Path(IMAGES_DIR) / image_field,
        Path(IMAGES_DIR) / Path(image_field).name,
        Path("ChartQAPro") / image_field,
    ]
    for p in candidates:
        if p.exists():
            return Image.open(p).convert("RGB")
    raise FileNotFoundError(
        f"Image not found. Tried: {[str(c) for c in candidates]}"
    )


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

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=temperature,
            do_sample=(temperature > 0),
            top_p=0.9,
        )

    new_tokens = out_ids[0][inputs["input_ids"].shape[1]:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────────────────────
# STEP 4 — PROMPTS  (zero-shot CoT and few-shot CoT)
# ─────────────────────────────────────────────────────────────

def zero_shot_prompt(question: str, qtype: str,
                     choices: list = None) -> str:
    """
    Zero-Shot-CoT prompt — paper §3.3, used when pool is empty.
    For MCQ: include answer options.
    For Factoid: plain chain-of-thought.
    """
    if qtype == "mcq" and choices:
        opts = "\n".join(
            f"({chr(65+i)}) {c}" for i, c in enumerate(choices)
        )
        return (
            "Look at the chart carefully and answer the question.\n\n"
            f"Question: {question}\n\nOptions:\n{opts}\n\n"
            "Let's think step by step."
        )
    return (
        "Look at the chart carefully and answer the question.\n\n"
        f"Question: {question}\n\n"
        "Let's think step by step."
    )


def few_shot_prompt(question: str, qtype: str,
                    demonstrations: list,
                    choices: list = None) -> str:
    """
    Few-Shot-CoT prompt — paper Equation 9:
    LLM(q1, r1, a1, ..., qk, rk, ak, qt)

    Each demonstration is a triplet: (question, rationale, answer).
    The chart image is passed separately to the VLM.
    """
    header = (
        "You are an expert at reading and understanding charts. "
        "Here are some examples of chart questions with step-by-step reasoning:\n\n"
    )
    examples = ""
    for i, d in enumerate(demonstrations, 1):
        examples += (
            f"[Example {i}]\n"
            f"Q: {d['question']}\n"
            f"Reasoning: {d['rationale']}\n"
            f"A: {d['answer']}\n\n"
        )

    if qtype == "mcq" and choices:
        opts = "\n".join(
            f"({chr(65+i)}) {c}" for i, c in enumerate(choices)
        )
        task = (
            "Now answer the question about the chart above:\n\n"
            f"Q: {question}\n\nOptions:\n{opts}\n\n"
            "Let's think step by step."
        )
    else:
        task = (
            "Now answer the question about the chart above:\n\n"
            f"Q: {question}\n\n"
            "Let's think step by step."
        )

    return header + examples + task


# ─────────────────────────────────────────────────────────────
# STEP 5 — ANSWER EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_answer(raw: str, qtype: str) -> str:
    """
    Pull the final answer from a chain-of-thought output.
    For MCQ: tries to extract a letter (A/B/C/D/E).
    For Factoid: takes the last clean line.
    """
    text = raw.strip()

    # Look for explicit answer marker
    for marker in [
        "the answer is", "final answer:", "answer:", "therefore,",
        "thus,", "so the answer", "correct answer is",
    ]:
        idx = text.lower().rfind(marker)
        if idx != -1:
            tail = text[idx + len(marker):].strip()
            answer = tail.split("\n")[0].split(".")[0].strip()
            if answer:
                return answer.lower()

    # MCQ fallback: look for a letter on its own line
    if qtype == "mcq":
        for line in reversed(text.split("\n")):
            line = line.strip()
            if line in {"a", "b", "c", "d", "e",
                        "(a)", "(b)", "(c)", "(d)", "(e)"}:
                return line.strip("()")

    # General fallback: last non-empty line
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return lines[-1].lower() if lines else text.lower()


# ─────────────────────────────────────────────────────────────
# STEP 6 — COUNT STEPS  (paper Equation 4–5)
# ─────────────────────────────────────────────────────────────

def count_steps(text: str) -> int:
    """
    Count reasoning steps in a rationale.
    Paper: 'we see a line as one reasoning step' (§3.1).
    We count non-empty lines that contain reasoning indicators.
    """
    indicators = [
        "step", "first", "second", "third", "finally",
        "looking at", "from the chart", "the chart shows",
        "we can see", "because", "since", "therefore",
        "thus", "so", "note that", "observing", "=", "+", "-", "/",
    ]
    count = 0
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if any(ind in line.lower() for ind in indicators):
            count += 1
    return max(count, 1)  # at least 1 to avoid zero-complexity entries


# ─────────────────────────────────────────────────────────────
# STEP 7 — SELF-CONSISTENCY + UNCERTAINTY (paper Eq. 1–3)
# ─────────────────────────────────────────────────────────────

def run_m_paths(model, processor, image: Image.Image,
                prompt: str, qtype: str,
                m: int = M_PATHS) -> tuple:
    """
    Generate m reasoning paths. Return:
      majority_answer (str)
      all_raw_outputs (list of str)
      best_rationale  (str)  — path with most steps (Eq. 5)
      uncertainty     (float) — Shannon entropy (Eq. 1-3)
      complexity      (float) — avg steps of majority paths (Eq. 4)
    """
    raw_outputs = []
    answers     = []

    for run in range(m):
        raw = call_vlm(model, processor, image, prompt, TEMPERATURE)
        ans = extract_answer(raw, qtype)
        raw_outputs.append(raw)
        answers.append(ans)

    # Unique answers and their probabilities  (Eq. 1–2)
    counts  = Counter(answers)
    total   = len(answers)
    probs   = {a: c / total for a, c in counts.items()}

    # Shannon entropy (Eq. 3)
    entropy = -sum(p * math.log(p + 1e-12) for p in probs.values())
    # Normalize by log(m) so it's in [0, 1]
    uncertainty = entropy / (math.log(m) + 1e-12)

    # Majority answer
    majority_answer = counts.most_common(1)[0][0]

    # Paths that agree with majority (R* in Eq. 4)
    majority_paths = [
        raw for raw, ans in zip(raw_outputs, answers)
        if ans == majority_answer
    ]

    # Complexity = average steps of majority paths (Eq. 4)
    step_counts = [count_steps(r) for r in majority_paths]
    complexity  = sum(step_counts) / len(step_counts) if step_counts else 0.0

    # Best rationale = path with most steps (Eq. 5)
    best_rationale = max(majority_paths, key=count_steps)

    return majority_answer, raw_outputs, best_rationale, uncertainty, complexity


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

    def add(self, question: str, rationale: str, answer: str,
            qtype: str, uncertainty: float, complexity: float):
        emb = self.embedder.encode(question, convert_to_numpy=True)
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
        bucket_size = n // k
        buckets = []
        for b in range(k):
            start = b * bucket_size
            end   = start + bucket_size if b < k - 1 else n
            buckets.append(list(sim_sorted_indices[start:end]))

        # Remove truly empty buckets, then split the largest until len==k
        buckets = [b for b in buckets if b]
        while len(buckets) < k:
            # Find the largest bucket
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
        # Guarantee non-empty
        if not filtered:
            # Fall back to the one with minimum uncertainty
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
                    k: int = K_DEMONSTRATIONS) -> list:
        """
        Full RoSE orchestration for a new question:
        diversity → uncertainty filtering → complexity selection.
        Returns a list of up to k experience dicts.
        """
        if self.size() == 0:
            return []

        n = min(self.size(), 3 * k)   # candidate pool: top 3k

        # Similarity of all pool questions to test question
        q_emb = self.embedder.encode(question, convert_to_numpy=True)
        sims  = self._embeddings.dot(q_emb) / (
            np.linalg.norm(self._embeddings, axis=1) *
            np.linalg.norm(q_emb) + 1e-12
        )

        # Sort by similarity LOW → HIGH (diversity: spread from least to most similar)
        sorted_indices = np.argsort(sims)[:n].tolist()  # ascending

        # Algorithm 1: partition into k buckets
        buckets = self._partition(sorted_indices, k)

        demonstrations = []
        for bucket in buckets[:k]:
            # Step 1: filter high-uncertainty experiences
            filtered = self._filter_by_uncertainty(bucket)
            # Step 2: pick most complex (Eq. 8)
            selected = self._select_from_bucket(filtered)
            demonstrations.append(selected)

        return demonstrations[:k]


# ─────────────────────────────────────────────────────────────
# STEP 9 — CORRECTNESS CHECK
# ─────────────────────────────────────────────────────────────

def is_correct(prediction: str, ground_truth: str) -> bool:
    """
    Relaxed match: lowercase + strip.
    Also checks if ground truth is contained in prediction
    (handles cases where model says 'the answer is 42' and truth is '42').
    """
    pred  = prediction.lower().strip()
    truth = str(ground_truth).lower().strip()

    if pred == truth:
        return True
    if truth in pred:
        return True
    # For MCQ: check if matching letter appears
    if len(truth) == 1 and truth in pred.split():
        return True
    return False


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

    # Decide which phase we're in
    if pool.size() < K_DEMONSTRATIONS:
        # Phase A: zero-shot CoT
        prompt = zero_shot_prompt(question, qtype, choices)
        method = "zero_shot_cot"
        demos  = []
    else:
        # Phase B: orchestrate + few-shot CoT
        demos  = pool.orchestrate(question, k=K_DEMONSTRATIONS)
        prompt = few_shot_prompt(question, qtype, demos, choices)
        method = "rose_few_shot"

    # Generate m reasoning paths
    maj_answer, all_raws, best_rationale, uncertainty, complexity = \
        run_m_paths(model, processor, image, prompt, qtype, m=M_PATHS)

    # Add to pool (always — RoSE grows its pool on every question)
    pool.add(
        question    = question,
        rationale   = best_rationale,
        answer      = maj_answer,
        qtype       = qtype,
        uncertainty = uncertainty,
        complexity  = complexity,
    )

    correct = is_correct(maj_answer, truth)

    return {
        "id":           sample.get("id", ""),
        "question":     question,
        "question_type": qtype,
        "ground_truth": truth,
        "prediction":   maj_answer,
        "is_correct":   correct,
        "method":       method,
        "uncertainty":  round(uncertainty, 4),
        "complexity":   round(complexity, 4),
        "n_demos":      len(demos),
        "pool_size":    pool.size(),
    }


# ─────────────────────────────────────────────────────────────
# STEP 11 — LOAD + FILTER DATASET
# ─────────────────────────────────────────────────────────────

def load_factoid_mcq(path: str) -> list:
    """Load ChartQAPro and keep only Factoid + MCQ questions."""
    with open(path) as f:
        data = json.load(f)

    filtered = [
        d for d in data
        if d.get("question_type", "").lower().strip() in TARGET_TYPES
    ]

    print(f"\n[3/3] Dataset loaded")
    print(f"   Total questions : {len(data)}")
    print(f"   Factoid + MCQ   : {len(filtered)}")

    by_type = Counter(
        d["question_type"].lower() for d in filtered
    )
    for qt, n in by_type.items():
        print(f"   └─ {qt:<12} : {n}")

    return filtered


# ─────────────────────────────────────────────────────────────
# STEP 12 — MAIN RUN LOOP
# ─────────────────────────────────────────────────────────────

def run(model=None, processor=None, embedder=None):
    """
    Entry point. Call run() directly in Colab after loading models.
    Or call run() with no arguments to load models internally.
    """
    Path(RESULTS_DIR).mkdir(exist_ok=True)

    # Load models if not passed in
    if model is None:
        model, processor, embedder = load_models()

    # Load dataset
    data = load_factoid_mcq(DATASET_PATH)

    # Resume from checkpoint if it exists
    results   = []
    start_idx = 0
    if Path(CHECKPOINT_FILE).exists():
        with open(CHECKPOINT_FILE) as f:
            results = json.load(f)
        start_idx = len(results)
        print(f"\n↩  Resuming from checkpoint — sample {start_idx}/{len(data)}")

    # Initialise pool
    pool = ExperiencePool(embedder)

    # Pre-fill pool from already-processed results (warm restart)
    for r in results:
        pool.add(
            question    = r["question"],
            rationale   = r.get("best_rationale", r["prediction"]),
            answer      = r["prediction"],
            qtype       = r["question_type"],
            uncertainty = r.get("uncertainty", 0.5),
            complexity  = r.get("complexity", 1.0),
        )

    # ── Main loop ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RoSE on ChartQAPro  |  Factoid + MCQ only")
    print(f"  Paths per Q: {M_PATHS}   |   k demonstrations: {K_DEMONSTRATIONS}")
    print(f"  λ = {LAMBDA}   |   Questions: {len(data) - start_idx} remaining")
    print(f"{'='*60}")

    for i, sample in enumerate(data[start_idx:], start=start_idx):

        qtype  = sample.get("question_type", "?")
        method = "RoSE" if pool.size() >= K_DEMONSTRATIONS else "ZeroShot"
        print(f"\n[{i+1:04d}/{len(data)}]  type={qtype}  pool={pool.size()}  {method}")

        try:
            result = process_one(sample, model, processor, pool)
            results.append(result)

            mark = "✓" if result["is_correct"] else "✗"
            print(f"  {mark}  pred='{result['prediction'][:50]}'")
            print(f"     truth='{result['ground_truth']}'")
            print(f"     u={result['uncertainty']:.3f}  c={result['complexity']:.2f}")

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

        # Checkpoint
        if (i + 1) % CHECKPOINT_EVERY == 0:
            with open(CHECKPOINT_FILE, "w") as f:
                json.dump(results, f, indent=2)
            acc_so_far = (
                sum(r.get("is_correct", False) for r in results if "error" not in r)
                / max(sum(1 for r in results if "error" not in r), 1)
            ) * 100
            print(f"\n  💾 Checkpoint saved ({i+1} done, running acc={acc_so_far:.1f}%)")

    # Final save
    with open(FINAL_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓  Results saved → {FINAL_FILE}")

    # Print table
    print_results_table(results)
    return results


# ─────────────────────────────────────────────────────────────
# STEP 13 — RESULTS TABLE  (what you show your professor)
# ─────────────────────────────────────────────────────────────

def print_results_table(results: list):
    """
    Print accuracy by question type and by method.
    This is the comparison table for your paper/report.
    """
    valid = [r for r in results if "error" not in r]

    # By question type
    by_type   = defaultdict(lambda: {"correct": 0, "total": 0})
    by_method = defaultdict(lambda: {"correct": 0, "total": 0})

    for r in valid:
        qt     = r.get("question_type", "unknown")
        method = r.get("method", "unknown")
        ok     = r.get("is_correct", False)
        by_type[qt]["total"]     += 1
        by_method[method]["total"] += 1
        if ok:
            by_type[qt]["correct"]     += 1
            by_method[method]["correct"] += 1

    print("\n" + "="*55)
    print("  ACCURACY BY QUESTION TYPE")
    print("="*55)
    all_correct = all_total = 0
    for qt, s in sorted(by_type.items()):
        acc = s["correct"] / s["total"] * 100 if s["total"] else 0
        print(f"  {qt:<20}  {acc:5.1f}%  ({s['correct']}/{s['total']})")
        all_correct += s["correct"]
        all_total   += s["total"]
    overall = all_correct / all_total * 100 if all_total else 0
    print(f"  {'OVERALL':<20}  {overall:5.1f}%  ({all_correct}/{all_total})")

    print("\n" + "="*55)
    print("  ACCURACY BY METHOD  (zero-shot vs RoSE)")
    print("="*55)
    for method, s in sorted(by_method.items()):
        acc = s["correct"] / s["total"] * 100 if s["total"] else 0
        print(f"  {method:<25}  {acc:5.1f}%  ({s['correct']}/{s['total']})")
    print("="*55 + "\n")


if __name__ == "__main__":
    run()