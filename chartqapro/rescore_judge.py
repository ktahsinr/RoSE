# =============================================================
# Re-score ChartQAPro results + LLM-as-judge (Pydantic validated)
# =============================================================
# ZERO COST. No paid API, no API key, no internet required.
#
# Reads a finished results file (checkpoint_*.json / results_*.csv) and
# reports three metrics side by side:
#
#   1. raw          — the is_correct flag as written during the run
#   2. normalized   — recomputed with scoring.py (the OFFICIAL metric,
#                     the one to compare against the paper)
#   3. judged       — normalized, plus rows a local LLM judge rules
#                     semantically equivalent (a SECONDARY metric)
#
# The judge runs on a LOCAL model. Two free backends:
#
#   --judge qwen     reuse the Qwen2.5-VL already loaded for inference,
#                    in text-only mode. Zero extra VRAM, zero cost.
#   --judge ollama   a local Ollama text model (CPU). Zero cost.
#
# The judge never sees the chart. It only compares prediction against
# gold. If it could see the chart it would start re-answering the
# question and rescuing wrong predictions, which would turn the
# evaluator into a second model in the pipeline.
#
# Because a local model is NOT schema-constrained by an API, Pydantic is
# doing real work here: every verdict is parsed and validated, and an
# invalid one is retried with a corrective nudge rather than silently
# mis-read. A verdict that never validates is scored NOT correct.
#
# Usage:
#   python rescore_judge.py results/checkpoint_factoid.json
#   python rescore_judge.py rose.json --baseline zeroshot.json
#   python rescore_judge.py rose.json --judge ollama
#
# Scoring needs nothing but the standard library. The judge needs
# `pip install pydantic` (and, for --judge qwen, the GPU stack that
# rose_chartqapro.py already uses).
# =============================================================

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Literal

try:
    from .scoring import is_correct, normalize_answer, is_unanswerable, extract_answer
except ImportError:
    from scoring import is_correct, normalize_answer, is_unanswerable, extract_answer


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

# Ollama defaults. Note 127.0.0.1, NOT localhost — localhost resolves to
# ::1 first in some Kaggle/Colab images and the connection is refused.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = "qwen2.5:7b-instruct"

# Verdicts are cached by hash of (question, gold, prediction) so repeat
# runs cost nothing and the reported number is reproducible.
CACHE_FILE = "results/judge_cache.json"

# The judge only runs on rows normalization marks WRONG, so it can only
# ever raise the score. To measure the other direction, it also audits a
# random sample of rows marked CORRECT and reports the false-positive rate.
AUDIT_SAMPLE = 50
AUDIT_SEED = 20260807

# How many times to re-ask when the model returns unparseable JSON.
MAX_JUDGE_ATTEMPTS = 3

JUDGE_MAX_TOKENS = 256


# ─────────────────────────────────────────────────────────────
# PYDANTIC SCHEMA — the judge's output contract
# ─────────────────────────────────────────────────────────────
# This model is the single source of truth: it generates the schema shown
# to the judge in the prompt AND validates what comes back.

def build_schema():
    from pydantic import BaseModel, Field

    class JudgeVerdict(BaseModel):
        """A single semantic-equivalence verdict."""

        correct: bool = Field(
            description="true if the prediction means the same thing as the "
                        "gold answer, false otherwise"
        )
        category: Literal[
            "exact",               # identical after trivial cleanup
            "formatting",          # brackets, quotes, casing, whitespace
            "units",               # units, currency, %, thousands separators
            "rounding",            # numerically equivalent within rounding
            "semantic_equivalent", # different words, same meaning
            "unanswerable_match",  # both say the chart cannot answer it
            "wrong",               # genuinely different answer
            "unparseable",         # prediction is truncated or not an answer
        ] = Field(description="why the prediction matches, or how it fails")
        reason: str = Field(
            description="one short sentence justifying the verdict"
        )

    return JudgeVerdict


_RULES = """You grade answers to chart questions.

You are given a QUESTION, the GOLD answer, and a PREDICTION. Decide one \
thing only: does the PREDICTION mean the same thing as the GOLD answer?

Rules:
- Judge equivalence to the GOLD answer. Do NOT assess whether the GOLD \
answer is itself correct, and do not try to answer the question yourself.
- You cannot see the chart. Never guess what the chart shows.
- Ignore differences in formatting, casing, units, currency symbols, \
percent signs, thousands separators, and bracket or quote wrapping. \
"[107,995]" and "107995" are equivalent; "[9]" and "9%" are equivalent.
- Numbers equal after reasonable rounding are equivalent (85.0 vs 85, \
12.47 vs 12.5). Numbers that differ in value are NOT.
- Dates and months in different forms are equivalent ("Dec" vs "December").
- "Unanswerable", "Cannot be determined" and similar all mean the chart \
does not support an answer. Such a prediction is correct ONLY when the \
gold answer also says the question is unanswerable — never when the gold \
answer is a real value.
- If the prediction is truncated, empty, or is reasoning rather than an \
answer, it is not correct: use category "unparseable".
- When genuinely uncertain, mark it not correct."""

_OUTPUT_CONTRACT = """
Reply with a single JSON object and nothing else. No markdown fences, no \
commentary before or after.

The object must match this JSON schema exactly:
{schema}

Example of a valid reply:
{{"correct": true, "category": "units", "reason": "9 and 9% are the same value."}}"""


def judge_system_prompt(JudgeVerdict) -> str:
    """Rules + the output contract, generated from the Pydantic model."""
    schema = json.dumps(JudgeVerdict.model_json_schema(), separators=(",", ":"))
    return _RULES + "\n" + _OUTPUT_CONTRACT.format(schema=schema)


def judge_user_prompt(row: dict) -> str:
    return (
        f"QUESTION: {row['question']}\n"
        f"GOLD ANSWER: {row['ground_truth']}\n"
        f"PREDICTION: {row['prediction']}\n\n"
        "Does the prediction mean the same thing as the gold answer?"
    )


# ─────────────────────────────────────────────────────────────
# LOADING RESULTS
# ─────────────────────────────────────────────────────────────

def load_results(path: str) -> list:
    """Load a results file — JSON list, JSON dict of rows, or CSV."""
    p = Path(path)
    if not p.exists():
        sys.exit(f"error: no such file: {path}")

    if p.suffix.lower() == ".csv":
        with p.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    else:
        with p.open(encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            rows = data.get("results") or list(data.values())
        else:
            rows = data

    return [r for r in rows if isinstance(r, dict)]


_FIELD_ALIASES = {
    "question":      ("question", "Question", "query"),
    "ground_truth":  ("ground_truth", "answer", "Answer", "gold", "truth",
                      "gt"),
    "prediction":    ("prediction", "pred", "Prediction", "model_answer"),
    "question_type": ("question_type", "qtype", "Question Type", "type"),
    "is_correct":    ("is_correct", "correct", "Correct"),
    "method":        ("method", "Method"),
    "raw_output":    ("best_rationale", "raw_output", "raw"),
    "choices":       ("options_shown", "choices", "options", "Options"),
}


def _get(row: dict, field: str, default=""):
    for key in _FIELD_ALIASES[field]:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return default


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def normalize_rows(rows: list, reextract: bool = False) -> list:
    """Flatten heterogeneous result files into one shape."""
    out = []
    for r in rows:
        qtype = str(_get(r, "question_type", "factoid")).lower().strip()
        if qtype not in {"factoid", "mcq"}:
            qtype = "factoid"

        pred = _get(r, "prediction")
        raw = _get(r, "raw_output")
        # Optionally re-derive the answer span from the logged generation.
        # Useful for old runs whose extraction truncated decimals.
        if reextract and raw:
            respan = extract_answer(str(raw), qtype)
            if respan:
                pred = respan

        choices = _get(r, "choices", None)
        if isinstance(choices, str):
            # CSV round-trips lists as strings; recover a JSON-ish list.
            try:
                choices = json.loads(choices.replace("'", '"'))
            except (json.JSONDecodeError, AttributeError):
                choices = None
        if not isinstance(choices, (list, tuple)):
            choices = None

        out.append({
            "question":      str(_get(r, "question")),
            "choices":       list(choices) if choices else None,
            "ground_truth":  _get(r, "ground_truth"),
            "prediction":    pred,
            "question_type": qtype,
            "raw_flag":      _truthy(_get(r, "is_correct", False)),
            "method":        str(_get(r, "method", "unknown")),
            "is_error":      "error" in r or str(pred).strip() == "ERROR",
        })
    return out


def score(rows: list) -> list:
    """Attach the normalized verdict to every row."""
    for r in rows:
        r["norm_flag"] = (
            False if r["is_error"]
            else is_correct(r["prediction"], r["ground_truth"],
                            r["question_type"], choices=r.get("choices"))
        )
    return rows


# ─────────────────────────────────────────────────────────────
# FREE JUDGE BACKENDS
# ─────────────────────────────────────────────────────────────

def _ollama_generate(system: str, prompt: str, model: str,
                     host: str = OLLAMA_HOST) -> str:
    """
    Call a local Ollama model. Uses urllib so there is no extra dependency.

    Requires `ollama serve` to be running and the model pulled:
        ollama pull qwen2.5:7b-instruct
    """
    import urllib.request
    import urllib.error

    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "format": "json",          # ask Ollama for JSON-only decoding
        "options": {
            "temperature": 0,       # deterministic verdicts
            "num_predict": JUDGE_MAX_TOKENS,
        },
    }
    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body.get("response", "")


def _qwen_generate(system: str, prompt: str, qwen) -> str:
    """
    Reuse the Qwen2.5-VL already loaded for inference, in TEXT-ONLY mode.

    Zero extra VRAM and zero cost, because the weights are already
    resident. `qwen` is the (model, processor) tuple from load_models().

    Independence caveat for the writeup: this is the same model that
    produced the predictions. That is much weaker than an independent
    judge for open-ended quality judgment — but this task is string
    equivalence with the gold answer in hand, not "is this answer good",
    so self-judging bias is far less of a concern. State it explicitly
    rather than glossing over it.
    """
    import torch

    model, processor = qwen
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user",   "content": [{"type": "text", "text": prompt}]},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=JUDGE_MAX_TOKENS,
            do_sample=False,          # greedy — deterministic verdicts
        )
    new_tokens = out_ids[0][inputs["input_ids"].shape[1]:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────────────────────
# JSON RECOVERY + PYDANTIC VALIDATION
# ─────────────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _extract_json_object(text: str):
    """
    Pull the first balanced {...} out of a model reply.

    Local models are not schema-constrained, so they wrap JSON in
    markdown fences or add a sentence of preamble. This recovers the
    object; Pydantic then decides whether it is actually valid.
    """
    if not text:
        return None
    cleaned = _FENCE_RE.sub("", text.strip())

    start = cleaned.find("{")
    if start == -1:
        return None
    depth, in_string, escaped = 0, False, False
    for i, ch in enumerate(cleaned[start:], start):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(cleaned[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _one_verdict(row: dict, JudgeVerdict, generate, system: str) -> dict:
    """
    Ask the judge about one row, retrying while the reply fails validation.

    Returns a plain dict. A verdict that never validates is recorded as
    NOT correct — a judge failure must never be scored as a pass.
    """
    from pydantic import ValidationError

    base_prompt = judge_user_prompt(row)
    prompt = base_prompt
    last_error = "no reply"

    for attempt in range(1, MAX_JUDGE_ATTEMPTS + 1):
        try:
            reply = generate(system, prompt)
        except Exception as exc:                       # noqa: BLE001
            last_error = f"backend error: {exc}"
            break

        payload = _extract_json_object(reply)
        if payload is None:
            last_error = "no JSON object in reply"
        else:
            try:
                verdict = JudgeVerdict(**payload)
                out = verdict.model_dump()
                if attempt > 1:
                    out["judge_attempts"] = attempt
                return out
            except ValidationError as exc:
                last_error = f"schema validation failed: " \
                             f"{exc.errors()[0].get('msg', 'invalid')}"

        # Corrective nudge for the retry.
        prompt = (
            f"{base_prompt}\n\n"
            f"Your previous reply was rejected: {last_error}. "
            "Reply with ONLY a single JSON object matching the schema. "
            "No fences, no commentary."
        )

    return {"correct": False, "category": "unparseable",
            "reason": f"judge produced no valid verdict ({last_error})",
            "judge_error": last_error,
            "judge_attempts": MAX_JUDGE_ATTEMPTS}


# ─────────────────────────────────────────────────────────────
# CACHING + BATCH JUDGING
# ─────────────────────────────────────────────────────────────

def _cache_key(row: dict) -> str:
    payload = "\x1f".join([
        str(row["question"]), str(row["ground_truth"]), str(row["prediction"]),
        row["question_type"],
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _load_cache() -> dict:
    p = Path(CACHE_FILE)
    if p.exists():
        try:
            with p.open(encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(cache: dict):
    p = Path(CACHE_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def judge_rows(rows: list, backend: str = "ollama", qwen=None,
               ollama_model: str = OLLAMA_MODEL,
               ollama_host: str = OLLAMA_HOST) -> dict:
    """
    Judge every row. Returns {cache_key: verdict_dict}.

    backend:
      "qwen"   — reuse the loaded Qwen2.5-VL; pass qwen=(model, processor)
      "ollama" — local Ollama server
    """
    if not rows:
        return {}

    try:
        JudgeVerdict = build_schema()
    except ImportError:
        sys.exit("error: pip install pydantic   (or run without --judge)")

    system = judge_system_prompt(JudgeVerdict)

    if backend == "qwen":
        if qwen is None:
            # Only reachable from the CLI; in a notebook you pass the
            # already-loaded pair and skip this entirely.
            try:
                from .rose_chartqapro import load_models
            except ImportError:
                from rose_chartqapro import load_models
            print("  loading Qwen for judging (notebook users: pass "
                  "qwen=(model, processor) to reuse the loaded one)")
            model, processor, _ = load_models()
            qwen = (model, processor)
        generate = lambda s, p: _qwen_generate(s, p, qwen)   # noqa: E731
    elif backend == "ollama":
        generate = lambda s, p: _ollama_generate(              # noqa: E731
            s, p, ollama_model, ollama_host)
    else:
        sys.exit(f"error: unknown judge backend {backend!r} "
                 "(choose 'qwen' or 'ollama')")

    cache = _load_cache()
    pending = [r for r in rows if _cache_key(r) not in cache]
    print(f"\n  judge[{backend}]: {len(rows)} rows, "
          f"{len(rows) - len(pending)} cached, {len(pending)} to run")

    invalid = 0
    failed = {}
    for i, row in enumerate(pending, 1):
        verdict = _one_verdict(row, JudgeVerdict, generate, system)
        key = _cache_key(row)
        if "judge_error" in verdict:
            # Do NOT cache a failure. Otherwise one run with the server
            # down or the model unpulled would poison the cache, and every
            # later run would silently reuse those non-verdicts instead of
            # retrying them.
            invalid += 1
            failed[key] = verdict
        else:
            cache[key] = verdict

        if i % 10 == 0 or i == len(pending):
            print(f"    {i}/{len(pending)}")
            _save_cache(cache)

    _save_cache(cache)
    if invalid:
        # Report this — it is a validity statistic about the judge itself,
        # and it belongs in the writeup alongside the judged metric.
        print(f"    ⚠  {invalid}/{len(pending)} rows produced no valid "
              f"verdict after {MAX_JUDGE_ATTEMPTS} attempts "
              f"(scored NOT correct, NOT cached — rerun to retry)")
        if invalid == len(pending) and pending:
            print("    ⚠  EVERY judge call failed. Is the backend up? "
                  "For ollama: `ollama serve` + "
                  f"`ollama pull {ollama_model}`, host {ollama_host}")

    out = {_cache_key(r): cache[_cache_key(r)] for r in rows
           if _cache_key(r) in cache}
    out.update(failed)     # visible to the report, absent from the cache
    return out


# ─────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────

def _pct(n, d):
    return f"{n / d * 100:5.1f}%  ({n}/{d})" if d else "  n/a"


def report(label: str, rows: list, verdicts: dict,
           audit: dict = None) -> dict:
    valid = [r for r in rows if not r["is_error"]]
    n = len(valid)
    if not n:
        print(f"\n{label}: no valid rows")
        return {}

    raw = sum(r["raw_flag"] for r in valid)
    norm = sum(r["norm_flag"] for r in valid)
    judged = sum(
        r["norm_flag"] or verdicts.get(_cache_key(r), {}).get("correct", False)
        for r in valid
    )

    print(f"\n{'=' * 62}")
    print(f"  {label}   (n={n}, errors excluded: {len(rows) - n})")
    print(f"{'=' * 62}")
    print(f"  raw flag from run          : {_pct(raw, n)}")
    print(f"  normalized  [OFFICIAL]     : {_pct(norm, n)}")
    if verdicts:
        print(f"  + judge     [SECONDARY]    : {_pct(judged, n)}")

    # Split answerable / unanswerable — a big slice of factoid gold is
    # "Unanswerable", and the two behave very differently.
    for bucket, pred in (("answerable", False), ("unanswerable", True)):
        sub = [r for r in valid if is_unanswerable(r["ground_truth"]) == pred]
        if sub:
            print(f"    └─ {bucket:<14} norm : "
                  f"{_pct(sum(r['norm_flag'] for r in sub), len(sub))}")

    # Where the recovered points came from — this is the error taxonomy,
    # generated from the judge's constrained category enum.
    recovered = [verdicts[_cache_key(r)] for r in valid
                 if not r["norm_flag"] and _cache_key(r) in verdicts
                 and verdicts[_cache_key(r)].get("correct")]
    if recovered:
        print("\n  recovered by judge, by category:")
        for cat, count in Counter(v["category"] for v in recovered).most_common():
            print(f"    {cat:<22} {count}")

    still_wrong = [verdicts[_cache_key(r)] for r in valid
                   if not r["norm_flag"] and _cache_key(r) in verdicts
                   and not verdicts[_cache_key(r)].get("correct")]
    if still_wrong:
        print("\n  still wrong after judge, by category:")
        for cat, count in Counter(v["category"] for v in still_wrong).most_common():
            print(f"    {cat:<22} {count}")

    # The one-sided-gate check. Judging only the wrong rows can only ever
    # raise the score, so a sample of rows marked CORRECT is audited too.
    if audit:
        n_audit = len(audit)
        # A failed judge call is not a disagreement. Counting it as one
        # would report a large fake over-count whenever the backend is down.
        errors = sum(1 for v in audit.values() if "judge_error" in v)
        usable = n_audit - errors
        fp = sum(1 for v in audit.values()
                 if "judge_error" not in v and not v.get("correct"))
        print(f"\n  false-positive audit ({n_audit} rows marked correct):")
        if errors:
            print(f"    judge failed on          : {_pct(errors, n_audit)}"
                  f"  — excluded from the rate below")
        if usable:
            print(f"    judge disagrees on       : {_pct(fp, usable)}")
            if fp:
                est = norm * fp / usable
                print(f"    implied over-count       : ~{est:.1f} rows "
                      f"(~{est / n * 100:.1f}pp)")
        else:
            print("    no usable verdicts — cannot estimate false positives")

    return {"n": n, "raw": raw, "normalized": norm,
            "judged": judged if verdicts else None}


def main():
    ap = argparse.ArgumentParser(
        description="Re-score ChartQAPro results, and optionally run a free "
                    "local LLM judge on the rows normalization marks wrong.")
    ap.add_argument("results", help="RoSE results JSON or CSV")
    ap.add_argument("--baseline", help="zero-shot baseline results, scored "
                                      "identically so the delta is fair")
    ap.add_argument("--judge", choices=["none", "qwen", "ollama"],
                    default="none",
                    help="judge backend. Default 'none' = scoring only, no "
                         "model calls at all. Both backends are free/local.")
    ap.add_argument("--ollama-model", default=OLLAMA_MODEL,
                    help=f"Ollama model tag (default {OLLAMA_MODEL})")
    ap.add_argument("--ollama-host", default=OLLAMA_HOST,
                    help="use 127.0.0.1, not localhost")
    ap.add_argument("--reextract", action="store_true",
                    help="re-derive answers from logged raw output "
                         "(recovers runs whose extraction truncated decimals)")
    ap.add_argument("--audit-sample", type=int, default=AUDIT_SAMPLE,
                    help="rows marked correct to audit for false positives")
    ap.add_argument("--out", help="write per-row verdicts to this JSON file")
    args = ap.parse_args()

    datasets = [("RoSE", args.results)]
    if args.baseline:
        datasets.append(("Zero-shot baseline", args.baseline))

    use_judge = args.judge != "none"
    qwen = None   # loaded once, reused across both datasets

    summary = {}
    for label, path in datasets:
        rows = score(normalize_rows(load_results(path), args.reextract))

        verdicts, audit = {}, {}
        if use_judge:
            def _judge(subset):
                return judge_rows(subset, backend=args.judge, qwen=qwen,
                                  ollama_model=args.ollama_model,
                                  ollama_host=args.ollama_host)

            wrong = [r for r in rows if not r["is_error"] and not r["norm_flag"]]
            verdicts = _judge(wrong)

            # Symmetric check: audit a random sample of the CORRECT rows.
            right = [r for r in rows if not r["is_error"] and r["norm_flag"]]
            if right and args.audit_sample > 0:
                rng = random.Random(AUDIT_SEED)
                sample = rng.sample(right, min(args.audit_sample, len(right)))
                audit = _judge(sample)

        summary[label] = report(label, rows, verdicts, audit)

        if args.out:
            out_path = Path(args.out)
            if len(datasets) > 1:
                out_path = out_path.with_name(
                    f"{out_path.stem}_{label.split()[0].lower()}{out_path.suffix}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump([
                    {**r, "verdict": verdicts.get(_cache_key(r))}
                    for r in rows
                ], f, indent=2)
            print(f"\n  per-row verdicts → {out_path}")

    # The delta is the claim, and it is only meaningful when both sides
    # went through the identical scorer and the identical judge.
    if len(summary) == 2 and all(summary.values()):
        rose = summary["RoSE"]
        base = summary["Zero-shot baseline"]
        print(f"\n{'=' * 62}")
        print("  ROSE vs ZERO-SHOT  (same scorer, same judge)")
        print(f"{'=' * 62}")
        for metric in ("normalized", "judged"):
            if rose.get(metric) is None or base.get(metric) is None:
                continue
            r_acc = rose[metric] / rose["n"] * 100
            b_acc = base[metric] / base["n"] * 100
            print(f"  {metric:<12} RoSE {r_acc:5.1f}%  vs  "
                  f"zero-shot {b_acc:5.1f}%   "
                  f"delta {r_acc - b_acc:+.1f}pp")
        print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
