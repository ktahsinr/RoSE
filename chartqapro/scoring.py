# =============================================================
# Answer extraction + normalization + correctness for ChartQAPro
# =============================================================
# Pure Python — NO torch / transformers / sentence-transformers.
# This means results can be re-scored on a laptop with no GPU,
# decoupled from inference. Imported by both:
#   rose_chartqapro.py   (scores during the run)
#   rescore_judge.py     (re-scores a finished results file)
# =============================================================

import re
from collections import Counter

# Relative tolerance for numeric answers (5%).
NUMERIC_TOLERANCE = 0.05

# ─────────────────────────────────────────────────────────────
# Unanswerable handling
# A large slice of ChartQAPro gold answers are "Unanswerable".
# These must only ever match each other — never a numeric value.
# ─────────────────────────────────────────────────────────────

UNANSWERABLE_FORMS = {
    "unanswerable", "not answerable", "unanswered", "no answer",
    "n/a", "na", "none", "unknown", "not applicable",
    "cannot be answered", "cannot be determined", "can't be determined",
    "not enough information", "insufficient information",
    "cannot be inferred", "not stated", "not shown",
}

_UNANSWERABLE_SUBSTRINGS = (
    "unanswerable", "not answerable", "cannot be answered",
    "cannot be determined", "can't be determined", "cannot be inferred",
    "not enough information", "insufficient information",
    "no answer", "not applicable", "not stated in the chart",
)


def is_unanswerable(text: str) -> bool:
    """True if this answer means 'the chart does not support an answer'."""
    if text is None:
        return False
    t = str(text).strip().lower().strip(".!?  ")
    if not t:
        return False
    if t in UNANSWERABLE_FORMS:
        return True
    # Only substring-match short strings; a long rationale mentioning
    # "no answer" in passing should not be classed as unanswerable.
    if len(t) <= 60:
        return any(p in t for p in _UNANSWERABLE_SUBSTRINGS)
    return False


# ─────────────────────────────────────────────────────────────
# Number parsing
# ─────────────────────────────────────────────────────────────

MULTIPLIERS = {
    "k": 1e3, "thousand": 1e3, "thousands": 1e3,
    "m": 1e6, "mn": 1e6, "million": 1e6, "millions": 1e6,
    "b": 1e9, "bn": 1e9, "billion": 1e9, "billions": 1e9,
    "t": 1e12, "tn": 1e12, "trillion": 1e12, "trillions": 1e12,
}

# Words we ignore when deciding "is this string just a number?"
_UNIT_WORDS = {
    "percent", "percentage", "pct", "pp",
    "usd", "eur", "gbp", "inr", "jpy",
    "dollar", "dollars", "euro", "euros", "pound", "pounds", "rupees",
    "point", "points", "people", "persons", "person",
    "year", "years", "day", "days", "month", "months", "hour", "hours",
    "time", "times", "x", "unit", "units", "of",
} | set(MULTIPLIERS)

_CURRENCY = "$€£₹¥"
_NUM_FULL = re.compile(r"[-+]?\d+(?:\.\d+)?")

MONTHS = {
    "jan": "january", "feb": "february", "mar": "march", "apr": "april",
    "jun": "june", "jul": "july", "aug": "august", "sep": "september",
    "sept": "september", "oct": "october", "nov": "november", "dec": "december",
    "may": "may",
}

_WRAP_PAIRS = [("[", "]"), ("(", ")"), ("{", "}"), ('"', '"'), ("'", "'"),
               ("`", "`"), ("«", "»")]


def _strip_wrappers(s: str) -> str:
    """Remove surrounding brackets/quotes, repeatedly.  '[85]' -> '85'."""
    s = s.strip()
    changed = True
    while changed and len(s) >= 2:
        changed = False
        for left, right in _WRAP_PAIRS:
            if s.startswith(left) and s.endswith(right):
                s = s[1:-1].strip()
                changed = True
    return s


def _strip_edge_punct(s: str) -> str:
    """Trim punctuation at the ends only — never touches '12.5' internally."""
    return s.strip().strip(".,;:!?*  ").strip()


def number_variants(text) -> list:
    """
    Parse a short answer into candidate numeric values.

    Returns [] when the string is not purely a number (+ units), so that
    'December 2019' is never compared numerically against 2019.

    A magnitude word yields TWO candidates, because ChartQAPro gold
    sometimes carries the magnitude and sometimes doesn't:
        '3.4 billion' -> [3.4e9, 3.4]
    """
    if text is None:
        return []
    s = str(text).strip().lower()

    # Accounting negatives: (1,234) means -1234. Detect this BEFORE
    # stripping wrappers, which would otherwise remove the parentheses
    # that carry the sign.
    negative = s.startswith("(") and s.endswith(")")

    s = _strip_wrappers(s)
    if not s:
        return []

    for ch in _CURRENCY:
        s = s.replace(ch, " ")
    s = s.replace(",", "")
    s = s.replace("%", " percent ")
    s = s.replace("(", " ").replace(")", " ")

    numbers, multiplier, parseable = [], 1.0, True
    for token in s.split():
        token = _strip_edge_punct(token)
        if not token:
            continue
        if _NUM_FULL.fullmatch(token):
            numbers.append(float(token))
        elif token in MULTIPLIERS:
            multiplier *= MULTIPLIERS[token]
        elif token in _UNIT_WORDS:
            continue
        else:
            parseable = False
            break

    if not parseable or len(numbers) != 1:
        return []

    base = numbers[0]
    if negative and base > 0:
        base = -base

    if multiplier != 1.0:
        return [base * multiplier, base]
    return [base]


def _format_number(value: float) -> str:
    """Canonical string for a number: 85.0 -> '85', 12.50 -> '12.5'."""
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(round(value, 6)).rstrip("0").rstrip(".")


def numbers_match(pred_values: list, gold_values: list,
                  tolerance: float = NUMERIC_TOLERANCE) -> bool:
    """True if any prediction candidate is within `tolerance` of any gold."""
    for p in pred_values:
        for g in gold_values:
            if p == g:
                return True
            denom = max(abs(g), 1e-9)
            if abs(p - g) / denom <= tolerance:
                return True
    return False


# ─────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────

def normalize_answer(text, qtype: str = "factoid") -> str:
    """
    Canonical comparable form of an answer.

    Handles: case, whitespace, bracket wrapping, thousands separators,
    currency symbols, percent signs, magnitude words, month abbreviations,
    MCQ letter forms, and every spelling of 'Unanswerable'.

        '[107,995]'      -> '107995'
        '[9]' / '9%'     -> '9'
        '(B)' / 'b.'     -> 'b'      (qtype='mcq')
        'Dec'            -> 'december'
        'Cannot be determined' -> 'unanswerable'
    """
    if text is None:
        return ""
    s = str(text).strip().lower()
    s = _strip_wrappers(s)
    s = _strip_edge_punct(s)
    s = re.sub(r"\s+", " ", s)
    if not s:
        return ""

    if is_unanswerable(s):
        return "unanswerable"

    if qtype == "mcq":
        m = re.fullmatch(r"\(?([a-h])\)?", s)
        if m:
            return m.group(1)
        m = re.search(r"\boption\s*\(?([a-h])\)?\b", s)
        if m:
            return m.group(1)
        # not a bare letter — fall through and compare as text

    values = number_variants(s)
    if values:
        return _format_number(values[0])

    # Text answer: expand month abbreviations, drop stray punctuation.
    words = [MONTHS.get(w, w) for w in s.split()]
    s = " ".join(words)
    s = re.sub(r"[^a-z0-9 %./\-]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def split_gold(ground_truth) -> list:
    """
    ChartQAPro sometimes lists several acceptable answers.
    Returns every acceptable gold string.
    """
    if ground_truth is None:
        return []
    if isinstance(ground_truth, (list, tuple, set)):
        return [str(g) for g in ground_truth if str(g).strip()]
    s = str(ground_truth)
    parts = re.split(r"\||\n", s)
    return [p.strip() for p in parts if p.strip()] or [s.strip()]


# ─────────────────────────────────────────────────────────────
# Vote aggregation  (paper Eq. 1–3, plus an optional extension)
# ─────────────────────────────────────────────────────────────
# Lives here rather than in the pipeline so ablate.py can recompute
# aggregation offline from logged per-path answers, with no GPU.

def cluster_numeric(values, tolerance: float = NUMERIC_TOLERANCE) -> list:
    """
    Greedily group numeric answers that agree within relative tolerance.
    Returns a list of lists, largest group first.
    """
    clusters = []
    for v in values:
        for c in clusters:
            if abs(v - c[0]) / max(abs(c[0]), 1e-9) <= tolerance:
                c.append(v)
                break
        else:
            clusters.append([v])
    return sorted(clusters, key=len, reverse=True)


def _entropy(group_sizes: list, total: int) -> float:
    """Shannon entropy over group sizes, normalized to [0, 1] (Eq. 3)."""
    import math
    if total <= 1:
        return 0.0
    ent = -sum((s / total) * math.log(s / total + 1e-12) for s in group_sizes)
    return ent / (math.log(total) + 1e-12)


def aggregate_answers(canonical: list, tolerance: float = NUMERIC_TOLERANCE,
                      use_clustering: bool = False) -> dict:
    """
    Aggregate m per-path canonical answers into one verdict.

    PAPER-FAITHFUL MODE (use_clustering=False) — exact-string majority
    vote and entropy over distinct answers, as in Eq. 1-3.

    EXTENSION (use_clustering=True) — when every path produced a number,
    group them within `tolerance` first, take the MEDIAN of the largest
    group, and compute entropy over groups.

    Why the extension: for continuous chart values at m=3, byte-identical
    agreement is rare, so exact-string mode usually finds no majority and
    Counter.most_common silently returns whichever answer came first —
    i.e. path 0, arbitrarily. If path 0 is the outlier you take the
    outlier and discard two agreeing paths. It also reports entropy 1.0
    (maximum disagreement) for answers that agree to within 1%, which is
    the signal Eq. 6-7 filters the experience pool on.

    Returns:
      winner      str   — the winning canonical answer
      members     list  — indices of the paths backing the winner (R* in Eq. 4)
      uncertainty float — normalized entropy
      agreement   float — share of paths backing the winner
      n_groups    int   — distinct answers (or clusters)
      mode        str   — "cluster" or "exact"
    """
    total = len(canonical)
    if total == 0:
        return {"winner": "", "members": [], "uncertainty": 0.0,
                "agreement": 0.0, "n_groups": 0, "mode": "exact"}

    if use_clustering:
        values = [number_variants(a) for a in canonical]
        # Only cluster when EVERY path gave a number. A mix of numeric and
        # textual answers is not safely comparable on a number line.
        if all(v for v in values) and total > 1:
            primary = [v[0] for v in values]
            clusters = cluster_numeric(primary, tolerance)
            best = sorted(clusters[0])
            median = best[len(best) // 2]

            # Paths whose value falls in the winning cluster.
            members = [i for i, v in enumerate(primary)
                       if any(v == b for b in clusters[0])]
            return {
                "winner":      _format_number(median),
                "members":     members,
                "uncertainty": _entropy([len(c) for c in clusters], total),
                "agreement":   len(clusters[0]) / total,
                "n_groups":    len(clusters),
                "mode":        "cluster",
            }

    counts = Counter(canonical)
    winner = counts.most_common(1)[0][0]
    return {
        "winner":      winner,
        "members":     [i for i, a in enumerate(canonical) if a == winner],
        "uncertainty": _entropy(list(counts.values()), total),
        "agreement":   counts[winner] / total,
        "n_groups":    len(counts),
        "mode":        "exact",
    }


def has_final_answer(raw: str) -> bool:
    """
    True if a generation actually emitted the requested 'Final Answer:'
    line. A path that did not is being read by the last-line fallback,
    which is noise — and it currently votes with full weight.
    """
    return bool(raw) and bool(_FINAL_RE.search(str(raw)))


# ─────────────────────────────────────────────────────────────
# MCQ option resolution
# ─────────────────────────────────────────────────────────────

def resolve_choice(answer, choices):
    """
    Resolve an answer to a 0-based index into `choices`, or None.

    Accepts a letter ('B', '(b)', 'option b'), the option's text
    ('45%'), or a numerically equal form of the option's text ('45').
    This recovers real answers the model got right but wrote in the
    wrong form — a formatting loss, not a knowledge loss.

    Uses EXACT numeric equality, not the 5% tolerance: options are
    discrete and two of them may legitimately be 3% apart, so a
    tolerant match could silently pick the wrong option.
    """
    if not choices:
        return None
    n = len(choices)
    s = str(answer).strip()
    if not s:
        return None

    # 1. letter form
    letter = normalize_answer(s, "mcq")
    if len(letter) == 1 and "a" <= letter <= chr(ord("a") + n - 1):
        return ord(letter) - ord("a")

    # 2. exact normalized text match against the option list
    pred_norm = normalize_answer(s, "factoid")
    if pred_norm:
        targets = [normalize_answer(c, "factoid") for c in choices]
        hits = [i for i, t in enumerate(targets) if t and t == pred_norm]
        if len(hits) == 1:
            return hits[0]

        # 3. exact numeric match against the option list
        pred_values = number_variants(s)
        if pred_values:
            hits = [
                i for i, c in enumerate(choices)
                if number_variants(c)
                and numbers_match(pred_values, number_variants(c), 0.0)
            ]
            if len(hits) == 1:
                return hits[0]

    # 4. an Unanswerable-style answer resolves to an Unanswerable option
    if is_unanswerable(s):
        hits = [i for i, c in enumerate(choices) if is_unanswerable(c)]
        if len(hits) == 1:
            return hits[0]

    return None


# ─────────────────────────────────────────────────────────────
# Correctness
# ─────────────────────────────────────────────────────────────

def is_correct(prediction, ground_truth, qtype: str = "factoid",
               tolerance: float = NUMERIC_TOLERANCE, choices=None) -> bool:
    """
    Normalized exact match, plus numeric tolerance.

    Deliberately does NOT do substring matching. The old
    `if truth in pred` rule scored gold '9' correct against a
    prediction of '1997', inflating the reported accuracy — and on MCQ it
    was worse: 'Unanswerable' contains the letters a, b and e, so a
    punted answer scored correct on 3 of 5 possible gold letters.

    When `choices` is supplied for an MCQ, both sides are resolved to an
    option index first. That makes gold-as-letter and gold-as-text
    equivalent, and lets a prediction written as the option's text count.
    """
    if qtype == "mcq" and choices:
        pred_idx = resolve_choice(prediction, choices)
        if pred_idx is not None:
            for gold in split_gold(ground_truth):
                gold_idx = resolve_choice(gold, choices)
                if gold_idx is not None:
                    if pred_idx == gold_idx:
                        return True
                    # Both resolved to real options and they differ:
                    # that is a definitive miss, don't fall through to a
                    # looser text comparison.
                    return False
        # unresolvable on either side — fall through to text comparison

    pred_norm = normalize_answer(prediction, qtype)
    if not pred_norm:
        return False

    pred_values = number_variants(prediction)

    for gold in split_gold(ground_truth):
        gold_norm = normalize_answer(gold, qtype)
        if not gold_norm:
            continue

        if pred_norm == gold_norm:
            return True

        # 'Unanswerable' only ever matches 'Unanswerable'.
        if pred_norm == "unanswerable" or gold_norm == "unanswerable":
            continue

        gold_values = number_variants(gold)
        if pred_values and gold_values and numbers_match(
                pred_values, gold_values, tolerance):
            return True

    return False


# ─────────────────────────────────────────────────────────────
# Answer extraction from chain-of-thought output
# ─────────────────────────────────────────────────────────────

_FINAL_RE = re.compile(r"final\s*answer\s*[:\-–]\s*(.+)", re.IGNORECASE)

_LEGACY_MARKERS = [
    "the answer is", "correct answer is", "so the answer is",
    "answer:", "therefore,", "thus,",
]


def _first_clause(text: str) -> str:
    """
    First sentence of `text`, splitting only on a period that is followed
    by whitespace or end-of-string.

    The old code used `.split(".")[0]`, which turned 12.5 into 12 and
    '3.4 billion' into '3' — destroying every decimal answer in the set.
    """
    line = text.strip().split("\n")[0].strip()
    if not line:
        return ""
    parts = re.split(r"\.(?=\s|$)", line)
    return parts[0].strip() if parts else line


def extract_answer(raw: str, qtype: str = "factoid") -> str:
    """
    Pull the final answer out of a chain-of-thought generation.

    Priority:
      1. the strict 'Final Answer:' marker the prompt asks for
      2. (MCQ) a bracketed option letter
      3. legacy answer markers, for outputs that ignored the format
      4. last non-empty line

    Returns the answer with original casing; normalize_answer() handles
    canonicalization, so the raw span stays readable in logs.
    """
    if not raw:
        return ""
    text = raw.strip()

    matches = _FINAL_RE.findall(text)
    if matches:
        candidate = _first_clause(matches[-1])
        if candidate:
            if qtype == "mcq":
                # 'Final Answer: (B)' -> 'B'. Full-match only, so that a
                # word answer like 'Unanswerable' is not mined for a stray
                # letter and reduced to 'a'.
                m = re.fullmatch(r"\(?([a-hA-H])\)?[.):]?", candidate.strip())
                if m:
                    return m.group(1)
            return candidate

    if qtype == "mcq":
        letters = re.findall(r"\(([a-hA-H])\)", text)
        if letters:
            return letters[-1]
        for line in reversed(text.split("\n")):
            stripped = line.strip()
            if re.fullmatch(r"\(?[a-hA-H]\)?[.):]?", stripped):
                return stripped

    lowered = text.lower()
    for marker in _LEGACY_MARKERS:
        idx = lowered.rfind(marker)
        if idx != -1:
            candidate = _first_clause(text[idx + len(marker):])
            if candidate:
                return candidate

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return lines[-1] if lines else text
