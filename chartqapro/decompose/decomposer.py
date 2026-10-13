# =============================================================
# Decomposer — prompt + JSON parser + fallback  (CPU, no GPU deps)
# =============================================================
# The model is asked for a JSON list of exactly two sub-queries.
# parse_subqueries() ALWAYS returns two non-empty strings: anything the
# model gets wrong (prose around the JSON, code fences, wrong arity,
# non-string items, no JSON at all) degrades to the documented fallback
# [question, question] — the pipeline then behaves like plain RoSE with
# the question asked twice, never crashes.
# =============================================================

import json
import re

try:
    from . import prompts
except ImportError:                 # flat / notebook use
    import prompts

# Code fences first, then any bracketed span that parses as JSON.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_ARRAY_RE = re.compile(r"\[[^\[\]]*\]", re.DOTALL)


def _candidate_payloads(raw: str) -> list:
    """Every substring of `raw` worth trying json.loads on, best first."""
    text = raw.strip()
    cands = [text]
    for m in _FENCE_RE.finditer(text):
        cands.append(m.group(1).strip())
    for m in _ARRAY_RE.finditer(text):
        cands.append(m.group(0))
    return cands


def _two_strings(parsed) -> "list | None":
    """A JSON value → exactly two non-empty strings, or None."""
    if not isinstance(parsed, list):
        return None
    items = []
    for it in parsed:
        if isinstance(it, dict):        # e.g. {"sub_query": "..."} — refuse,
            return None                  # guessing a key invites silent junk
        s = str(it).strip()
        if s:
            items.append(s)
    if len(items) < 2:
        return None
    return items[:2]                     # over-long lists: first two


def parse_subqueries(raw: str, question: str):
    """
    Parse a decomposition generation into exactly two sub-queries.

    Returns (sub_queries, used_fallback):
      sub_queries   list of exactly two non-empty strings
      used_fallback True when the output could not be parsed and the
                    documented fallback [question, question] was used
    """
    if raw:
        for payload in _candidate_payloads(str(raw)):
            try:
                parsed = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                continue
            two = _two_strings(parsed)
            if two:
                return two, False
    return [question, question], True


def decompose(question: str, call_fn):
    """
    Split `question` into two sub-queries with one model call.

    call_fn(prompt) -> raw generation string. The caller owns the model,
    the chart image and the decoding config (greedy, token budget), so
    this function stays GPU-free and directly testable.

    Returns (sub_queries, used_fallback, raw_output).
    """
    raw = call_fn(prompts.decomposition_prompt(question))
    sub_queries, used_fallback = parse_subqueries(raw, question)
    return sub_queries, used_fallback, raw
