# =============================================================
# Prompts for query decomposition — pure strings, no GPU deps
# =============================================================
# Everything the decompose arms ever say to a model lives here, so the
# exact wording is reviewable (and testable) in one place.
# =============================================================

# ── decomposition (both algorithms, same prompt) ─────────────
# Sent with the chart image. Asks for a JSON list of exactly two
# sub-queries; decomposer.parse_subqueries handles every way a model can
# get that wrong (prose around the JSON, code fences, wrong arity,
# no JSON at all → [question, question] fallback).

_DECOMPOSITION_TEMPLATE = (
    "Look at the chart. You will NOT answer the question below. Instead, "
    "split it into exactly TWO simpler sub-queries that, answered in "
    "order from this chart, make the original question easy to answer.\n\n"
    "Rules:\n"
    "  - each sub-query must be answerable from this chart alone\n"
    "  - sub-query 1 should extract the value(s) or category the question "
    "hinges on\n"
    "  - sub-query 2 should cover the remaining comparison, computation "
    "or lookup\n"
    "  - do not answer anything; do not explain\n\n"
    "Question: {question}\n\n"
    "Respond with ONLY a JSON list of two strings, nothing else. "
    'Example format: ["first sub-query", "second sub-query"]'
)


def decomposition_prompt(question: str) -> str:
    return _DECOMPOSITION_TEMPLATE.format(question=question)


# ── sub-answer (Algorithm 1 only) ────────────────────────────
# A small model answers ONE sub-query against the chart. Same bare-value
# contract as the RoSE factoid format, so scoring.extract_answer /
# normalize_answer apply unchanged.

_SUB_ANSWER_TEMPLATE = (
    "Look at the chart carefully and answer the question.\n\n"
    "Question: {sub_query}\n\n"
    "Let's think step by step.\n\n"
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


def sub_answer_prompt(sub_query: str) -> str:
    return _SUB_ANSWER_TEMPLATE.format(sub_query=sub_query)


# ── blocks injected into the RoSE reasoning prompt ───────────
# These are PREPENDED to the pipeline's own zero-shot / few-shot prompt
# (decomp_runner wraps the prompt builders for the duration of one
# process_one call); the RoSE prompt itself is untouched.

EVIDENCE_HEADER = (
    "Sub-question evidence from helper models (verify each claim against "
    "the chart before relying on it; a low confidence means the helper's "
    "sampled answers disagreed):"
)


def subquery_steps_block(sub_queries: list) -> str:
    """Algorithm 2: the chosen sub-queries as ordered reasoning steps."""
    steps = "\n".join(f"  Step {i}: {q}" for i, q in enumerate(sub_queries, 1))
    return (
        "Answer these sub-questions in order first, then the main "
        "question:\n" + steps + "\n\n"
    )
