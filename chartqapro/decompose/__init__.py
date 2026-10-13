# Query decomposition for RoSE on ChartQAPro — two algorithms, no routing.
#
# Both algorithms split a chart question into two simpler sub-queries and
# then run the repo's ORIGINAL RoSE pipeline (rose_chartqapro.process_one,
# all hyperparameters unchanged) on Qwen2.5-VL-7B for the final answer.
# They differ in what the small models contribute:
#
#   Algorithm 1 — decompose → parallel answer → evidence → RoSE
#     The 7B splits the question; Gemma 3 4B answers sub-query 1 and
#     Qwen3.5-4B answers sub-query 2 (3 sampled paths each, agreement
#     ratio as confidence); the sub-answers are prepended to the RoSE
#     reasoning prompt as an evidence block.
#
#   Algorithm 2 — both small models decompose → vote on decomposition → RoSE
#     Gemma and Qwen3.5 each propose two sub-queries; an embedding vote
#     (all-mpnet-base-v2) merges agreeing proposals or picks the one
#     closer to the original question; the chosen sub-queries are inserted
#     as "answer these in order, then the main question" steps. No small
#     model answers anything.
#
#   prompts.py       decomposition / sub-answer prompt + injected blocks
#   decomposer.py    prompt + JSON parser + [question, question] fallback
#   subanswer.py     3-path self-consistency sub-answer with confidence
#   vote.py          answer_evidence (Alg 1) + choose_decomposition (Alg 2)
#   models.py        standalone model registry + loader (NO routing import)
#   decomp_runner.py CLI: checkpointed, budgeted, CSV + meta, both algorithms
#
# This package NEVER imports chartqapro.routing — no kNN router, no
# routing pool, no escalation. tests/test_decompose.py enforces that.

from .decomposer import decompose, parse_subqueries
from .vote import answer_evidence, choose_decomposition
