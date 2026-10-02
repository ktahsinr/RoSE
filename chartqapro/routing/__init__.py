# Experience-based kNN model routing for RoSE on ChartQAPro.
#
# The idea in one line: questions a small model has answered correctly in
# the past look like questions it will answer correctly in the future — so
# embed every pilot question, remember which models got it right
# ("experience"), and send each NEW question to the smallest model whose
# nearest pilot neighbours suggest it will cope. The 7B model is the
# escalation target, never a gamble.
#
#   pool.py            build/load the routing pool from pilot result files
#   router.py          the kNN router + leave-one-out evaluation
#   evaluate_pilot.py  CLI: every pilot number (accuracy, oracle, latency…)
#   pilot_runner.py    GPU-side: run ONE model over the pilot sample
#   config.py          model registry + routing hyperparameters
#
# Everything except pilot_runner.py is CPU-only, in keeping with the rest
# of chartqapro/: inference happens once on a GPU, every later question is
# answered from the logs.

from .config import MODELS, ESCALATION_ORDER, K_NEIGHBOURS, TAU, EMBED_MODEL
from .pool import build_pool, load_pool, save_pool
from .router import Router, leave_one_out
