# RoSE — Reasoning with Orchestrated Streaming Experiences

A local, runnable implementation of the EMNLP 2024 paper
[*Making Large Language Models Better Reasoners with Orchestrated Streaming Experiences*](https://aclanthology.org/2024.emnlp-main.48/)
(Liu, He, Qiu — Fudan University), with a web frontend and backend.

Everything runs **locally on your Mac via [Ollama](https://ollama.com)** — no API keys, no cloud, no cost.

## What it does

RoSE wraps an LLM so it **self-improves as it answers questions**:

1. The LLM answers a reasoning question with chain-of-thought, sampling several
   paths and taking a **self-consistency** majority vote.
2. The answered question + its reasoning are stored in a **streaming experience pool**.
3. For each new question, RoSE embeds it, ranks pool items by **similarity**,
   splits them into **buckets** (for diversity — avoids the "copy effect"), and
   from each bucket picks the experience with **low uncertainty** and **high
   complexity** as a few-shot demonstration.
4. As the pool grows, accuracy climbs above the **Zero-Shot-CoT** and **Auto-CoT**
   baselines.

## The website

- **Interactive Demo** — pick a CommonsenseQA question, watch RoSE orchestrate
  experiences from the pool, and see its reasoning path, vote distribution,
  uncertainty/complexity metrics, and the demonstrations it selected — side by
  side with the two baselines.
- **Benchmark Dashboard** — stream N questions through all three methods and
  compare accuracy (mirrors the paper's Table 2 for CommonsenseQA), showing
  RoSE trending above the baselines.
- **ChartQAPro** — the vision extension. Reads finished GPU runs and shows
  accuracy breakdowns, the learning curve against pool size, the offline
  aggregation ablation, judge verdicts, and every question's reasoning paths.
  See [ChartQAPro Extension](#chartqapro-extension).

## Stack

| Layer      | Tech                                                    |
|------------|---------------------------------------------------------|
| LLM        | Ollama `llama3.1:8b` (reasoning) + `nomic-embed-text` (similarity) |
| Backend    | Node.js + Express (`backend/`)                          |
| Frontend   | React + Vite (`frontend/`)                              |
| Dataset    | CommonsenseQA (Kaggle JSONL) → normalized in `backend/data/` |

## Run it

```bash
./start.sh            # ensures Ollama + models, then launches both servers
# → open http://localhost:5173
```

Or manually:

```bash
# once: pull models + prep data
ollama pull llama3.1:8b && ollama pull nomic-embed-text
cd backend && npm install && npm run prepare-data && npm start
# in another shell:
cd frontend && npm install && npm run dev
```

## API

| Endpoint | Purpose |
|----------|---------|
| `GET  /api/health` | Ollama + model + pool status |
| `GET  /api/dataset?limit=N` | list CommonsenseQA questions |
| `POST /api/answer` | answer one question with RoSE (+ optional baselines) |
| `POST /api/benchmark` | stream N questions through all 3 methods |
| `POST /api/pool/warmup` `/reset` · `GET /api/pool` | manage the experience pool |
| `GET  /api/chartqapro/runs` | ChartQAPro result files found on disk |
| `GET  /api/chartqapro/summary?run=` | accuracy breakdowns, learning curve, ablation, judge |
| `GET  /api/chartqapro/rows?run=` · `/row?run=&id=` | browse questions · one question with every path |
| `GET  /api/chartqapro/export.csv?run=` | CSV export (`&perPath=1` for one row per path) |
| `POST /api/chartqapro/upload?name=&kind=` | import a run from a notebook |

## Notes on faithfulness

- The mechanism (streaming pool, similarity bucketing, uncertainty via
  self-consistency entropy, complexity via reasoning-step count, diversity
  selection) follows the paper.
- Differences for local practicality: the paper uses `gpt-3.5-turbo-16k` /
  `LLaMA2-13B` and `all-mpnet-base-v2`; this uses a local 8B model and
  `nomic-embed-text`, and defaults to fewer self-consistency paths. Absolute
  accuracy will differ from the paper; the **method and the relative gains** are
  what this reproduces.
- Only CommonsenseQA (1 of the paper's 9 benchmarks) is wired up. The data
  loader is structured so the other 8 (GSM8K, AQuA, AddSub, SingleEq, SingleOp,
  SVAMP, StrategyQA, Date) can be added as normalized JSON.

## ChartQAPro Extension

We extend RoSE from text-only (CommonsenseQA) to vision (chart images). The
mechanism is unchanged — streaming pool, similarity bucketing, entropy
uncertainty, complexity selection — but the questions are now about charts the
model has to read.

| Dataset | Model | Questions | Method |
|---------|-------|-----------|--------|
| CommonsenseQA | llama3.1:8b (local) | 1,221 | Original RoSE |
| ChartQAPro | Qwen2.5-VL-7B-Instruct (4-bit) | Factoid + MCQ | RoSE-CQA (this work) |

Inference needs a GPU, so it runs in a notebook (Colab / Kaggle T4) rather than
on the Ollama stack — but everything **after** inference is CPU-only. A run logs
every sampled reasoning path, so it can be re-scored, ablated, judged and
exported on a laptop without touching a GPU again.

### The pipeline

| File | Does |
|------|------|
| `chartqapro/rose_chartqapro.py` | the run: pool, orchestration, m sampled paths, voting |
| `chartqapro/scoring.py` | answer extraction, normalization, correctness — no GPU deps, so it is the single source of truth for every verdict |
| `chartqapro/ablate.py` | recomputes the vote under all four aggregation configs, offline |
| `chartqapro/rescore_judge.py` | LLM judge for answers that are right but written differently |
| `chartqapro/export_csv.py` | results → CSV (`--per-path` for one row per reasoning path) |
| `chartqapro/summarize.py` | the whole run as JSON — what the web dashboard reads |

### Extensions beyond the paper

Ten deviations, each separately switchable, so a faithful baseline and the
improved system can both be reported (`PAPER_FAITHFUL = True` disables all ten).
The three that only change how the m per-path answers are **combined** can be
ablated from a finished run; the rest change what the model is shown or how it
decodes, so each needs its own run.

| Extension | Ablatable offline? |
|---|---|
| `numeric_vote_clustering` — group numeric answers within tolerance, take the median of the largest group | **yes** |
| `drop_malformed_paths` — exclude paths with no `Final Answer:` line from the vote | **yes** |
| `confidence_weighted_vote` — weight each path by decoding and well-formedness | **yes** |
| `mcq_permute_options` — rotate MCQ option order across paths so position bias cancels | no (prompt) |
| `type_aware_retrieval` — retrieve demonstrations only from the same question type | no (retrieval) |
| `greedy_first_path` — decode path 0 at T=0 | no (decoding) |
| `chart_reading_scaffold` — axis-and-units-first prompt scaffold | no (prompt) |
| `mcq_demo_anchor_fix` — store MCQ demos as `(B) 45%` so the letter stays anchored to a value | no (pool contents) |
| `two_stage_reasoning` — describe the chart first, then answer from that description | no (prompt) |
| `visual_hybrid_retrieval` — blend CLIP image similarity into retrieval | no (retrieval) |

### Run it

```bash
# 1. inference (GPU) — Notebook/RoSE_ChartQAPro_Notebook.ipynb
#    writes results/rose_results.json + results/meta.json

# 2. everything else is CPU-only
cd chartqapro
python ablate.py        results/rose_results.json --by-type
python rescore_judge.py results/rose_results.json --judge ollama \
       --out results/verdicts.json
python export_csv.py    results/rose_results.json --per-path
```

### See it in the app

Drop the run's JSON files into `chartqapro/results/` (or import them from the
page) and open the **ChartQAPro** tab at
[localhost:5173/#cqa](http://localhost:5173/#cqa). It shows accuracy by question
type and by phase, answerable-vs-unanswerable confusion, accuracy against a
growing pool, whether the entropy signal is calibrated, the offline aggregation
ablation, the judge's verdicts, and every question with its individual reasoning
paths. Numbers come from `summarize.py`, so the dashboard and the CLI tables can
never disagree. The backend shells out to `python3` for this — set `ROSE_PYTHON`
if it is not on your PATH.
