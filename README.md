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
