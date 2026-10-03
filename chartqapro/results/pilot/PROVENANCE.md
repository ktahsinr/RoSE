# Provenance of the cached 7B factoid results

The routing baseline's 7B answers come from `results/rose_factoid.json`,
a normalized copy of a teammate's **August 2026** run. (An earlier July
checkpoint was wired in first by mistake; it has been replaced, and the
seed-499 pilot sample was redrawn from this file.)

## The source file

| | |
|---|---|
| File | `factoid_first_rose_results.json` (md5 `e3dc0212f2174172693c1fde795e60da`) |
| Local copy | `results/checkpoint_factoid_aug_raw.json` (gitignored) |
| Pipeline | the **August notebook** era — stored `is_correct` agrees with the current `scoring.py` on every row, so the scoring/extraction fixes are in |
| Model | Qwen2.5-VL-7B-Instruct, 4-bit |
| Coverage | **the first 540 of 1,081** ChartQAPro factoid questions, in dataset order (`factoid-00000` … `factoid-00539`; the file's original underscore ids were renamed to the repo's hyphen convention — positions are identical) |
| Accuracy | **32.8 %** (177/540); rose_few_shot 33.0 % (177/537), zero_shot_cot 0/3 (pool warm-up rows) |
| Unanswerable share | 29.1 % (157/540) — the pilot sample mirrors it at 29.3 % (44/150) |

## History

The factoid sweep was split between two teammates, each covering half
the set. Only the first member's results — this file, questions
0–539 — could be retrieved. **Questions 540–1080 have no cached 7B
answers.**

## Known limitations (permanent for this file)

- **Half coverage.** The pilot sample and routing pool are confined to
  the first 540 questions. For Stage 2, the "always-7B" comparison
  column exists only for those 540; a routed-vs-7B comparison over the
  full set requires re-running the 7B arm on questions 540–1080 (or
  reporting the comparison on the first half only, clearly labelled).
- **No per-question latency** → the 7B column of any time budget must
  come from a fresh run or an estimate, not from this file.
- **One logged path per question** (`raw_outputs` has a single entry,
  `uncertainty` 0.0, `agreement` 1.0 throughout) → no offline
  aggregation ablation for this run, and its uncertainty values carry
  no signal.

## Superseded file

The previous source — a July-pipeline `checkpoint_factoid.json`
(md5 `fbac81df3fa8ed79e03b4f7df2d41fc1`, all 1,081 questions, 21.0 %
rescored) — predates the scoring fixes and was the wrong artifact.
Its conversion is kept locally as `results/rose_factoid_july_converted.json`
(gitignored) for reference only; nothing downstream reads it.
