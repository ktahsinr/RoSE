# Provenance of the cached 7B factoid results

The routing baseline's 7B answers come from `results/rose_factoid.json`,
which is NOT a run of the current pipeline. It was produced by
converting a legacy checkpoint with `scripts/convert_legacy_results.py`.

## The source file

| | |
|---|---|
| File | `checkpoint_factoid.json` (md5 `fbac81df3fa8ed79e03b4f7df2d41fc1` — identical copies in Downloads and Desktop) |
| Local copy | `results/checkpoint_factoid_legacy.json` (gitignored) |
| Pipeline | the **July 2026 notebook** (`RoSE_ChartQAPro_Notebook.ipynb`-era), *before* the scoring/extraction fixes and the ten extensions |
| Model | Qwen2.5-VL-7B-Instruct, 4-bit |
| Coverage | all 1,081 ChartQAPro factoid questions, in dataset order (`_pool_size` = position+1 throughout, so position-derived ids `factoid-00000`… are valid) |
| Rescored accuracy | **21.0 %** (227/1081) with the current `scoring.py` |

## History

The factoid sweep was run by two teammates. Only one member's final
results could be retrieved — this file. The retrieved run covers the
full factoid set on its own, so no questions are missing; what was lost
is a second, duplicate run that could have served as a cross-check.

## Known limitations (permanent for this file)

- Predictions suffer the old extraction bugs (many truncated spans such
  as `"[85"`), so 21.0 % understates what the current pipeline achieves.
  Any routed-vs-7B comparison built on this file flatters routing; say
  so wherever the numbers are reported, or re-run the 7B arm with the
  current pipeline (at minimum over the 150 pilot questions).
- No per-question latency → the 7B column of any time budget must come
  from a fresh run or an estimate, not from this file.
- No per-path logs → no offline aggregation ablation for this run.
