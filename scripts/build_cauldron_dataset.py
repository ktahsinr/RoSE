#!/usr/bin/env python3
"""Build the first-N FigureQA or PlotQA factoid dataset for the IRIS backend.

CPU-only — no GPU needed. Mirrors the dataset cells of the routed Kaggle
notebooks EXACTLY: both stream the dataset's subset of
HuggingFaceM4/the_cauldron and take one question from each of the first N
usable charts in stream order, with ids keyed to the stream row index
(figureqa-000123 / plotqa-000045). A routed_<dataset>.json produced on
Kaggle therefore lines up with these questions by id.

    .venv/bin/python scripts/build_cauldron_dataset.py figureqa --n 100 \
        --out data/figureqa
    .venv/bin/python scripts/build_cauldron_dataset.py plotqa --n 100 \
        --out data/plotqa

Then serve all three datasets together (comma-separated lists):

    IRIS_DATASET_JSON=$PWD/data/chartqapro/dataset.json,$PWD/data/figureqa/dataset.json,$PWD/data/plotqa/dataset.json \
    IRIS_IMAGE_DIR=$PWD/data/chartqapro/images,$PWD/data/figureqa/images,$PWD/data/plotqa/images \
    make backend
"""

import argparse
import json
from pathlib import Path

MAX_GOLD_CHARS = 30                    # PlotQA: keep answers factoid-short


def figureqa_qa(texts):
    """FigureQA: the first QA per chart; answers are yes/no."""
    qa = texts[0]
    q = qa["user"].split("\n")[0].strip()          # drop "Answer yes or no."
    a = qa["assistant"].strip().rstrip(".")
    if not q or a.lower() not in ("yes", "no"):
        return None
    return q, a


def plotqa_qa(texts):
    """PlotQA: the first SHORT non-yes/no answer (true factoid)."""
    for qa in texts:
        q = qa["user"].split("\n")[0].strip()      # drop "Make the answer very short." etc.
        a = qa["assistant"].strip().rstrip(".")
        if not q or not a or len(a) > MAX_GOLD_CHARS:
            continue
        if a.lower() in ("yes", "no"):
            continue
        return q, a
    return None


PICKERS = {"figureqa": figureqa_qa, "plotqa": plotqa_qa}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", choices=sorted(PICKERS),
                    help="which the_cauldron subset to build")
    ap.add_argument("--n", type=int, default=100,
                    help="first N usable charts in stream order (default 100)")
    ap.add_argument("--out", default=None,
                    help="output directory (default data/<dataset>)")
    args = ap.parse_args()

    out = Path(args.out or f"data/{args.dataset}")
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    pick = PICKERS[args.dataset]

    from datasets import load_dataset
    print(f"streaming HuggingFaceM4/the_cauldron ({args.dataset}) from the HF Hub…")
    ds = load_dataset("HuggingFaceM4/the_cauldron", args.dataset,
                      split="train", streaming=True)

    records = []
    for row_idx, row in enumerate(ds):
        if len(records) >= args.n:
            break
        if not row["texts"] or not row["images"]:
            continue
        qa = pick(row["texts"])
        if qa is None:
            continue
        path = images / f"{args.dataset}_{row_idx:06d}.png"
        row["images"][0].convert("RGB").save(path, format="PNG")
        records.append({
            "id":            f"{args.dataset}-{row_idx:06d}",
            "question":      qa[0],
            "choices":       None,
            "answer":        qa[1],
            "question_type": "factoid",
            "image":         str(path),
        })
        if len(records) % 25 == 0:
            print(f"  {len(records)}/{args.n} charts written")

    (out / "dataset.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2))
    print(f"\n✓ {len(records)} questions → {out / 'dataset.json'}")
    print(f"✓ images → {images}")


if __name__ == "__main__":
    main()
