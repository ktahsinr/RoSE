#!/usr/bin/env python3
"""Build the first-N ChartQAPro factoid dataset for the IRIS backend.

CPU-only — no GPU needed. Mirrors the dataset cell of the decomp notebooks
exactly: ids are factoid ordinals in split order (factoid-00000 …), so the
rows line up by id with any decomp_alg1/alg2 run produced on Kaggle, and a
dropped-in decomp_alg2.json replays against these questions.

    .venv/bin/python scripts/build_chartqapro_dataset.py --n 250 \
        --out data/chartqapro

Then serve it:

    IRIS_DATASET_JSON=$PWD/data/chartqapro/dataset.json \
    IRIS_IMAGE_DIR=$PWD/data/chartqapro/images \
    make backend
"""

import argparse
import ast
import io
import json
from pathlib import Path


def _gold(v):
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    s = str(v).strip()
    if s.startswith("["):
        try:
            p = ast.literal_eval(s)
            if isinstance(p, (list, tuple)):
                return [str(x) for x in p]
        except Exception:
            pass
    return s


def _save_image(field, path: Path):
    from PIL import Image
    if isinstance(field, dict) and field.get("bytes"):
        img = Image.open(io.BytesIO(field["bytes"]))
    elif isinstance(field, (bytes, bytearray)):
        img = Image.open(io.BytesIO(field))
    elif hasattr(field, "convert"):          # PIL image
        img = field
    else:
        return str(field)
    img.convert("RGB").save(path, format="PNG")
    return str(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=250,
                    help="first N factoid questions in split order (default 250)")
    ap.add_argument("--out", default="data/chartqapro",
                    help="output directory (dataset.json + images/)")
    args = ap.parse_args()

    out = Path(args.out)
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    print("downloading ahmed-masry/ChartQAPro (test split) from the HF Hub…")
    ds = load_dataset("ahmed-masry/ChartQAPro", split="test")

    qtypes = [str(t).strip().lower() for t in ds["Question Type"]]
    fact_idx = [i for i, t in enumerate(qtypes) if t == "factoid"]
    print(f"factoid questions in split: {len(fact_idx)}")

    take = min(args.n, len(fact_idx))
    records = []
    for idx in range(take):                  # idx = factoid ordinal = id
        row = ds[fact_idx[idx]]
        records.append({
            "id":            f"factoid-{idx:05d}",
            "question":      str(row.get("Question", "")).strip(),
            "choices":       None,
            "answer":        _gold(row.get("Answer")),
            "question_type": "factoid",
            "image":         _save_image(row.get("image"),
                                         images / f"{idx:05d}.png"),
        })
        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{take} images written")

    (out / "dataset.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2))
    print(f"\n✓ {len(records)} questions → {out / 'dataset.json'}")
    print(f"✓ images → {images}")
    print("\nServe it:\n"
          f"  IRIS_DATASET_JSON=$PWD/{out}/dataset.json \\\n"
          f"  IRIS_IMAGE_DIR=$PWD/{out}/images \\\n"
          "  make backend")


if __name__ == "__main__":
    main()
