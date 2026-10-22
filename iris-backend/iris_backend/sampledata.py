"""Synthetic demo data: charts, a dataset JSON and a stored alg2 run.

The real decomp_alg2 outputs live on Kaggle; this module builds a small,
clearly synthetic stand-in with the exact field shapes decomp_runner writes,
so demo mode, frontend development and CI work without them. Generated with:

    python -m iris_backend.sampledata iris-backend/sample_data
"""

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

# (id suffix, question — one stored as a 1-element list to exercise the
#  unwrap fix, title, {label: value}, unit, gold)
CHARTS = [
    ("00000", "In which year did revenue peak?", "Annual revenue (sample data)",
     {"2019": 48, "2020": 31, "2021": 42, "2022": 45}, "$M", "2019"),
    ("00001", ["What was the unemployment rate in March?"],
     "Unemployment rate by month (sample data)",
     {"Jan": 6, "Feb": 5, "Mar": 8, "Apr": 7}, "%", "8"),
    ("00002", "How many more units did the North region sell than the South?",
     "Units sold by region (sample data)",
     {"North": 120, "South": 85, "East": 95, "West": 60}, "units", "35"),
    ("00003", "Which product had the lowest sales?", "Sales by product (sample data)",
     {"Alpha": 54, "Beta": 23, "Gamma": 61, "Delta": 38}, "k", "Beta"),
    ("00004", "What is the difference between the highest and lowest bars?",
     "Rainfall by season (sample data)",
     {"Spring": 210, "Summer": 90, "Autumn": 180, "Winter": 150}, "mm", "120"),
    ("00005", "What share of the total do cars represent?",
     "Vehicles registered (sample data)",
     {"Cars": 50, "Trucks": 25, "Bikes": 15, "Buses": 10}, "%", "50"),
]


def _draw_chart(path: Path, title: str, data: dict, unit: str):
    w, h = 640, 420
    img = Image.new("RGB", (w, h), "#FFFFFF")
    d = ImageDraw.Draw(img)
    d.text((24, 16), title, fill="#1A1A2E")
    d.text((24, 36), f"values in {unit}", fill="#555577")
    left, bottom, top = 60, h - 60, 80
    maxv = max(data.values())
    bw = (w - left - 40) // len(data)
    d.line([(left, top), (left, bottom), (w - 30, bottom)], fill="#333355", width=2)
    for i, (label, v) in enumerate(data.items()):
        x0 = left + 12 + i * bw
        bh = int((bottom - top - 10) * v / maxv)
        d.rectangle([x0, bottom - bh, x0 + bw - 24, bottom], fill="#5347B8")
        d.text((x0, bottom + 8), label, fill="#1A1A2E")
        d.text((x0, bottom - bh - 18), str(v), fill="#1A1A2E")
    img.save(path)


def _sub_queries(question: str, data: dict) -> list:
    keys = list(data)
    return [f"What are the values shown for {keys[0]} and {keys[-1]}?",
            f"Using those values, {question[0].lower() + question[1:]}"]


def _alg2_row(i: int, rec: dict, data: dict) -> dict:
    """One stored result in decomp_runner's JSON row shape."""
    q = rec["question"]
    subs = _sub_queries(q, data)
    gemma = subs
    qwen = [subs[0].replace("What are", "Which are"), subs[1]]
    rationale = (f"Step 1: {subs[0]} The chart shows the plotted values.\n"
                 f"Step 2: {subs[1]} Comparing them gives the result.\n"
                 f"Final Answer: {rec['answer']}")
    raw_outputs = [rationale,
                   rationale,
                   rationale.replace("Comparing", "Reading off and comparing")]
    return {
        "id": rec["id"], "question": q, "question_type": "factoid",
        "choices": None, "options_shown": None, "option_perms": None,
        "ground_truth": rec["answer"],
        "prediction": str(rec["answer"]),
        "prediction_norm": str(rec["answer"]).lower(),
        "is_correct": True,
        "method": "zero_shot_cot" if i < 3 else "rose_few_shot",
        "uncertainty": round(0.15 + 0.05 * i, 4),
        "complexity": 2.0,
        "agreement": 1.0 if i % 2 == 0 else round(2 / 3, 4),
        "vote_mode": "exact", "n_groups": 1, "paths_voting": 3,
        "n_demos": 0 if i < 3 else 3,
        "pool_size": i,
        "chart_description": (
            f"A vertical bar chart titled “{CHARTS[i][2]}” with "
            f"{len(data)} bars ({', '.join(data)}), y-axis in {CHARTS[i][4]}. "
            f"The tallest bar is {max(data, key=data.get)} "
            f"({max(data.values())}) and the shortest is "
            f"{min(data, key=data.get)} ({min(data.values())})."),
        "best_rationale": rationale,
        "raw_outputs": raw_outputs,
        "extracted": [str(rec["answer"])] * 3,
        "normalized": [str(rec["answer"]).lower()] * 3,
        "algorithm": 2,
        "sub_queries": subs,
        "sub_answers": [], "sub_confidences": [], "sub_models": [],
        "decomp_fallback": {"gemma3-4b": False, "qwen3.5-4b": False},
        "vote_rule": "merge" if i % 2 == 0 else "pick:gemma3-4b",
        "proposals": {"gemma3-4b": gemma, "qwen3.5-4b": qwen},
        "vote": {
            "rule": "merge" if i % 2 == 0 else "pick",
            "picked_model": None if i % 2 == 0 else "gemma3-4b",
            "matching": "straight",
            "pair_sims": [0.91, 0.87] if i % 2 == 0 else [0.62, 0.55],
            "mean_sim_to_question": {"gemma3-4b": 0.74, "qwen3.5-4b": 0.69},
            "threshold": 0.8,
        },
        "stage_latency_s": {"decompose_gemma3-4b": 9.4,
                            "decompose_qwen3.5-4b": 12.1,
                            "choose": 0.2, "rose_final": 96.5},
        "params_b_used": 16.6,
        "latency_s": 118.2,
        "model_key": "qwen2.5-vl-7b",
    }


def make_sample_data(dest: Path) -> dict:
    dest = Path(dest)
    images = dest / "images"
    results = dest / "results"
    images.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    records = []
    for i, (suffix, question, title, data, unit, gold) in enumerate(CHARTS):
        name = f"{suffix}.png"
        _draw_chart(images / name, title, data, unit)
        records.append({"id": f"factoid-{suffix}", "question": question,
                        "choices": None, "answer": gold,
                        "question_type": "factoid",
                        "image": f"/kaggle/temp/images/{name}"})
    (dest / "dataset.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2))

    # Stored alg2 results for the first 4 questions only, so the frontend
    # also exercises the "no stored result" state.
    rows = []
    for i, rec in enumerate(records[:4]):
        rec_unwrapped = dict(rec, question=(
            rec["question"][0] if isinstance(rec["question"], list)
            else rec["question"]))
        rows.append(_alg2_row(i, rec_unwrapped, CHARTS[i][3]))
    (results / "decomp_alg2.json").write_text(json.dumps(rows, indent=2))
    (results / "decomp_alg2_meta.json").write_text(json.dumps({
        "algorithm": 2, "n": len(rows), "n_total": len(records),
        "complete": False, "accuracy_rescored": 100.0,
        "note": "SYNTHETIC sample data for demo mode — not a real run",
    }, indent=2))

    return {"dataset_json": dest / "dataset.json", "image_dir": images,
            "results_dir": results}


if __name__ == "__main__":
    out = make_sample_data(Path(sys.argv[1] if len(sys.argv) > 1
                                else "sample_data"))
    print(json.dumps({k: str(v) for k, v in out.items()}, indent=2))
