"""Dataset loading for the IRIS backend.

Permitted change #3 lives here: ChartQAPro stores "Question" as a list, and a
record that reached a dataset JSON un-unwrapped shows the model the literal
string "['…']". unwrap_question() fixes both shapes — a real one-element list
and its stringified form — in the loader the backend uses. Stored result
files are never rewritten.
"""

import ast
import json
from pathlib import Path


def unwrap_question(q) -> str:
    """One-element question lists → their element; everything else unchanged.

    Handles both a real list (["How many…?"]) and the stringified form
    ("['How many…?']") that str() of such a list produces.
    """
    if isinstance(q, (list, tuple)):
        if len(q) == 1:
            return str(q[0]).strip()
        return str(q).strip()
    s = str(q).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return s
        if isinstance(parsed, (list, tuple)) and len(parsed) == 1:
            return str(parsed[0]).strip()
    return s


def load_dataset(path: Path) -> list:
    """Load a dataset JSON (the notebooks' record shape) into API records.

    Each record keeps the run files' field names; `image` is reduced to its
    basename so charts are served from IRIS_IMAGE_DIR regardless of where
    the dataset was built (Kaggle paths are absolute).
    """
    rows = json.loads(Path(path).read_text())
    records = []
    for row in rows:
        image = str(row.get("image", "") or "")
        records.append({
            "id": str(row.get("id", "")),
            "question": unwrap_question(row.get("question", "")),
            "question_type": str(row.get("question_type", "factoid")).lower(),
            "choices": row.get("choices"),
            "ground_truth": row.get("answer", ""),
            "image_name": Path(image).name if image else "",
        })
    return records
