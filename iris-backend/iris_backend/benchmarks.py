"""Benchmark tables served by GET /benchmarks.

Published numbers are transcribed from the ChartQAPro paper (Masry et al.,
2025). NOTE / verify before release: the column headers below (Factoid,
Multiple Choice, Hypothetical, Fact Checking, Conversational, Overall) must
be checked against the paper's table — the transcription order was supplied
as-is and has not been verified against the PDF.
"""

COLUMNS = ["Factoid", "Multiple Choice", "Hypothetical", "Fact Checking",
           "Conversational", "Overall"]

# (model, group, [scores in COLUMNS order])
PUBLISHED_ROWS = [
    ("GPT4-o", "Closed-source", [35.76, 46.72, 34.75, 45.49, 28.91, 37.67]),
    ("Gemini-Flash-2.0", "Closed-source", [43.43, 60.28, 40.25, 67.62, 24.47, 46.85]),
    ("Gemini-Flash-1.5", "Closed-source", [39.96, 57.00, 39.70, 47.13, 45.31, 42.96]),
    ("Claude Sonnet 3.5", "Closed-source", [38.84, 51.40, 44.53, 55.60, 45.48, 43.58]),
    ("Intern-VL2.5-1B", "Open-source, small", [9.15, 7.00, 6.20, 16.63, 8.17, 9.33]),
    ("Janus-1.3B", "Open-source, small", [4.56, 1.86, 6.74, 40.98, 5.31, 9.21]),
    ("Qwen-VL2-2B", "Open-source, small", [15.90, 27.57, 24.26, 34.42, 12.82, 20.68]),
    ("Intern-VL2.5-2B", "Open-source, small", [13.86, 10.74, 14.02, 45.90, 18.92, 17.81]),
    ("SmolVLM-2.3B", "Open-source, small", [13.32, 16.82, 17.71, 46.31, 25.21, 19.14]),
    ("Ovis1.6-LLama3.2-3B", "Open-source, small", [12.87, 0.46, 4.18, 40.98, 10.17, 13.50]),
    ("DeepSeek-VL2-3.4B", "Open-source, small", [12.20, 7.47, 19.40, 36.88, 19.21, 16.28]),
    ("Phi 3.5-Vision-4B", "Open-source, small", [17.48, 30.37, 28.54, 41.99, 37.27, 24.73]),
    ("Qwen-VL2-7B", "Open-source, large", [30.70, 44.85, 35.68, 48.36, 37.23, 35.59]),
    ("Intern-VL2.5-8B", "Open-source, large", [35.21, 25.70, 32.26, 53.27, 29.61, 35.67]),
    ("Idefics-3-LLama-3.1-8B", "Open-source, large", [20.69, 2.29, 31.96, 10.76, 36.83, 20.03]),
    ("LLaVA-Next-Mistral-7B", "Open-source, large", [15.35, 35.98, 21.09, 41.80, 17.79, 21.97]),
    ("Ovis1.6-Gemma2-9B", "Open-source, large", [30.25, 4.67, 28.93, 27.86, 28.21, 26.83]),
    ("LLama 3.2-Vision-11B", "Open-source, large", [12.34, 2.33, 0.19, 27.18, 10.93, 11.09]),
]

# IRIS results on ChartQAPro factoid, first 250 questions. 95% CIs.
IRIS_ROWS = [
    {"key": "rose-7b", "label": "Existing RoSE 7B",
     "accuracy_pct": 38.8, "correct": 97, "n": 250, "ci": [33.0, 45.0],
     "settings": "M=5 paths, two-stage description on"},
    {"key": "alg1", "label": "Algorithm 1 (decompose → sub-answers → RoSE)",
     "accuracy_pct": 42.8, "correct": 107, "n": 250, "ci": [36.8, 49.0],
     "settings": "M=3 paths, two-stage off"},
    {"key": "alg2", "label": "Algorithm 2 (decompose → vote → RoSE)",
     "accuracy_pct": 47.6, "correct": 119, "n": 250, "ci": [41.5, 53.8],
     "settings": "M=3 paths, two-stage off"},
    {"key": "routing", "label": "Model-routing extension",
     "accuracy_pct": 24.0, "correct": 60, "n": 250, "ci": [19.1, 29.7],
     "settings": ""},
]

IRIS_FOOTNOTE = (
    "Settings and question slices differ between arms (paths sampled, "
    "two-stage description, image resolution), so the arms are indicative "
    "rather than strictly comparable. The Algorithm 2 run predates fixes to "
    "known defects (a fallback decomposition could win the vote with cosine "
    "1.0, and list-wrapped questions reached the models as \"['…']\"); both "
    "are fixed in this codebase."
)


def payload() -> dict:
    """The full /benchmarks response. Best-per-column is computed, not typed."""
    best = [max(r[2][c] for r in PUBLISHED_ROWS) for c in range(len(COLUMNS))]
    rows = [{
        "model": name,
        "group": group,
        "scores": scores,
        "best": [scores[c] == best[c] for c in range(len(COLUMNS))],
    } for name, group, scores in PUBLISHED_ROWS]
    return {
        "published": {
            "citation": "ChartQAPro (Masry et al., 2025)",
            "columns": COLUMNS,
            "rows": rows,
        },
        "iris": {
            "title": "IRIS results on ChartQAPro factoid (250 questions)",
            "rows": IRIS_ROWS,
            "footnote": IRIS_FOOTNOTE,
        },
    }
