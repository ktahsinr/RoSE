"""IRIS backend — FastAPI server over the chartqapro decompose pipeline.

The chartqapro/ directory is a flat package used via sys.path (exactly as
chartqapro/tests/ does), so importing iris_backend makes `decompose`,
`rose_chartqapro` and `scoring` importable. Heavy modules (torch,
transformers) are only imported by live-run code paths, never in demo mode.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CHARTQAPRO_DIR = REPO_ROOT / "chartqapro"
if str(CHARTQAPRO_DIR) not in sys.path:
    sys.path.insert(0, str(CHARTQAPRO_DIR))
