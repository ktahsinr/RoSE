#!/usr/bin/env python3
"""Zero-cost constraint check: IRIS source may not reference external hosts.

Fails (exit 1) when any IRIS source file contains a URL whose hostname is not
on the allowlist: localhost, 127.0.0.1/0.0.0.0, huggingface.co (weight
downloads). The IRIS_API_URL environment variable is how the frontend reaches
the backend, so code referencing that variable is fine by construction — this
script only hunts literal hostnames. xmlns namespace identifiers (w3.org) are
identifiers, not network calls, and are allowed.

Run from the repo root:  python scripts/check_hostnames.py
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SCAN_DIRS = ["iris-backend", "iris-frontend"]
EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
              ".css", ".html", ".json", ".yml", ".yaml"}
SKIP_PARTS = {"node_modules", ".next", "__pycache__", "test-results",
              "playwright-report", "sample_data"}

ALLOWED = re.compile(
    r"^(localhost|127\.0\.0\.1|0\.0\.0\.0|"
    r"([a-z0-9-]+\.)*huggingface\.co|"
    r"www\.w3\.org)$", re.IGNORECASE)

URL_RE = re.compile(r"(?:https?:)?//([a-z0-9.-]+\.[a-z]{2,}|localhost|"
                    r"127\.0\.0\.1|0\.0\.0\.0)(?::\d+)?", re.IGNORECASE)


def main() -> int:
    bad = []
    for d in SCAN_DIRS:
        base = ROOT / d
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if (not path.is_file() or path.suffix not in EXTENSIONS
                    or SKIP_PARTS & set(path.parts)):
                continue
            text = path.read_text(errors="ignore")
            for m in URL_RE.finditer(text):
                host = m.group(1).split(":")[0]
                if not ALLOWED.match(host):
                    line = text[:m.start()].count("\n") + 1
                    bad.append(f"{path.relative_to(ROOT)}:{line}: {m.group(0)}")
    if bad:
        print("External hostnames found (zero-cost constraint violated):")
        print("\n".join(f"  {b}" for b in bad))
        return 1
    print("✓ no external hostnames in IRIS source")
    return 0


if __name__ == "__main__":
    sys.exit(main())
