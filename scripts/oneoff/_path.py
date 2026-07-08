"""
Bootstrap helper to make the project root importable when running an archived
one-off script directly (e.g. `python scripts/oneoff/reprocess_html_backfill.py`).

These scripts live one directory deeper than `scripts/`, so the project root is
`parents[2]` here (oneoff -> scripts -> root).
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
