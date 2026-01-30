"""
Deprecated. Use scripts/download_assets.py instead.
"""

import sys

try:
    from . import _path  # type: ignore  # ensures project root is on sys.path
except ImportError:
    import _path  # type: ignore  # noqa: F401

from scripts.download_assets import main


if __name__ == "__main__":
    print("[warn] download_pdfs.py is deprecated; use download_assets.py")
    sys.exit(main() or 0)
