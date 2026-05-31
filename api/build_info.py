from __future__ import annotations

import functools
import subprocess
from pathlib import Path


@functools.lru_cache(maxsize=1)
def build_info() -> dict[str, object]:
    """Git identity of the code this *process* loaded, captured once at import.

    The API runs uvicorn without --reload, so the running code is frozen at
    startup. Reporting this (via /health and the /ask debug payload) makes
    file-vs-server drift impossible to miss: a heavy eval run can record exactly
    which commit produced it, and `git_dirty=True` flags that uncommitted code is
    serving traffic — the failure mode that silently invalidated an overnight run.
    """
    root = Path(__file__).resolve().parent.parent

    def _git(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *args],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return None

    status = _git("status", "--porcelain")
    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": (bool(status.strip()) if status is not None else None),
    }
