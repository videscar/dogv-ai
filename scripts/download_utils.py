from __future__ import annotations

from pathlib import Path

import requests


def build_dest_path(url_path: str, cache_root: Path) -> Path:
    path = Path(url_path.lstrip("/"))
    parts = path.parts
    if len(parts) >= 4:
        return cache_root / parts[0] / parts[1] / parts[2] / parts[-1]
    return cache_root / path


def download_to_path(full_url: str, dest: Path, chunk_size: int) -> bool:
    if dest.exists():
        print(f"[skip] exists: {dest}")
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = None
    try:
        resp = requests.get(full_url, timeout=60, stream=True)
        resp.raise_for_status()
        total = 0
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                fh.write(chunk)
                total += len(chunk)
        print(f"[ok] {dest} ({total} bytes)")
    except Exception as exc:
        print(f"[error] {full_url}: {exc}")
        return False
    finally:
        if resp is not None:
            resp.close()
    return True
