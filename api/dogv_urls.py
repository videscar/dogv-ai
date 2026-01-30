from __future__ import annotations

from .config import get_settings

settings = get_settings()


def _build_url(path: str | None) -> str | None:
    if not path:
        return None
    if path.startswith("http://") or path.startswith("https://"):
        return path
    base = settings.dogv_base_url.rstrip("/")
    return f"{base}/datos/{path.lstrip('/')}"


def build_pdf_url(pdf_url: str | None) -> str | None:
    return _build_url(pdf_url)


def build_html_url(html_url: str | None) -> str | None:
    return _build_url(html_url)
