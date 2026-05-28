from __future__ import annotations

from .temporal import has_relative_time_expression
from .text_tokens import VALENCIAN_MARKERS


def is_feed_query(text: str) -> bool:
    return is_relative_time_query(text)


def is_relative_time_query(text: str) -> bool:
    return has_relative_time_expression(text or "")


def guess_language(text: str) -> str:
    if not text:
        return "es_es"
    lower = text.lower()
    if "·" in lower or "ç" in lower:
        return "va_va"
    if "l'" in lower:
        return "va_va"
    hits = sum(1 for marker in VALENCIAN_MARKERS if marker in lower)
    if hits >= 2:
        return "va_va"
    return "es_es"
