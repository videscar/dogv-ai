from __future__ import annotations

import re

from .temporal import has_relative_time_expression
from .text_tokens import VALENCIAN_MARKERS

# A normative citation by number/year ("Decret 74/2026", "Ordre 4/2025") or a
# norm-word anchored to a month ("la resolució d'abril"). When a query hinges on
# such a specific reference, a HyDE hypothetical drifts off the exact disposition
# and can evict the correctly-retrieved gold from the fused pool (e.g. v2-092):
# is_reference_query() gates HyDE off for these. Validated on eval_v2: fires on
# 9/10 wrong_ref, 0/16 vague, and never on a query HyDE is known to recover.
_NORM_WORD = (
    r"(?:decret|decreto|ordre|orden|resoluci[oó]n?|resoluci[oó]|acord|acuerdo|"
    r"llei|ley|ordenan[cz]a|reglament(?:o)?|conveni|convenio)"
)
_MONTH = (
    r"(?:gener|febrer|mar[çc]|abril|maig|juny|juliol|agost|setembre|octubre|"
    r"novembre|desembre|enero|febrero|marzo|mayo|junio|julio|agosto|septiembre|"
    r"noviembre|diciembre)"
)
_REFERENCE_NUMBER_RE = re.compile(r"\b\d{1,4}/\d{2,4}\b")
_NORM_MONTH_RE = re.compile(
    _NORM_WORD + r"\b[^.?!\n]{0,40}?\bd[e']?\s*" + _MONTH + r"\b", re.IGNORECASE
)


def is_reference_query(text: str) -> bool:
    """True when the query cites a specific norm (number/year or norm-word + month)."""
    if not text:
        return False
    return bool(_REFERENCE_NUMBER_RE.search(text) or _NORM_MONTH_RE.search(text))


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
