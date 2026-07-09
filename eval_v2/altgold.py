"""Alt-gold citation scoring.

The single-gold `gold_cited` proxy understates quality on vague questions, which
legitimately have several acceptable answers (this session: read-judged ~5/7 correct
vs proxy 2/7). These helpers score a citation set against:

  - gold_sets : list of OR-groups (each group is satisfied by citing ANY member);
                "full" coverage requires every group to be hit (matters for multihop).
  - accept    : a wider set of acceptable doc-ids (gold + valid siblings, e.g. the
                es/va twin of a disposition). "any" hit = at least one accepted cite.

Pure functions, no I/O, so they can back both the multi-turn runner and unit tests.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _ints(values: Iterable[Any]) -> set[int]:
    out: set[int] = set()
    for v in values or []:
        try:
            out.add(int(v))
        except (TypeError, ValueError):
            continue
    return out


def accept_set(scenario: dict[str, Any]) -> set[int]:
    """Acceptable doc-ids = explicit accept_doc_ids ∪ every gold id."""
    accept = _ints(scenario.get("accept_doc_ids"))
    accept |= _ints(scenario.get("gold_doc_ids"))
    for group in scenario.get("gold_sets") or []:
        accept |= _ints(group)
    return accept


def citation_any_hit(cited: Iterable[Any], accept: Iterable[int]) -> bool:
    """True if at least one cited doc is acceptable."""
    return bool(_ints(cited) & set(accept))


def citation_full_hit(cited: Iterable[Any], gold_sets: list[list[Any]] | None) -> bool:
    """True if every OR-group has at least one of its members cited."""
    if not gold_sets:
        return False
    cited_set = _ints(cited)
    return all(bool(_ints(group) & cited_set) for group in gold_sets)
