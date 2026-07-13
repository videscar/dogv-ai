"""Single-document anchoring for multi-field grant queries.

A question like "d'aquella beca de Psicologia: requisits, retribució mensual,
hores, durada, import global, termini, BDNS..." asks many structured FIELDS of
ONE document (a grant/scholarship extract). The generic pipeline treats the
field-request clause as retrieval signal, which fails two ways:

1. Retrieval: the field vocabulary (import, hores, seguretat social...) swamps
   the document identity ("beca de Psicologia, màster en Cervell i Conducta,
   UJI") — the whole-question embedding ranks the target below documents that
   merely share field vocabulary (a Social-Security law outranked the gold
   extract), and the facet/second-hop sub-queries built from the field clause
   retrieve per-field noise docs that then get pinned into the read set.
2. Evidence/synthesis: per-field chunk selectors pick the best chunk PER
   KEYWORD across every read doc, so near-identical sibling convocatòries
   cross-contaminate (the answer blends €440/month from one beca with the
   total of another).

This module provides the query-side primitives: detect a multi-field grant
query, list the asked fields, and strip the field-request clauses to get an
IDENTITY query that retrieves by document identity only. Pipeline stages use
these to anchor retrieval, reading and synthesis on one resolved document.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = [
    "asked_fields",
    "field_cue_terms",
    "identity_query",
    "is_multi_field_grant_query",
]


def _fold(text: str) -> str:
    return "".join(
        ch
        for ch in unicodedata.normalize("NFKD", (text or "").lower())
        if not unicodedata.combining(ch)
    )


# One entry per structured field a grant extract states. Patterns are matched on
# the accent-folded, lowercased question. Kept tight on purpose: a field counts
# only when the question plausibly ASKS for it, not merely mentions the topic.
_FIELD_RES: dict[str, re.Pattern[str]] = {
    "import": re.compile(
        r"\bimport\w*|\bquantia|\bcuantia|\bretribuci\w+|\bsou\b|\bsueldo\b|"
        r"\bcobra\w*|\bdiners\b|\bdinero\b|\bcuanto\s+se\s+paga|\beuros?\b"
    ),
    "hores": re.compile(r"\bhores\b|\bhoras\b|\bdedicacio\w*"),
    "durada": re.compile(r"\bdurada\b|\bduracion?\b|\bquant\s+dura\b|\bcuanto\s+dura\b"),
    "termini": re.compile(
        r"\btermini\w*|\bplazos?\b|\bquan\s+acaba\b|\bcuando\s+acaba\b|"
        r"\bdata\s+limit\w*|\bfecha\s+limite\b"
    ),
    "requisits": re.compile(
        r"\brequisits?\b|\brequisitos?\b|\bbeneficiari\w*|\bbeneficiario\w*|"
        r"\bdestinatari\w*|\bqui\s+(?:hi\s+)?pot\s+(?:optar|demanar|sol|presentar)|"
        r"\bquien(?:es)?\s+puede"
    ),
    "bases": re.compile(
        r"\bbases\s+regulador\w+|\bon\s+son\s+les\s+bases\b|\bdonde\s+estan\s+las\s+bases\b"
    ),
    "seguretat_social": re.compile(r"\bseguretat\s+social\b|\bseguridad\s+social\b"),
    "bdns": re.compile(r"\bbdns\b"),
    "finalitat": re.compile(r"\bfinalitat\b|\bfinalidad\b"),
    "departament": re.compile(
        r"\b(?:a\s+)?quin\s+departament\b|\bque\s+departamento\b|"
        r"\bdepartament\s+o\s+institut\b|\bdepartamento\s+o\s+instituto\b"
    ),
    "tramitar": re.compile(
        r"\btramitar\b|\bon\s+(?:es\s+)?(?:tramita|presenta)\b|\bdonde\s+se\s+(?:tramita|presenta)\b"
    ),
}

_GRANT_NOUN_RE = re.compile(
    r"\bbec(?:a|ques)\b|\bajud(?:a|es)\b|\bayudas?\b|\bsubvenci\w+|\bconvocatori\w+"
)

# Ask markers that open a field-request clause ("dis-me qui hi pot optar...",
# "quant es cobra...", "cuánto dura..."). Used only to decide whether a clause
# is a field REQUEST (droppable from the identity query) vs identity content.
_ASK_MARKER_RE = re.compile(
    r"\bdi(?:s|gues)[-\s]?me\b|\bdime\b|\bindica\w*\b|\bquant\w*\b|\bcuant\w*\b|"
    r"\bquan\b|\bcuando\b|\bqui\b|\bquien\w*\b|\bon\b|\bdonde\b|\bquins?\b|"
    r"\bquines?\b|\bque\s+requisits\b|\bcuales?\b"
)

# Numbered field-list items ("1) requisits, 2) finalitat, ... 10) codi BDNS"):
# pure field enumeration, never identity signal.
_NUMBERED_ITEM_RE = re.compile(r"\b\d{1,2}\)\s*[^,.;:?()]*")

_CLAUSE_SPLIT_RE = re.compile(r"[.?!;:]")

# Windowing cue terms per field (accent-folded substrings of extract section
# text: "Tercer. Característiques... 20 hores setmanals... Cinqué. Import...").
# Fed into the reader's salient-keyword set so the chunk window lands on the
# section that states the asked field instead of only the chunk head.
_FIELD_CUES: dict[str, list[str]] = {
    "import": ["import", "retribucio", "euros"],
    "hores": ["hores setmanals", "horas semanales", "dedicacio"],
    "durada": ["durada", "duracion", "mesos", "meses"],
    "termini": ["termini", "plazo", "licituds", "licitudes"],
    "requisits": ["beneficiari", "requisit"],
    "bases": ["bases reguladores", "bases reguladoras"],
    "seguretat_social": ["seguretat social", "seguridad social"],
    "bdns": ["bdns"],
    "finalitat": ["finalitat", "finalidad"],
    "departament": ["departament", "departamento", "institut"],
    "tramitar": ["licituds", "licitudes", "presentacio", "presentacion"],
}


def asked_fields(question: str) -> set[str]:
    """Distinct structured fields the question asks for."""
    folded = _fold(question)
    if not folded:
        return set()
    return {name for name, pattern in _FIELD_RES.items() if pattern.search(folded)}


# Cue priority when the budget truncates: the amount/deadline sections are the
# ones the prefix-biased window historically cut (they sit at the extract tail).
_FIELD_PRIORITY = [
    "import",
    "termini",
    "hores",
    "durada",
    "bdns",
    "seguretat_social",
    "departament",
    "requisits",
    "bases",
    "finalitat",
    "tramitar",
]


def field_cue_terms(fields: set[str], limit: int = 12) -> list[str]:
    """Accent-folded chunk-window cue substrings for the asked fields."""
    out: list[str] = []
    ordered = [f for f in _FIELD_PRIORITY if f in fields]
    ordered += sorted(fields - set(_FIELD_PRIORITY))
    for name in ordered:
        for cue in _FIELD_CUES.get(name, []):
            if cue not in out:
                out.append(cue)
            if len(out) >= limit:
                return out
    return out


def is_multi_field_grant_query(question: str) -> bool:
    """True for a question asking >=2 structured fields of a grant/convocatòria.

    Deliberately conservative: needs a grant noun, two distinct field kinds, and
    neither an enumeration ("list every...") nor a multi-reference comparison —
    those query classes must keep their existing pipelines.
    """
    folded = _fold(question)
    if not folded or not _GRANT_NOUN_RE.search(folded):
        return False
    if len(asked_fields(question)) < 2:
        return False
    # Lazy imports: cheap-regex module used from low-level pipeline stages.
    from api.dogv_resolver import parse_references
    from api.enumeration import is_enumeration_query

    if is_enumeration_query(question):
        return False
    if len(parse_references(question)) >= 2:
        return False
    return True


def _is_field_request_clause(clause: str) -> bool:
    folded = _fold(clause)
    if not folded.strip():
        return True
    hits = sum(1 for pattern in _FIELD_RES.values() if pattern.search(folded))
    if hits == 0:
        return False
    if _GRANT_NOUN_RE.search(folded) and hits < 2:
        # "hi ha una beca ... termini?" — identity and a single field in one
        # clause: keep it, the identity content outweighs one field mention.
        return False
    return bool(_ASK_MARKER_RE.search(folded)) or hits >= 2


def identity_query(question: str) -> str:
    """The question minus its field-request clauses — what identifies the doc.

    "Sóc estudianta del màster en Cervell i Conducta a la UJI. Digues-me
    d'aquella beca de Psicologia: 1) requisits, ... 10) codi BDNS."
    -> "Sóc estudianta del màster en Cervell i Conducta a la UJI Digues-me
    d'aquella beca de Psicologia"

    Returns "" when nothing identity-like survives (caller skips anchoring).
    """
    text = _NUMBERED_ITEM_RE.sub(" ", question or "")
    kept: list[str] = []
    for clause in _CLAUSE_SPLIT_RE.split(text):
        clause = clause.strip(" ,¿¡«»\"'")
        if not clause:
            continue
        if _is_field_request_clause(clause):
            continue
        kept.append(clause)
    identity = " ".join(kept).strip()
    if len(identity) < 15:
        return ""
    return identity
