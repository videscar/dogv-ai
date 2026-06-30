"""DOGV "bis"-edition completeness — enumerate dispositions the date-sumario JSON drops.

When a DOGV date publishes both an ordinary and a *bis* (extraordinary) edition,
the date-keyed sumario JSON (`GET /dogv-portal/dogv?date=`) returns only ONE of
them, so the sibling edition's dispositions are never ingested and no gap is
recorded (the small bis issue looks complete). The portal exposes no JSON for the
sibling edition and no signatura->id endpoint, so the only complete enumeration is
the sibling's sumario PDF (one `resultat-dogv?signatura=` link per item); each
signatura is then resolved to its disposition id via the full-text search.

This module is the single source of truth for that enumeration/resolution. It is
used both by the live ingest pipeline (`ingest_pipeline` calls
`missing_sibling_disposicion_ids` per date so every ingest path captures both
editions) and by the one-off backfill script (`scripts/recover_bis_editions.py`).
"""
from __future__ import annotations

import io
import logging
import re

import requests
from pypdf import PdfReader
from sqlalchemy import text as sa_text

from .config import get_settings
from .db import SessionLocal

logger = logging.getLogger("dogv.bis_editions")
_BASE = get_settings().dogv_base_url.rstrip("/")

# Each sumario entry's text ends with its CVE marker ("...Generalitat.DOGV-C-2025-44601"),
# so the title precedes the marker and the marker carries the signatura (2025/44601).
_CVE_MARKER = re.compile(r"DOGV-[A-Z]*-?(\d{4})-(\d{3,6})")
_TIPO_WORD = re.compile(
    r"(?:DECRETO|DECRET|ORDEN|ORDRE|RESOLUCI[ÓO]N?|RESOLUCI[ÓO]|ACUERDO|ACORD|LEY|LLEI|"
    r"CORRECCI[ÓO]N|EXTRACTO|EXTRACTE|ANUNCIO|ANUNCI|EDICTO|EDICTE|INFORMACI[ÓO]N|"
    r"ADJUDICACI[ÓO]N|LICITACI[ÓO]N|NOTIFICACI[ÓO]N|CITACI[ÓO]N|SUBVENCI[ÓO]N|BASES|ACTO|ACTE)"
)
_LANG_SUFFIX = {"es_es": "es", "va_va": "va"}


def edition_pdf_urls(date_iso: str, numero: str, lang: str) -> tuple[str, str]:
    """(ordinary, bis) sumario PDF URLs for an issue."""
    y, m, d = date_iso.split("-")
    suf = _LANG_SUFFIX.get(lang, "es")
    stem = f"{_BASE}/datos/{y}/{m}/{d}/pdf/sumario_{y}_{numero}"
    return f"{stem}_{suf}.pdf", f"{stem}_bis_{suf}.pdf"


def fetch_pdf(url: str, *, timeout: int = 40) -> bytes | None:
    try:
        r = requests.get(url, timeout=timeout)
    except requests.RequestException as exc:
        logger.warning("bis.pdf_fetch_error url=%s err=%s", url, exc)
        return None
    if r.status_code != 200 or "pdf" not in r.headers.get("content-type", ""):
        return None
    return r.content


def parse_pdf_titles(content: bytes) -> dict[str, str]:
    """Map signatura -> full disposition title, parsed from the sumario PDF text.

    Splits on the per-entry CVE markers; each entry's title runs from its leading
    tipo word (DECRETO/ORDEN/RESOLUCIÓN/...) up to the marker that carries its sig."""
    reader = PdfReader(io.BytesIO(content))
    full = re.sub(r"[ \t]+", " ", "\n".join(p.extract_text() or "" for p in reader.pages))
    pairs: dict[str, str] = {}
    parts = re.split(r"(DOGV-[A-Z]*-?\d{4}-\d{3,6})", full)
    for i in range(1, len(parts), 2):
        mk = _CVE_MARKER.search(parts[i])
        if not mk:
            continue
        sig = f"{mk.group(1)}/{mk.group(2)}"
        pre = parts[i - 1].replace("\n", " ")
        tipos = list(_TIPO_WORD.finditer(pre))
        if not tipos:
            continue
        title = re.sub(r"\s+", " ", pre[tipos[-1].start():]).strip()
        if len(title) > 20:
            pairs[sig] = title
    return pairs


def search_dogv(texto: str, lang: str, *, timeout: int = 25) -> list[dict]:
    """Full-text search hits (each carries id + codigoInsercion).

    Gotchas, the hard way: punctuation (commas, the N/YYYY slash) makes the portal
    search 500 -> HTTP 440, so strip to word chars. The search is AND-semantics and
    verbatim, so a full long title (30+ words) trips one token mismatch -> 0 hits,
    while too few words (<=10) leaves 50+ undifferentiated hits past the date-sorted
    page. ~16 leading words is the sweet spot."""
    clean = re.sub(r"[^\wáéíóúüçñÁÉÍÓÚÑ ]", " ", texto)
    clean = " ".join(re.sub(r"\s+", " ", clean).split()[:16])
    try:
        r = requests.post(
            f"{_BASE}/dogv-portal/dogv/search",
            params={"lang": lang, "page": 0, "size": 50, "sort": "fecha,desc"},
            json={"texto": clean, "seccion": []},
            timeout=timeout,
        )
        if r.status_code != 200:
            return []
        return r.json().get("content") or []
    except requests.RequestException:
        return []


def resolve_signatura_ids(missing: set[str], titles: dict[str, str], lang: str) -> dict[str, int]:
    """signatura -> disposition id: search each target's own (distinctive) title and
    match on codigoInsercion. The exact match makes false positives impossible."""
    found: dict[str, int] = {}
    for sig in missing:
        title = titles.get(sig)
        if not title:
            continue
        for hit in search_dogv(title, lang):
            if hit.get("codigoInsercion") == sig and hit.get("id"):
                found[sig] = int(hit["id"])
                break
    return found


def existing_refs(refs: set[str], lang: str) -> set[str]:
    """Subset of `refs` already present as documents for `lang` (es/va twins share a
    ref, so the language join keeps them distinct)."""
    if not refs:
        return set()
    with SessionLocal() as db:
        rows = db.execute(
            sa_text(
                "SELECT d.ref FROM dogv_documents d JOIN dogv_issues i ON i.id=d.issue_id "
                "WHERE d.ref = ANY(:refs) AND i.language = :lang"
            ),
            {"refs": list(refs), "lang": lang},
        ).fetchall()
    return {r[0] for r in rows}


def enumerate_edition_titles(date_iso: str, numero: str, lang: str, *, both: bool = True) -> dict[str, str]:
    """signatura -> title across the issue's sumario PDF(s). `both` reads ordinary +
    bis (the backfill path); pass a single url via the sibling helper for the live path."""
    titles: dict[str, str] = {}
    for url in edition_pdf_urls(date_iso, numero, lang) if both else ():
        content = fetch_pdf(url)
        if content:
            titles.update(parse_pdf_titles(content))
    return titles


def missing_sibling_disposicion_ids(date_iso: str, numero: str, lang: str, is_bis: bool) -> dict[str, int]:
    """{signatura: disposition_id} for the SIBLING edition the date-sumario JSON did
    not return — i.e. the dropped edition's dispositions still absent from the corpus.

    `is_bis` is the ingested edition's esBis flag: if the JSON gave the bis, the
    sibling is the ordinary (the big miss); otherwise check for a bis (usually none).
    Returns {} for the common no-sibling date (sibling PDF 404) with one cheap fetch."""
    ordinary_url, bis_url = edition_pdf_urls(date_iso, numero, lang)
    sibling_url = ordinary_url if is_bis else bis_url
    content = fetch_pdf(sibling_url)
    if not content:
        return {}
    titles = parse_pdf_titles(content)
    sigs = set(titles)
    if not sigs:
        return {}
    missing = sigs - existing_refs(sigs, lang)
    if not missing:
        return {}
    resolved = resolve_signatura_ids(missing, titles, lang)
    unresolved = missing - set(resolved)
    if unresolved:
        logger.info(
            "bis.sibling %s/%s %s: sibling_pdf_sigs=%d missing=%d resolved=%d unresolved=%d",
            date_iso, numero, lang, len(sigs), len(missing), len(resolved), len(unresolved),
        )
    return resolved
