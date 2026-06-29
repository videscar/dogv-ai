"""Recover dispositions dropped on DOGV "bis"-collision dates.

When a date publishes both an ordinary and a *bis* (extraordinary) edition, the
portal's date-keyed sumario JSON (`GET /dogv-portal/dogv?date=`) returns ONLY the
bis edition, so the ordinary edition's ~50 dispositions are never ingested and no
gap is recorded (the pipeline thinks the small bis issue is complete).

This script recovers the missing edition:
  1. Enumerate every disposition signatura from BOTH edition sumario PDFs
     (the PDFs reliably carry one `resultat-dogv?signatura=` link per item).
  2. For signaturas absent from the corpus, resolve signatura -> disposition id
     via the full-text search (matching `codigoInsercion`), then ingest each via
     the normal on-demand path (ingest_one_disposicion: create row, extract body,
     classify, chunk+embed). Idempotent and resumable.

Usage:
    python scripts/recover_bis_editions.py --date 2025-10-31           # one date
    python scripts/recover_bis_editions.py --date 2025-10-31 --dry-run # report only
    python scripts/recover_bis_editions.py --all                       # full backfill
    python scripts/recover_bis_editions.py --all --dry-run
"""
from __future__ import annotations

import argparse
import io
import logging
import re
import sys
import time
from typing import Iterable

import requests
from pypdf import PdfReader
from sqlalchemy import text as sa_text

sys.path.insert(0, ".")

from api.auto_ingest import ingest_one_disposicion
from api.config import get_settings
from api.db import SessionLocal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("recover_bis")
settings = get_settings()
BASE = settings.dogv_base_url.rstrip("/")

# Each sumario entry's text ends with its CVE marker (e.g. "...Generalitat.DOGV-C-2025-44601"),
# so the title text immediately precedes the marker and the marker carries the signatura.
_CVE_MARKER = re.compile(r"DOGV-[A-Z]*-?(\d{4})-(\d{3,6})")
_TIPO_WORD = re.compile(
    r"(?:DECRETO|DECRET|ORDEN|ORDRE|RESOLUCI[ÓO]N?|RESOLUCI[ÓO]|ACUERDO|ACORD|LEY|LLEI|"
    r"CORRECCI[ÓO]N|EXTRACTO|EXTRACTE|ANUNCIO|ANUNCI|EDICTO|EDICTE|INFORMACI[ÓO]N|"
    r"ADJUDICACI[ÓO]N|LICITACI[ÓO]N|NOTIFICACI[ÓO]N|CITACI[ÓO]N|SUBVENCI[ÓO]N|BASES|ACTO|ACTE)"
)
_LANG_SUFFIX = {"es_es": "es", "va_va": "va"}


def _pdf_urls(date_iso: str, numero: str, lang: str) -> list[str]:
    """Both edition sumario PDF URLs (ordinary + bis) for an issue."""
    y, m, d = date_iso.split("-")
    suf = _LANG_SUFFIX[lang]
    stem = f"{BASE}/datos/{y}/{m}/{d}/pdf/sumario_{y}_{numero}"
    return [f"{stem}_{suf}.pdf", f"{stem}_bis_{suf}.pdf"]


def _fetch_pdf(url: str) -> bytes | None:
    try:
        r = requests.get(url, timeout=40)
    except requests.RequestException as exc:
        logger.warning("pdf.fetch_error url=%s err=%s", url, exc)
        return None
    if r.status_code != 200 or "pdf" not in r.headers.get("content-type", ""):
        return None
    return r.content


def _parse_pdf(content: bytes) -> dict[str, str]:
    """Map signatura -> full disposition title, parsed from the sumario text.

    Splits the text on the per-entry CVE markers; each entry's title is the run
    from its last leading tipo word (DECRETO/ORDEN/RESOLUCIÓN/...) up to the marker."""
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


def _search(texto: str, lang: str) -> list[dict]:
    # Punctuation (commas, the N/YYYY slash) makes the portal search 500 -> HTTP 440;
    # strip to word chars. The search is AND-semantics: every token must appear
    # verbatim, so a full long title (30+ words) reliably trips one mismatch and
    # returns 0 hits, while too few words (<=10) leaves 50+ undifferentiated hits
    # past the date-sorted page. ~16 leading words is the sweet spot — enough to
    # narrow to a handful, short enough to all match.
    clean = re.sub(r"[^\wáéíóúüçñÁÉÍÓÚÑ ]", " ", texto)
    clean = " ".join(re.sub(r"\s+", " ", clean).split()[:16])
    try:
        r = requests.post(
            f"{BASE}/dogv-portal/dogv/search",
            params={"lang": lang, "page": 0, "size": 50, "sort": "fecha,desc"},
            json={"texto": clean, "seccion": []},
            timeout=25,
        )
        if r.status_code != 200:
            return []
        return r.json().get("content") or []
    except requests.RequestException:
        return []


def _resolve_ids(missing: set[str], titles: dict[str, str], lang: str) -> dict[str, int]:
    """signatura -> disposition id: search each target's own (distinctive) title and
    match on codigoInsercion. The exact-match makes false positives impossible."""
    found: dict[str, int] = {}
    for sig in missing:
        title = titles.get(sig)
        if not title:
            continue
        for hit in _search(title, lang):
            if hit.get("codigoInsercion") == sig and hit.get("id"):
                found[sig] = int(hit["id"])
                break
    return found


def _existing_refs(refs: set[str], lang: str) -> set[str]:
    if not refs:
        return set()
    short = lang.split("_")[0]  # es / va — distinguish es/va twins sharing a ref
    with SessionLocal() as db:
        rows = db.execute(
            sa_text(
                "SELECT d.ref FROM dogv_documents d JOIN dogv_issues i ON i.id=d.issue_id "
                "WHERE d.ref = ANY(:refs) AND i.language = :lang"
            ),
            {"refs": list(refs), "lang": lang},
        ).fetchall()
    return {r[0] for r in rows}


def recover_date(date_iso: str, numero: str, lang: str, dry_run: bool) -> dict:
    titles: dict[str, str] = {}
    for url in _pdf_urls(date_iso, numero, lang):
        content = _fetch_pdf(url)
        if content is None:
            continue
        titles.update(_parse_pdf(content))
    target_sigs = set(titles)

    have = _existing_refs(target_sigs, lang)
    missing = target_sigs - have
    stats = {
        "date": date_iso, "numero": numero, "lang": lang,
        "in_pdfs": len(target_sigs), "already": len(have), "missing": len(missing),
        "resolved": 0, "ingested": 0, "unresolved": [], "failed": [],
    }
    if not missing:
        logger.info("recover %s/%s %s: complete (pdf=%d already=%d)",
                    date_iso, numero, lang, len(target_sigs), len(have))
        return stats

    resolved = _resolve_ids(missing, titles, lang)
    stats["resolved"] = len(resolved)
    stats["unresolved"] = sorted(missing - set(resolved))
    logger.info("recover %s/%s %s: pdf=%d already=%d missing=%d resolved=%d unresolved=%d",
                date_iso, numero, lang, len(target_sigs), len(have),
                len(missing), len(resolved), len(stats["unresolved"]))
    if dry_run:
        return stats

    for sig, disp_id in resolved.items():
        try:
            doc_id = ingest_one_disposicion(disp_id, lang)
            if doc_id:
                stats["ingested"] += 1
            else:
                stats["failed"].append(sig)
        except Exception:
            logger.exception("ingest_fail sig=%s disp_id=%s lang=%s", sig, disp_id, lang)
            stats["failed"].append(sig)
    logger.info("recover %s/%s %s: ingested=%d failed=%d",
                date_iso, numero, lang, stats["ingested"], len(stats["failed"]))
    return stats


def _bis_issues() -> list[tuple[str, str, str]]:
    """(date_iso, numero, language) for every bis-truncated issue in the window."""
    with SessionLocal() as db:
        rows = db.execute(
            sa_text(
                "SELECT i.date::text, i.numero, i.language FROM dogv_issues i "
                "WHERE (i.raw_json->>'esBis')='true' AND i.numero IS NOT NULL "
                "AND i.date >= current_date - interval '24 months' "
                "ORDER BY i.date, i.language"
            )
        ).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="single issue date YYYY-MM-DD")
    ap.add_argument("--lang", default=None, help="es_es / va_va (default: both)")
    ap.add_argument("--all", action="store_true", help="all bis-truncated issues in window")
    ap.add_argument("--workers", type=int, default=1, help="parallel issues (--all path)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.all:
        targets = _bis_issues()
    elif args.date:
        langs = [args.lang] if args.lang else ["es_es", "va_va"]
        with SessionLocal() as db:
            targets = []
            for lang in langs:
                row = db.execute(
                    sa_text("SELECT numero FROM dogv_issues WHERE date=:d AND language=:l "
                            "AND numero IS NOT NULL"),
                    {"d": args.date, "l": lang},
                ).first()
                if row:
                    targets.append((args.date, row[0], lang))
    else:
        ap.error("pass --date or --all")

    logger.info("recover start: %d issue(s) workers=%d dry_run=%s",
                len(targets), args.workers, args.dry_run)
    totals = {"in_pdfs": 0, "already": 0, "missing": 0, "resolved": 0, "ingested": 0,
              "unresolved": 0, "failed": 0}
    t0 = time.time()

    def _one(target):
        date_iso, numero, lang = target
        try:
            return recover_date(date_iso, numero, lang, args.dry_run)
        except Exception:
            logger.exception("recover_issue_error %s/%s %s", date_iso, numero, lang)
            return None

    def _acc(s):
        if not s:
            return
        for k in ("in_pdfs", "already", "missing", "resolved", "ingested"):
            totals[k] += s[k]
        totals["unresolved"] += len(s["unresolved"])
        totals["failed"] += len(s["failed"])

    if args.workers > 1:
        # Issue-level parallelism: distinct dates don't collide (extract/classify/
        # chunk are scoped per date / per doc-id). Bounded to spare the embed+LLM.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for s in ex.map(_one, targets):
                _acc(s)
    else:
        for target in targets:
            _acc(_one(target))
    logger.info("recover DONE in %.0fs totals=%s", time.time() - t0, totals)


if __name__ == "__main__":
    main()
