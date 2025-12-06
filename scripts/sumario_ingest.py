"""
Fetch and store DOGV 'sumario' (issue metadata) into SQLite.

Usage:
    python scripts/sumario_ingest.py YYYY-MM-DD [lang]

Example:
    python scripts/sumario_ingest.py 2024-01-15 es_es
"""

import sys
from datetime import datetime
from typing import Any, Dict

import requests
from sqlalchemy.orm import Session

try:
    from . import _path  # type: ignore  # ensures project root is on sys.path
except ImportError:
    import _path  # type: ignore  # noqa: F401

from api.config import get_settings
from api.db import SessionLocal
from api.models import DogvIssue

settings = get_settings()


def _pick_first(source: dict, keys: list[str]) -> tuple[Any | None, str | None]:
    if not isinstance(source, dict):
        return None, None
    for key in keys:
        if key in source:
            return source[key], key
    return None, None


def upsert_issue(db: Session, date_str: str, lang: str, data: Dict[str, Any]) -> DogvIssue:
    """
    Insert or update one DOGV issue. Tries multiple key names for numero/title and logs what it finds.
    """
    date = datetime.strptime(date_str, "%Y-%m-%d").date()
    cabecera = data.get("cabecera") or {}

    numero, numero_key = _pick_first(cabecera, ["numero", "numeroDogv", "numeroDOGV"])
    if numero is None:
        numero, numero_key = _pick_first(data, ["numero", "numeroDogv", "numeroDOGV"])

    title, title_key = _pick_first(cabecera, ["titulo", "titol", "title"])
    if title is None:
        # Fallback: take title from first disposition if present
        dispos = data.get("disposiciones") or []
        first_disp = dispos[0] if dispos else {}
        title, title_key = _pick_first(first_disp, ["titulo", "titol", "title"])
        if title is not None:
            title_key = f"disposicion[0].{title_key}"

    print(f"[upsert_issue] numero source: {numero_key or 'missing'}, value: {numero!r}")
    print(f"[upsert_issue] title source: {title_key or 'missing'}, value: {title!r}")

    issue = (
        db.query(DogvIssue)
        .filter(DogvIssue.date == date, DogvIssue.language == lang)
        .one_or_none()
    )

    if issue is None:
        issue = DogvIssue(
            date=date,
            language=lang,
            numero=numero,
            title=title,
            raw_json=data,
        )
        db.add(issue)
    else:
        issue.numero = numero
        issue.title = title
        issue.raw_json = data

    db.commit()
    db.refresh(issue)
    return issue


def fetch_sumario_json(date: str, lang: str = "es_es") -> Dict[str, Any]:
    base = settings.dogv_base_url.rstrip("/")
    url = f"{base}/dogv-portal/dogv"
    params = {"date": date, "lang": lang}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/sumario_ingest.py YYYY-MM-DD [lang]")
        raise SystemExit(1)

    date_str = sys.argv[1]
    lang = sys.argv[2] if len(sys.argv) > 2 else "es_es"

    print(f"Fetching DOGV issue for {date_str} lang={lang} ...")
    data = fetch_sumario_json(date_str, lang)

    db = SessionLocal()
    try:
        issue = upsert_issue(db, date_str, lang, data)
        print(f"Stored DOGV issue id={issue.id}, numero={issue.numero}, title={issue.title}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
