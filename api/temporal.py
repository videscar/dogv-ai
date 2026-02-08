from __future__ import annotations

from datetime import date, datetime, timedelta
import re
import unicodedata
from zoneinfo import ZoneInfo


_EXPLICIT_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_LAST_N_DAYS_RE = re.compile(
    r"\b(?:ultim(?:os|as)?|ultims?|darrers?)\s+(\d{1,3})\s+(?:dias|dies)\b"
)
_TODAY_RE = re.compile(r"\b(?:hoy|hui|avui)\b")
_YESTERDAY_RE = re.compile(r"\b(?:ayer|ahir)\b")
_TOMORROW_RE = re.compile(r"\b(?:manana|dema)\b")
_THIS_WEEK_RE = re.compile(r"\b(?:esta semana|esta setmana|aquesta setmana)\b")
_LAST_WEEK_RE = re.compile(
    r"\b(?:la semana pasada|semana pasada|semana anterior|setmana passada|setmana anterior|darrera setmana)\b"
)
_THIS_MONTH_RE = re.compile(r"\b(?:este mes|aquest mes)\b")
_LAST_MONTH_RE = re.compile(
    r"\b(?:mes pasado|mes anterior|mes passat|el mes anterior|el mes passat)\b"
)
_THIS_YEAR_RE = re.compile(r"\b(?:este ano|aquest any)\b")
_LAST_YEAR_RE = re.compile(r"\b(?:ano pasado|ano anterior|any passat|any anterior)\b")
_GENERIC_RECENT_RE = re.compile(
    r"\b(?:ultimos dias|darrers dies|novedades|novetats|ultima semana|ultimes setmanes)\b"
)


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def _normalize(text: str) -> str:
    lowered = _strip_accents(text.lower())
    return re.sub(r"\s+", " ", lowered).strip()


def _safe_zoneinfo(name: str | None) -> ZoneInfo:
    tz_name = (name or "").strip() or "Europe/Madrid"
    try:
        return ZoneInfo(tz_name)
    except Exception:
        return ZoneInfo("Europe/Madrid")


def local_today(timezone_name: str | None = None) -> date:
    tz = _safe_zoneinfo(timezone_name)
    return datetime.now(tz).date()


def _parse_week_start(value: str | int | None) -> int:
    if isinstance(value, int):
        if 0 <= value <= 6:
            return value
        return 0
    if isinstance(value, str):
        token = _normalize(value)
        if token.isdigit():
            numeric = int(token)
            if 0 <= numeric <= 6:
                return numeric
        mapping = {
            "monday": 0,
            "lunes": 0,
            "dilluns": 0,
            "tuesday": 1,
            "martes": 1,
            "dimarts": 1,
            "wednesday": 2,
            "miercoles": 2,
            "dimecres": 2,
            "thursday": 3,
            "jueves": 3,
            "dijous": 3,
            "friday": 4,
            "viernes": 4,
            "divendres": 4,
            "saturday": 5,
            "sabado": 5,
            "dissabte": 5,
            "sunday": 6,
            "domingo": 6,
            "diumenge": 6,
        }
        return mapping.get(token, 0)
    return 0


def _month_start(day: date) -> date:
    return day.replace(day=1)


def _month_end(day: date) -> date:
    if day.month == 12:
        next_month = date(day.year + 1, 1, 1)
    else:
        next_month = date(day.year, day.month + 1, 1)
    return next_month - timedelta(days=1)


def _week_start(day: date, week_start: int) -> date:
    delta = (day.weekday() - week_start) % 7
    return day - timedelta(days=delta)


def has_relative_time_expression(text: str) -> bool:
    if not text:
        return False
    if _EXPLICIT_DATE_RE.search(text):
        return False
    normalized = _normalize(text)
    if _LAST_N_DAYS_RE.search(normalized):
        return True
    patterns = (
        _TODAY_RE,
        _YESTERDAY_RE,
        _TOMORROW_RE,
        _THIS_WEEK_RE,
        _LAST_WEEK_RE,
        _THIS_MONTH_RE,
        _LAST_MONTH_RE,
        _THIS_YEAR_RE,
        _LAST_YEAR_RE,
        _GENERIC_RECENT_RE,
    )
    return any(pattern.search(normalized) for pattern in patterns)


def resolve_relative_date_range(
    text: str,
    timezone_name: str | None = None,
    week_start: str | int | None = "monday",
    today: date | None = None,
) -> tuple[date, date] | None:
    if not text:
        return None
    if _EXPLICIT_DATE_RE.search(text):
        return None

    normalized = _normalize(text)
    anchor = today or local_today(timezone_name)
    week_start_idx = _parse_week_start(week_start)

    match_n_days = _LAST_N_DAYS_RE.search(normalized)
    if match_n_days:
        days = max(1, min(int(match_n_days.group(1)), 180))
        start = anchor - timedelta(days=days - 1)
        return (start, anchor)

    if _TODAY_RE.search(normalized):
        return (anchor, anchor)
    if _YESTERDAY_RE.search(normalized):
        point = anchor - timedelta(days=1)
        return (point, point)
    if _TOMORROW_RE.search(normalized):
        point = anchor + timedelta(days=1)
        return (point, point)

    if _THIS_WEEK_RE.search(normalized):
        start = _week_start(anchor, week_start_idx)
        return (start, anchor)
    if _LAST_WEEK_RE.search(normalized):
        this_week_start = _week_start(anchor, week_start_idx)
        start = this_week_start - timedelta(days=7)
        end = this_week_start - timedelta(days=1)
        return (start, end)

    if _THIS_MONTH_RE.search(normalized):
        start = _month_start(anchor)
        return (start, anchor)
    if _LAST_MONTH_RE.search(normalized):
        last_month_end = _month_start(anchor) - timedelta(days=1)
        return (_month_start(last_month_end), _month_end(last_month_end))

    if _THIS_YEAR_RE.search(normalized):
        return (date(anchor.year, 1, 1), anchor)
    if _LAST_YEAR_RE.search(normalized):
        year = anchor.year - 1
        return (date(year, 1, 1), date(year, 12, 31))

    if _GENERIC_RECENT_RE.search(normalized):
        start = anchor - timedelta(days=6)
        return (start, anchor)

    return None
