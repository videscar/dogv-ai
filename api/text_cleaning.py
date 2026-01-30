from __future__ import annotations

import re
from bs4 import BeautifulSoup


_DROP_TAGS = {
    "script",
    "style",
    "noscript",
    "header",
    "footer",
    "nav",
    "aside",
    "svg",
}


def _normalize_lines(text: str) -> str:
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if len(line) < 3:
            continue
        lines.append(line)
    return "\n".join(lines)


def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in list(soup.find_all(_DROP_TAGS)):
        tag.decompose()

    for tag in soup.find_all(True):
        classes = " ".join(tag.get("class", [])).lower()
        element_id = (tag.get("id") or "").lower()
        if any(key in classes for key in ("menu", "header", "footer", "nav")):
            tag.decompose()
            continue
        if any(key in element_id for key in ("menu", "header", "footer", "nav")):
            tag.decompose()

    text = soup.get_text(separator="\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = _normalize_lines(text)
    return text.strip()
