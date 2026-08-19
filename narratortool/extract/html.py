"""HTML to narratable text.

Shared by the .html extractor and the EPUB one, since EPUB chapters are XHTML.
Uses BeautifulSoup when available and falls back to a regex stripper otherwise, so
plain HTML never becomes a hard dependency failure.
"""
from __future__ import annotations

import html as html_module
import re
from pathlib import Path

from ..document import Chapter, Document

# Elements whose text should never be spoken.
_DROP = ("script", "style", "head", "nav", "footer", "aside", "figure", "table")
# Elements that should force a line break so sentences do not run together.
_BLOCK = ("p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "blockquote")


def extract_html(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8", errors="replace")
    return Document(
        title=first_heading(raw) or _title_tag(raw),
        chapters=[Chapter(text=html_to_text(raw))],
    )


def html_to_text(raw: str) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return _regex_to_text(raw)

    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(list(_DROP)):
        tag.decompose()
    text = soup.get_text("\n")
    return _collapse(text)


def first_heading(raw: str) -> str | None:
    match = re.search(r"<h[1-3][^>]*>(.*?)</h[1-3]>", raw, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return _collapse(_regex_to_text(match.group(1))) or None


def _title_tag(raw: str) -> str | None:
    match = re.search(r"<title[^>]*>(.*?)</title>", raw, re.IGNORECASE | re.DOTALL)
    return html_module.unescape(match.group(1)).strip() if match else None


def _regex_to_text(raw: str) -> str:
    for tag in _DROP:
        raw = re.sub(rf"<{tag}\b.*?</{tag}>", " ", raw, flags=re.IGNORECASE | re.DOTALL)
    raw = re.sub(rf"<\s*/?\s*(?:{'|'.join(_BLOCK)})\b[^>]*>", "\n", raw, flags=re.IGNORECASE)
    raw = re.sub(r"<[^>]+>", " ", raw)
    return _collapse(html_module.unescape(raw))


def _collapse(text: str) -> str:
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
