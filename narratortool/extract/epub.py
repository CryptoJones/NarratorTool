"""EPUB extraction.

Reads the spine in reading order, which is the only ordering that reliably matches
how the book is meant to be consumed. Chapter titles come from the nav/TOC where it
maps cleanly onto spine items, falling back to the document's own <h1>-<h3>.
"""
from __future__ import annotations

from pathlib import Path

from ..document import Chapter, Document
from .html import html_to_text, first_heading


def extract_epub(path: Path) -> Document:
    try:
        import ebooklib
        from ebooklib import epub
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "EPUB support needs `ebooklib` (pip install 'narratortool[epub]')"
        ) from exc

    book = epub.read_epub(str(path))

    doc = Document(
        title=_first_meta(book, "title"),
        author=_first_meta(book, "creator"),
    )

    toc_titles = _toc_titles(book)

    for item_id, _linear in book.spine:
        item = book.get_item_with_id(item_id)
        if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue

        raw = item.get_content().decode("utf-8", errors="replace")
        text = html_to_text(raw)
        if not text.strip():
            continue  # cover pages, nav documents

        title = toc_titles.get(item.get_name()) or first_heading(raw)
        doc.chapters.append(Chapter(text=text, title=title))

    return doc


def _first_meta(book, name: str) -> str | None:
    try:
        values = book.get_metadata("DC", name)
    except Exception:
        return None
    if not values:
        return None
    value = str(values[0][0]).strip()
    return value or None


def _toc_titles(book) -> dict[str, str]:
    """Map spine href -> TOC label. Flattens nested sections."""
    titles: dict[str, str] = {}

    def walk(entries) -> None:
        for entry in entries:
            if isinstance(entry, (list, tuple)):
                walk(entry)
                continue
            href = getattr(entry, "href", None)
            title = getattr(entry, "title", None)
            if href and title:
                titles[href.split("#")[0]] = str(title).strip()

    try:
        walk(book.toc)
    except Exception:
        pass
    return titles
