"""PDF extraction.

PDFs carry no reliable chapter structure, so this maps the outline (bookmarks) onto
page ranges when one exists and falls back to one chapter for the whole file when it
does not. Scanned PDFs with no text layer produce nothing — that is detected and
reported rather than silently yielding an empty narration.
"""
from __future__ import annotations

from pathlib import Path

from ..document import Chapter, Document


class NoTextLayer(Exception):
    """Raised for image-only PDFs, which need OCR before they can be narrated."""


def extract_pdf(path: Path) -> Document:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("PDF support needs `pypdf` (pip install 'narratortool[pdf]')") from exc

    reader = PdfReader(str(path))
    pages = [(p.extract_text() or "").strip() for p in reader.pages]

    if not any(pages):
        raise NoTextLayer(
            f"{path.name} has no extractable text — it is probably a scan. "
            "OCR it first (e.g. `ocrmypdf in.pdf out.pdf`), then narrate the result."
        )

    meta = reader.metadata or {}
    doc = Document(
        title=(meta.get("/Title") or "").strip() or None,
        author=(meta.get("/Author") or "").strip() or None,
    )

    bounds = _outline_bounds(reader, len(pages))
    if bounds:
        # The first bookmark rarely sits on page one. Whatever precedes it — title
        # page, abstract, dedication, preface — belongs to no chapter, and joining
        # only the bookmarked ranges drops it silently. It leads the document, so it
        # leads the narration, untitled: there is no heading to announce.
        front = _front_matter(pages, bounds[0][1])
        if front:
            doc.chapters.append(Chapter(text=front))
        for title, start, end in bounds:
            doc.chapters.append(Chapter(text="\n".join(pages[start:end]), title=title))
    else:
        doc.chapters.append(Chapter(text="\n".join(pages)))
    return doc


def _front_matter(pages: list[str], first_bookmarked_page: int) -> str:
    """The text ahead of the outline's first bookmark, empty when there is none."""
    if first_bookmarked_page <= 0:
        return ""
    return "\n".join(pages[:first_bookmarked_page]).strip()


def _outline_bounds(reader, page_count: int) -> list[tuple[str, int, int]]:
    """Flatten the PDF outline into (title, first_page, end_page) triples."""
    try:
        outline = reader.outline
    except Exception:
        return []

    marks: list[tuple[str, int]] = []

    def walk(items) -> None:
        for item in items:
            if isinstance(item, list):
                walk(item)
                continue
            try:
                page = reader.get_destination_page_number(item)
                title = str(item.title).strip()
            except Exception:
                continue
            if title:
                marks.append((title, page))

    try:
        walk(outline)
    except Exception:
        return []

    marks.sort(key=lambda m: m[1])
    if len(marks) < 2:
        return []

    return [
        (title, start, marks[i + 1][1] if i + 1 < len(marks) else page_count)
        for i, (title, start) in enumerate(marks)
    ]
