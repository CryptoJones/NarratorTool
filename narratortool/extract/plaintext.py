"""Plain text and Markdown."""
from __future__ import annotations

import re
from pathlib import Path

from ..document import Chapter, Document

# A heading keyword at the start of a short line.
_HEADING_WORD = re.compile(
    r"^(?:chapter|book|part|section|canto|act|scene|volume)\b", re.IGNORECASE
)
_MAX_HEADING_CHARS = 70


def _read(path: Path) -> str:
    # Gutenberg texts are frequently latin-1 or cp1252 despite a .txt extension.
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def extract_txt(path: Path) -> Document:
    raw = _read(path)
    chapters = _split_on_headings(raw)
    return Document(chapters=chapters or [Chapter(text=raw)])


def extract_markdown(path: Path) -> Document:
    raw = _read(path)
    chapters: list[Chapter] = []
    current_title: str | None = None
    buf: list[str] = []

    for line in raw.splitlines():
        heading = re.match(r"^(#{1,3})\s+(.*)", line)
        if heading:
            if buf:
                chapters.append(Chapter(text="\n".join(buf), title=current_title))
                buf = []
            current_title = heading.group(2).strip()
            continue
        buf.append(_strip_markdown(line))

    if buf:
        chapters.append(Chapter(text="\n".join(buf), title=current_title))
    return Document(chapters=chapters or [Chapter(text=raw)])


def _strip_markdown(line: str) -> str:
    """Remove the markup a narrator should not read aloud."""
    line = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", line)          # images
    line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)      # links -> label
    line = re.sub(r"`{1,3}([^`]*)`{1,3}", r"\1", line)        # code spans
    line = re.sub(r"[*_]{1,3}([^*_]+)[*_]{1,3}", r"\1", line)  # emphasis
    line = re.sub(r"^\s{0,3}>\s?", "", line)                   # blockquote marker
    return line


def _is_heading(lines: list[str], i: int) -> bool:
    """A heading is a SHORT, STANDALONE line — blank line above, content below.

    The standalone requirement is what makes this safe on hard-wrapped prose. Testing
    the keyword alone matches any wrapped line that happens to begin with "book" or
    "part", which on a Gutenberg Bible produces chapter titles like
    "book of the law, that he rent his clothes." — mid-sentence fragments.
    """
    line = lines[i].strip()
    if not line or len(line) > _MAX_HEADING_CHARS:
        return False
    # Trailing clause punctuation means we are mid-sentence, not at a title.
    if line[-1] in ",;:":
        return False
    if i > 0 and lines[i - 1].strip():
        return False  # no blank line above
    if i + 1 >= len(lines) or not lines[i + 1].strip():
        return False  # nothing follows, or another blank — not a heading + body

    return bool(_HEADING_WORD.match(line)) or (line.isupper() and len(line.split()) <= 12)


def _split_on_headings(raw: str) -> list[Chapter]:
    """Best-effort chapter split for unstructured text. Returns [] if nothing looks
    like a heading, so the caller can fall back to a single chapter."""
    lines = raw.splitlines()
    marks = [i for i in range(len(lines)) if _is_heading(lines, i)]
    if len(marks) < 2:
        return []

    chapters: list[Chapter] = []
    if marks[0] > 0:
        front = "\n".join(lines[: marks[0]])
        if front.strip():
            chapters.append(Chapter(text=front))

    for start, end in zip(marks, marks[1:] + [len(lines)]):
        chapters.append(
            Chapter(text="\n".join(lines[start + 1 : end]), title=lines[start].strip())
        )
    return chapters
