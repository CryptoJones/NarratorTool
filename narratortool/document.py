"""The document model every extractor produces.

Extractors differ wildly in what they can recover — a .txt file has no metadata at
all, an EPUB has a full spine with chapter titles — so the model is deliberately
forgiving: everything except `chapters` is optional, and a document with a single
untitled chapter is a perfectly valid result.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chapter:
    """One narratable unit. `title` is None when the source has no structure."""

    text: str
    title: str | None = None

    def __post_init__(self) -> None:
        self.text = self.text.strip()

    @property
    def is_empty(self) -> bool:
        return not self.text


@dataclass
class Document:
    chapters: list[Chapter] = field(default_factory=list)
    title: str | None = None
    author: str | None = None
    source_path: str | None = None

    @property
    def text(self) -> str:
        """Whole document as one string, chapters separated by blank lines."""
        return "\n\n".join(c.text for c in self.chapters if not c.is_empty)

    @property
    def char_count(self) -> int:
        return sum(len(c.text) for c in self.chapters)

    def drop_empty(self) -> "Document":
        """EPUBs in particular are full of nav/cover documents that extract to nothing."""
        self.chapters = [c for c in self.chapters if not c.is_empty]
        return self
