"""Turning extracted text into speakable chunks.

Two jobs:

`normalize` removes artifacts that sound wrong when read aloud — PDF line-break
hyphenation, page-number lines, Project Gutenberg's legal boilerplate.

`chunk` splits into pieces small enough for a TTS model's context. Splitting happens
on sentence boundaries because a chunk edge inside a sentence is audible: the model
drops the falling intonation of the sentence end, so the seam between two audio
segments clicks. Sentences longer than the limit are split on clause punctuation,
and only then on whitespace as a last resort.
"""
from __future__ import annotations

import re

# Abbreviations whose trailing period must not be read as a sentence end.
_ABBREV = (
    r"Mr|Mrs|Ms|Dr|Prof|St|Sr|Jr|Rev|Hon|Gen|Col|Capt|Lt|Sgt|Fr"
    r"|vs|etc|e\.g|i\.e|cf|al|Inc|Ltd|Co|Corp|approx|No|vol|pp"
    r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
_ABBREV_DOT = re.compile(rf"\b({_ABBREV})\.", re.IGNORECASE)
# A sentence terminator, plus any closing quote/bracket, followed by a break.
# Written as a lookahead rather than a lookbehind: Python's re requires fixed-width
# lookbehind, and the abbreviation list is not fixed width.
_TERMINATOR = re.compile(r"[.!?]+[\"')\]]*(?=\s|$)")
_CLAUSE_END = re.compile(r"(?<=[;:,—–])\s+")

# Stands in for an abbreviation's period while splitting, so "Dr." is not read as a
# sentence end. Restored immediately afterwards.
_DOT = "\x00"

_GUTENBERG_START = re.compile(r"^\*\*\*\s*START OF TH[EIS].*?\*\*\*\s*$", re.MULTILINE)
_GUTENBERG_END = re.compile(r"^\*\*\*\s*END OF TH[EIS].*?\*\*\*\s*$", re.MULTILINE)


def normalize(text: str, strip_boilerplate: bool = True) -> str:
    if strip_boilerplate:
        text = _strip_gutenberg(text)

    # PDF extraction leaves words split across line breaks: "narra-\ntion".
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # A lone number on its own line is a page number, not content.
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)
    # Runs of dots/underscores (tables of contents, form fields) are unspeakable.
    text = re.sub(r"[.…_]{4,}", " ", text)
    # Join hard-wrapped lines inside a paragraph, but keep paragraph breaks. The \n in
    # the lookbehind is what protects blank-line separators: without it the second
    # newline of a "\n\n" pair gets joined and every paragraph runs together.
    text = re.sub(r"(?<![.!?:;\"'\)\n])\n(?!\n)", " ", text)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_gutenberg(text: str) -> str:
    start = _GUTENBERG_START.search(text)
    if start:
        text = text[start.end():]
    end = _GUTENBERG_END.search(text)
    if end:
        text = text[: end.start()]
    return text


def split_sentences(text: str) -> list[str]:
    protected = _ABBREV_DOT.sub(lambda m: m.group(1) + _DOT, text)

    sentences: list[str] = []
    start = 0
    for match in _TERMINATOR.finditer(protected):
        # A real boundary is followed by something that can begin a sentence. When the
        # next word is lowercase this was dialogue punctuation inside one sentence —
        # `"Stop!" she cried.` is one sentence, and splitting it makes the narrator
        # drop its voice on "Stop!" and restart cold on "she cried".
        rest = protected[match.end():].lstrip()
        if rest and rest[0].islower():
            continue

        piece = protected[start : match.end()].strip()
        if piece:
            sentences.append(piece)
        start = match.end()

    tail = protected[start:].strip()
    if tail:
        sentences.append(tail)

    return [s.replace(_DOT, ".") for s in sentences]


def chunk(text: str, max_chars: int = 480) -> list[str]:
    """Group sentences into chunks of at most `max_chars`.

    The default suits Kokoro, which degrades past roughly 500 characters per pass.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be positive")

    chunks: list[str] = []
    buf = ""

    for para in [p for p in text.split("\n\n") if p.strip()]:
        for sentence in split_sentences(para):
            for piece in _fit(sentence, max_chars):
                if not buf:
                    buf = piece
                elif len(buf) + 1 + len(piece) <= max_chars:
                    buf = f"{buf} {piece}"
                else:
                    chunks.append(buf)
                    buf = piece
        # A paragraph boundary is a natural breath; do not merge across it.
        if buf:
            chunks.append(buf)
            buf = ""

    if buf:
        chunks.append(buf)
    return chunks


def _fit(sentence: str, max_chars: int) -> list[str]:
    """Break one over-long sentence down until every piece fits."""
    if len(sentence) <= max_chars:
        return [sentence]

    pieces: list[str] = []
    buf = ""
    for part in _CLAUSE_END.split(sentence):
        if not buf:
            buf = part
        elif len(buf) + 1 + len(part) <= max_chars:
            buf = f"{buf} {part}"
        else:
            pieces.append(buf)
            buf = part
    if buf:
        pieces.append(buf)

    # Still too long (no clause punctuation) — fall back to hard word wrapping.
    out: list[str] = []
    for piece in pieces:
        while len(piece) > max_chars:
            cut = piece.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            out.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        if piece:
            out.append(piece)
    return out
