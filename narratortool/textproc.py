"""Turning extracted text into speakable chunks.

Two jobs:

`normalize` removes artifacts that sound wrong when read aloud — PDF line-break
hyphenation, page-number lines, Project Gutenberg's legal boilerplate, and the text
that lives inside figures and diagrams.

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

# --- figures and diagrams -----------------------------------------------------------
#
# A PDF's text layer contains everything drawn on the page, including the text inside
# charts. Extraction returns it as loose fragments in drawing order, so a paper full of
# plots narrates as "Figure 3.2 colon accuracy versus epochs zero point two zero point
# four zero point six". None of it is prose and none of it is speakable.
#
# Two shapes are recognised: caption lines (labelled, so they can be matched exactly)
# and the scattered axis ticks, tick labels, and legend keys around them (which have no
# label, so they are judged by what they are made of). Everything here errs towards
# keeping a line: a wrongly kept axis label is a moment of nonsense, a wrongly dropped
# line is content the listener never learns was there.

_FIGURE_WORD = (
    r"fig(?:ure|s)?|table|chart|diagram|graph|plate|exhibit|listing|algorithm"
    r"|scheme|illustration|image|photo|panel"
)
# A figure number: 3, 3.2, 3.2.1, 12a, S1, A.1, IV, or a bare letter.
# The roman-numeral and bare-letter forms are case-sensitive even though the figure
# word is not: under IGNORECASE they would match any lowercase letter, so "Figures are
# shown" would parse as figure "s" of a caption.
_FIGURE_NUMBER = r"(?:[A-Za-z]?\d+(?:[.-]\d+)*[a-z]?|(?-i:[IVXLCDM]{1,6}|[A-Z]))"
_CAPTION = re.compile(
    rf"^\(?(?:{_FIGURE_WORD})\b\.?\s*{_FIGURE_NUMBER}\)?"
    r"(?:"
    r"\s*$"                    # a bare label line: "Figure 3.2"
    r"|\s*[.:;,|)\]]+\s*"       # "Figure 3.2:" / "Table 2."
    r"|\s+[-—–]\s+"            # "Fig. 4 — Results" (spaced dash, not "P2-1")
    r'|\s+(?=(?-i:[A-Z])|[(\[\"\'])'  # "Figure 3.2 Accuracy versus epochs"
    r")",
    re.IGNORECASE,
)
# "Figure 3 shows that ..." is prose, not a caption: the separator forms above require
# either punctuation or a capitalised word after the number, so a lowercase verb here
# leaves the line alone.

_ENDS_SENTENCE = re.compile(r"[.!?][\"')\]]*$")

# How much of a wrapped caption to follow past its first line.
_MAX_CAPTION_LINES = 2
# Debris is short. Anything longer is prose that happens to contain numbers.
_MAX_DEBRIS_CHARS = 60
# Axis ticks come in runs; two numbers on a line is a date or a citation, not an axis.
_MIN_DEBRIS_NUMBERS = 3
# An axis name or legend key carries no digits, so it can only be recognised by the
# company it keeps. These bounds are what stop that spreading into the prose.
_MAX_LABEL_CHARS = 30
_MAX_LABEL_WORDS = 3
_MAX_LABELS_PER_SIDE = 6
# "10ms" and "1e-3" are numbers with a unit; "@DougBlank2" is a word with a digit in it.
_MAX_UNIT_LETTERS = 3


def normalize(text: str, strip_boilerplate: bool = True, strip_figures: bool = True) -> str:
    if strip_boilerplate:
        text = _strip_gutenberg(text)

    text = _pre_clean(text)

    if strip_figures:
        # Must happen before the wrapped-line join below, while a figure's fragments are
        # still on lines of their own; once joined into a paragraph they are unfindable.
        text = "\n".join(_partition_figure_lines(text)[0])

    # Join hard-wrapped lines inside a paragraph, but keep paragraph breaks. The \n in
    # the lookbehind is what protects blank-line separators: without it the second
    # newline of a "\n\n" pair gets joined and every paragraph runs together.
    text = re.sub(r"(?<![.!?:;\"'\)\n])\n(?!\n)", " ", text)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def find_figure_lines(text: str, strip_boilerplate: bool = True) -> list[str]:
    """The lines `normalize` would suppress as figure text. For reporting and tests."""
    if strip_boilerplate:
        text = _strip_gutenberg(text)
    return _partition_figure_lines(_pre_clean(text))[1]


def _pre_clean(text: str) -> str:
    """Line-preserving cleanups that must precede any per-line judgement."""
    # PDF extraction leaves words split across line breaks: "narra-\ntion".
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # A lone number on its own line is a page number, not content.
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)
    # Runs of dots/underscores (tables of contents, form fields) are unspeakable.
    return re.sub(r"[.…_]{4,}", " ", text)


def _partition_figure_lines(text: str) -> tuple[list[str], list[str]]:
    """Split lines into (kept, dropped-as-figure-text)."""
    lines = text.split("\n")
    verdict = _classify_lines(lines)
    _sweep_chart_labels(lines, verdict)

    kept = [line for line, v in zip(lines, verdict) if v is None]
    dropped = [line.strip() for line, v in zip(lines, verdict) if v is not None]
    return kept, dropped


def _classify_lines(lines: list[str]) -> list[str | None]:
    """Label each line None (keep), "caption", or "debris"."""
    verdict: list[str | None] = []
    caption_lines_left = 0

    for line in lines:
        stripped = line.strip()

        if caption_lines_left and stripped:
            # A caption that wrapped onto the next line. Follow it only until a line
            # ends a sentence, and never for more than _MAX_CAPTION_LINES.
            caption_lines_left -= 1
            if _ENDS_SENTENCE.search(stripped):
                caption_lines_left = 0
            verdict.append("caption")
            continue
        caption_lines_left = 0

        if not stripped:
            verdict.append(None)
        elif _CAPTION.match(stripped):
            caption_lines_left = 0 if _ENDS_SENTENCE.search(stripped) else _MAX_CAPTION_LINES
            verdict.append("caption")
        elif _is_plot_debris(stripped):
            verdict.append("debris")
        else:
            verdict.append(None)

    return verdict


def _sweep_chart_labels(lines: list[str], verdict: list[str | None]) -> None:
    """Drop axis names and legend keys sitting against a run of chart numbers.

    "Accuracy", "Epochs", "baseline ours ablation" are ordinary words — nothing about
    them reads as chart text on their own, and a rule broad enough to catch them in
    isolation would eat every heading in the book. What marks them is where they sit:
    immediately against the tick numbers, with no blank line between.

    Only numeric debris seeds this, never a caption and never a punctuation-only line.
    A section heading following a caption is exactly the case that would otherwise be
    swallowed, and a stray "," left behind by an extractor must never pull the words
    around it out of the book.
    """
    # Seeds are snapshotted before sweeping. Reading the list while mutating it would
    # let each swept label seed another sweep, and the per-seed cap would bound nothing.
    seeds = [
        i for i, kind in enumerate(verdict)
        if kind == "debris" and any(c.isdigit() for c in lines[i])
    ]

    for seed in seeds:
        for step in (-1, 1):
            index = seed + step
            for _ in range(_MAX_LABELS_PER_SIDE):
                if not 0 <= index < len(lines):
                    break
                if verdict[index] is not None:  # already gone; keep walking outward
                    index += step
                    continue
                if not _is_chart_label(lines[index]):
                    break
                verdict[index] = "debris"
                index += step


def _is_chart_label(line: str) -> bool:
    """A bare word or two that could be an axis name or a legend key."""
    stripped = line.strip()
    if not stripped or len(stripped) > _MAX_LABEL_CHARS:
        return False
    if any(c.isdigit() for c in stripped):
        return False  # numeric lines are debris on their own terms
    if stripped[-1] in ".!?:;,":
        return False  # punctuation means a sentence or a list lead-in, not a label
    return len(stripped.split()) <= _MAX_LABEL_WORDS


def _alpha_count(token: str) -> int:
    return sum(1 for c in token if c.isalpha())


def _is_plot_debris(line: str) -> bool:
    """Is this line the loose text of a chart rather than a sentence?

    Axis ticks, tick labels, and legend keys extract as short lines made of numbers and
    stray symbols. Prose that merely mentions numbers keeps its words, which is what
    separates "Accuracy 0.2 0.4 0.6 0.8" from "In 1901 he sailed for two years."
    """
    if len(line) > _MAX_DEBRIS_CHARS:
        return False

    tokens = line.split()
    if not tokens:
        return False

    numeric = symbolic = single = words = 0
    for token in tokens:
        bare = token.strip("([{<>}])\"'“”‘’.,;:!?—–-")
        if not bare:
            symbolic += 1
        elif any(c.isdigit() for c in bare) and _alpha_count(bare) <= _MAX_UNIT_LETTERS:
            numeric += 1  # 0.5, 10ms, 1e-3, 90° — a number, optionally with a unit
        elif not any(c.isalnum() for c in bare):
            symbolic += 1
        elif any(c.isdigit() for c in bare):
            words += 1  # "@DougBlank2", "COVID19" — a word, not a tick label
        elif len(bare) == 1:
            single += 1  # axis names: x, y, n, k
        else:
            words += 1

    # Nothing but numbers, symbols and single letters. The digit requirement keeps a
    # lone roman numeral or a bare "I." — chapter headings in plain-text books — safe.
    if numeric + symbolic + single == len(tokens) and (numeric or symbolic == len(tokens)):
        return True

    # A run of numbers with a label or two attached: "Epochs 10 20 30 40".
    return numeric >= _MIN_DEBRIS_NUMBERS and words <= 2


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
