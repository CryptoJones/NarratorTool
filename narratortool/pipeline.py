"""Orchestration: document in, MP3 out."""
from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .audio import PCMWriter, Tags, encode_mp3, format_duration, require_ffmpeg
from .document import Document
from .extract import extract
from .textproc import chunk, normalize
from .tts import TTSBackend, get_backend

log = logging.getLogger(__name__)

# Pauses that make continuous narration sound like reading rather than a stream.
PAUSE_BETWEEN_CHUNKS = 0.18
PAUSE_BETWEEN_CHAPTERS = 1.1

ProgressFn = Callable[[int, int, str], None]


@dataclass
class NarrationResult:
    output: Path
    duration_seconds: float
    chunk_count: int
    chapter_count: int
    voice: str

    def summary(self) -> str:
        return (
            f"{self.output.name} — {format_duration(self.duration_seconds)}, "
            f"{self.chunk_count} chunks across {self.chapter_count} chapter(s), voice {self.voice}"
        )


def narrate_file(
    source: str | Path,
    output: str | Path | None = None,
    backend: TTSBackend | None = None,
    max_chars: int = 480,
    announce_chapters: bool = False,
    strip_boilerplate: bool = True,
    progress: ProgressFn | None = None,
    **backend_kwargs,
) -> NarrationResult:
    """Parse `source`, synthesize it, and write an MP3."""
    require_ffmpeg()  # fail before spending minutes on synthesis

    source = Path(source)
    doc = extract(source)
    if not doc.chapters:
        raise ValueError(f"{source.name} produced no text to narrate")

    backend = backend or get_backend("kokoro", **backend_kwargs)
    output = Path(output) if output else source.with_suffix(".mp3")

    units = list(_units(doc, max_chars, announce_chapters, strip_boilerplate))
    if not units:
        raise ValueError(f"{source.name} produced no speakable chunks")

    log.info("%s -> %s (%d chunks, voice %s)", source.name, output.name, len(units), backend.name)

    with tempfile.TemporaryDirectory(prefix="narratortool-") as tmp:
        pcm_path = Path(tmp) / "audio.pcm"
        with PCMWriter(pcm_path, backend.sample_rate) as writer:
            last_chapter = None
            for index, (chapter_index, text) in enumerate(units, start=1):
                if last_chapter is not None and chapter_index != last_chapter:
                    writer.append_silence(PAUSE_BETWEEN_CHAPTERS)
                elif index > 1:
                    writer.append_silence(PAUSE_BETWEEN_CHUNKS)
                last_chapter = chapter_index

                if progress:
                    progress(index, len(units), text)
                writer.append(backend.synthesize(text))

            duration = writer.duration_seconds

        encode_mp3(
            pcm_path,
            output,
            backend.sample_rate,
            tags=Tags(
                title=doc.title,
                artist=doc.author,
                album=doc.title,
                comment=f"Narrated by NarratorTool ({backend.name})",
            ),
        )

    return NarrationResult(
        output=output,
        duration_seconds=duration,
        chunk_count=len(units),
        chapter_count=len(doc.chapters),
        voice=backend.name,
    )


def _units(
    doc: Document, max_chars: int, announce_chapters: bool, strip_boilerplate: bool
) -> Iterable[tuple[int, str]]:
    """Yield (chapter_index, chunk_text) across the whole document."""
    for chapter_index, chapter in enumerate(doc.chapters):
        if announce_chapters and chapter.title:
            yield chapter_index, chapter.title
        body = normalize(chapter.text, strip_boilerplate=strip_boilerplate)
        for piece in chunk(body, max_chars=max_chars):
            yield chapter_index, piece
