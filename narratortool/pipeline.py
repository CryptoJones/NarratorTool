"""Orchestration: document in, MP3 out.

Synthesis of a full book is a long-running job over thousands of independent chunks,
and neural TTS hiccups: a chunk of odd punctuation trips the phonemizer, CUDA drops an
allocation, the model emits a zero-length tensor. Losing hours of GPU time to one bad
chunk is the failure mode worth designing against, so the loop here is deliberately
defensive:

  * every chunk gets retried, then retried sentence-by-sentence, before it counts as lost
  * a lost chunk is skipped (default) or aborts the run (`on_error="abort"`)
  * a run that dies anyway — abort, Ctrl-C, full disk — still encodes what it had to a
    `.partial.mp3` rather than throwing the audio away with the temp directory
  * every chunk, retry, and skip is recorded in a JSONL run log next to the output

A long run of consecutive failures is treated as a broken backend rather than bad text
and aborts regardless of policy; skipping five thousand chunks to produce silence helps
nobody.
"""
from __future__ import annotations

import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .audio import PCMWriter, Tags, encode_mp3, format_duration, require_ffmpeg
from .document import Document
from .extract import extract
from .runlog import RunLog, default_log_path
from .textproc import chunk, find_figure_lines, normalize, split_sentences
from .tts import BackendUnavailable, SynthesisError, TTSBackend, get_backend

log = logging.getLogger(__name__)

# Pauses that make continuous narration sound like reading rather than a stream.
PAUSE_BETWEEN_CHUNKS = 0.18
PAUSE_BETWEEN_CHAPTERS = 1.1

# Spoken where a chunk was lost. A silent gap is indistinguishable from a pause in the
# reading; saying so in the narrator's own voice makes the omission audible to whoever
# is listening, and lines up with the skip recorded in the run log.
OMISSION_NOTICE = "Passage Omitted."
PAUSE_AROUND_NOTICE = 0.35

# Retry policy for a hiccuping backend.
DEFAULT_RETRIES = 2
RETRY_BACKOFF = 0.75  # seconds, multiplied by attempt number
MAX_CONSECUTIVE_FAILURES = 5  # past this it is the backend, not the text

ProgressFn = Callable[[int, int, str], None]


class NarrationAborted(Exception):
    """A run stopped early. Carries whatever audio was salvaged, if any."""

    def __init__(self, message: str, partial_output: Path | None = None,
                 log_path: Path | None = None) -> None:
        super().__init__(message)
        self.partial_output = partial_output
        self.log_path = log_path


@dataclass
class ChunkFailure:
    index: int
    chapter: int
    text: str
    error: str

    @property
    def preview(self) -> str:
        return self.text[:60].replace("\n", " ")


@dataclass
class NarrationResult:
    output: Path
    duration_seconds: float
    chunk_count: int
    chapter_count: int
    voice: str
    failures: list[ChunkFailure] = field(default_factory=list)
    retried: int = 0
    degraded: int = 0  # chunks salvaged sentence-by-sentence, missing some audio
    notices_spoken: int = 0  # skipped chunks marked aloud in the audio
    log_path: Path | None = None
    partial: bool = False

    @property
    def ok(self) -> bool:
        return not self.failures and not self.partial

    def summary(self) -> str:
        parts = [
            f"{self.output.name} — {format_duration(self.duration_seconds)}, "
            f"{self.chunk_count} chunks across {self.chapter_count} chapter(s), voice {self.voice}"
        ]
        if self.retried:
            parts.append(f"{self.retried} chunk(s) needed a retry")
        if self.degraded:
            parts.append(f"{self.degraded} chunk(s) partially recovered")
        if self.failures:
            marked = " (marked aloud)" if self.notices_spoken == len(self.failures) else (
                f" ({self.notices_spoken} marked aloud)" if self.notices_spoken else "")
            parts.append(f"{len(self.failures)} chunk(s) SKIPPED{marked} — see the run log")
        if self.partial:
            parts.append("PARTIAL: the run did not finish")
        return "; ".join(parts)


def narrate_file(
    source: str | Path,
    output: str | Path | None = None,
    backend: TTSBackend | None = None,
    max_chars: int = 480,
    announce_chapters: bool = False,
    strip_boilerplate: bool = True,
    strip_figures: bool = True,
    progress: ProgressFn | None = None,
    retries: int = DEFAULT_RETRIES,
    on_error: str = "skip",
    omission_notice: str | None = OMISSION_NOTICE,
    log_path: str | Path | None = None,
    write_log: bool = True,
    **backend_kwargs,
) -> NarrationResult:
    """Parse `source`, synthesize it, and write an MP3.

    `on_error` is "skip" (log the chunk and keep going) or "abort" (stop, salvaging
    the audio produced so far to a .partial.mp3).

    `omission_notice` is spoken in place of each skipped chunk; None leaves a silent gap.
    """
    if on_error not in ("skip", "abort"):
        raise ValueError(f"on_error must be 'skip' or 'abort', not {on_error!r}")
    retries = max(0, int(retries))

    require_ffmpeg()  # fail before spending minutes on synthesis

    source = Path(source)
    doc = extract(source)
    if not doc.chapters:
        raise ValueError(f"{source.name} produced no text to narrate")

    backend = backend or get_backend("kokoro", **backend_kwargs)
    output = Path(output) if output else source.with_suffix(".mp3")

    units = list(_units(doc, max_chars, announce_chapters, strip_boilerplate, strip_figures))
    if not units:
        raise ValueError(f"{source.name} produced no speakable chunks")

    # Counted rather than inferred from the chunk count: a listener cannot hear what was
    # never spoken, so the number of suppressed figure lines belongs in the log.
    suppressed = (
        [line for ch in doc.chapters for line in find_figure_lines(ch.text, strip_boilerplate)]
        if strip_figures else []
    )
    if suppressed:
        log.info("suppressed %d line(s) of figure/diagram text", len(suppressed))

    resolved_log = Path(log_path) if log_path else (default_log_path(output) if write_log else None)
    log.info("%s -> %s (%d chunks, voice %s)", source.name, output.name, len(units), backend.name)
    if resolved_log:
        log.info("run log: %s", resolved_log)

    started = time.monotonic()
    with RunLog(resolved_log) as runlog:
        runlog.write(
            "run_start",
            source=str(source),
            output=str(output),
            backend=backend.name,
            sample_rate=backend.sample_rate,
            chunks=len(units),
            chapters=len(doc.chapters),
            title=doc.title,
            author=doc.author,
            figure_lines_suppressed=len(suppressed),
            figure_lines=suppressed[:50],  # capped: enough to audit, not a second copy
            settings={
                "max_chars": max_chars,
                "announce_chapters": announce_chapters,
                "strip_boilerplate": strip_boilerplate,
                "strip_figures": strip_figures,
                "retries": retries,
                "on_error": on_error,
                "omission_notice": omission_notice,
            },
        )
        try:
            return _run(
                units=units,
                doc=doc,
                backend=backend,
                output=output,
                progress=progress,
                retries=retries,
                on_error=on_error,
                omission_notice=omission_notice,
                runlog=runlog,
                log_path=resolved_log,
                started=started,
            )
        except NarrationAborted as exc:
            runlog.write(
                "run_end",
                status="aborted",
                error=str(exc),
                partial_output=str(exc.partial_output) if exc.partial_output else None,
                wall_seconds=round(time.monotonic() - started, 2),
            )
            raise


def _run(
    units: list[tuple[int, str]],
    doc: Document,
    backend: TTSBackend,
    output: Path,
    progress: ProgressFn | None,
    retries: int,
    on_error: str,
    omission_notice: str | None,
    runlog: RunLog,
    log_path: Path | None,
    started: float,
) -> NarrationResult:
    failures: list[ChunkFailure] = []
    retried = 0
    degraded = 0
    consecutive = 0
    spoken_samples = 0  # excludes inter-chunk padding, which would mask a silent run
    notice = _OmissionNotice(backend, omission_notice)

    with tempfile.TemporaryDirectory(prefix="narratortool-") as tmp:
        pcm_path = Path(tmp) / "audio.pcm"
        writer = PCMWriter(pcm_path, backend.sample_rate)
        try:
            last_chapter = None
            for index, (chapter_index, text) in enumerate(units, start=1):
                if last_chapter is not None and chapter_index != last_chapter:
                    writer.append_silence(PAUSE_BETWEEN_CHAPTERS)
                elif index > 1:
                    writer.append_silence(PAUSE_BETWEEN_CHUNKS)
                last_chapter = chapter_index

                if progress:
                    progress(index, len(units), text)

                offset = writer.duration_seconds
                chunk_started = time.monotonic()
                audio, attempts, partial_recovery, error = _synthesize(backend, text, retries)

                if error is not None:
                    consecutive += 1
                    failure = ChunkFailure(index, chapter_index, text, error)
                    log.warning("chunk %d/%d failed (%s): %s",
                                index, len(units), error, failure.preview)

                    # A run that is about to abort gets no notice: the audio simply ends.
                    fatal = on_error == "abort" or consecutive >= MAX_CONSECUTIVE_FAILURES
                    marker_samples = 0 if fatal else notice.append_to(writer)

                    runlog.write(
                        "chunk",
                        index=index,
                        chapter=chapter_index,
                        chars=len(text),
                        text=text[:200],
                        status="failed" if fatal else "skipped",
                        attempts=attempts,
                        offset_seconds=round(offset, 3),
                        notice=None if fatal else ("spoken" if marker_samples else "silent"),
                        notice_seconds=(round(marker_samples / backend.sample_rate, 3)
                                        if marker_samples and backend.sample_rate else 0.0),
                        error=error,
                    )

                    if on_error == "abort":
                        raise _abort(
                            f"chunk {index}/{len(units)} failed after {attempts} attempt(s): {error}",
                            writer, pcm_path, output, backend, doc, runlog, log_path,
                        )
                    if consecutive >= MAX_CONSECUTIVE_FAILURES:
                        raise _abort(
                            f"{consecutive} chunks failed in a row at chunk {index}/{len(units)} — "
                            f"the backend looks broken, not the text ({error})",
                            writer, pcm_path, output, backend, doc, runlog, log_path,
                        )
                    failures.append(failure)
                    continue

                consecutive = 0
                if attempts > 1:
                    retried += 1
                if partial_recovery:
                    degraded += 1

                written = writer.append(audio)
                spoken_samples += written
                runlog.write(
                    "chunk",
                    index=index,
                    chapter=chapter_index,
                    chars=len(text),
                    text=text[:200],
                    status=("recovered" if partial_recovery else "retried" if attempts > 1
                            else "silent" if written == 0 else "ok"),
                    attempts=attempts,
                    offset_seconds=round(offset, 3),
                    seconds=round(written / backend.sample_rate, 3) if backend.sample_rate else 0.0,
                    synthesis_seconds=round(time.monotonic() - chunk_started, 3),
                )
                if written == 0:
                    log.warning("chunk %d/%d produced no audio: %s",
                                index, len(units), text[:60].replace("\n", " "))

            if spoken_samples == 0:
                # The padding between chunks would otherwise encode happily into an MP3
                # of pure silence, which is worse than an error: it looks like success.
                writer.close()
                raise NarrationAborted(
                    "no audio was synthesized — every chunk failed or produced silence",
                    partial_output=None,
                    log_path=log_path,
                )

            duration = writer.duration_seconds
            writer.close()

            encode_mp3(pcm_path, output, backend.sample_rate, tags=_tags(doc, backend),
                       audio_filter=getattr(backend, "audio_filter", None))

        except NarrationAborted:
            raise
        except (KeyboardInterrupt, Exception) as exc:  # noqa: BLE001 - salvaged then re-raised
            raise _abort(
                f"{type(exc).__name__}: {exc}" if not isinstance(exc, KeyboardInterrupt)
                else "interrupted",
                writer, pcm_path, output, backend, doc, runlog, log_path,
            ) from exc
        finally:
            writer.close()

    result = NarrationResult(
        output=output,
        duration_seconds=duration,
        chunk_count=len(units),
        chapter_count=len(doc.chapters),
        voice=backend.name,
        failures=failures,
        retried=retried,
        degraded=degraded,
        notices_spoken=notice.spoken,
        log_path=log_path,
    )
    runlog.write(
        "run_end",
        status="ok" if result.ok else "completed_with_skips",
        output=str(output),
        duration_seconds=round(duration, 2),
        chunks=len(units),
        skipped=len(failures),
        notices_spoken=notice.spoken,
        retried=retried,
        recovered=degraded,
        nonfinite_chunks=writer.nonfinite_chunks,
        spoken_seconds=round(spoken_samples / backend.sample_rate, 2) if backend.sample_rate else 0.0,
        skipped_indexes=[f.index for f in failures],
        wall_seconds=round(time.monotonic() - started, 2),
    )
    return result


def _abort(message, writer, pcm_path, output, backend, doc, runlog, log_path) -> NarrationAborted:
    """Salvage whatever audio exists and build the exception to raise."""
    partial = _salvage(writer, pcm_path, output, backend, doc, runlog)
    return NarrationAborted(message, partial_output=partial, log_path=log_path)


def _salvage(writer, pcm_path: Path, output: Path, backend, doc, runlog) -> Path | None:
    """Encode the audio produced so far to `<output stem>.partial.mp3`.

    Called from the failure path, so it must not raise: a salvage that fails should
    leave the original error intact rather than replacing it.
    """
    try:
        writer.close()
        if writer.duration_seconds <= 0 or not pcm_path.is_file() or pcm_path.stat().st_size == 0:
            return None
        partial_path = output.with_name(f"{output.stem}.partial{output.suffix or '.mp3'}")
        encode_mp3(pcm_path, partial_path, backend.sample_rate, tags=_tags(doc, backend),
                   audio_filter=getattr(backend, "audio_filter", None))
        log.warning("salvaged %s of audio to %s",
                    format_duration(writer.duration_seconds), partial_path)
        runlog.write(
            "salvage",
            output=str(partial_path),
            duration_seconds=round(writer.duration_seconds, 2),
        )
        return partial_path
    except Exception as exc:  # noqa: BLE001 - salvage must never mask the real error
        log.warning("could not salvage partial audio: %s", exc)
        runlog.write("salvage", status="failed", error=str(exc))
        return None


class _OmissionNotice:
    """Speaks a short marker in the narrator's own voice where a chunk was lost.

    The audio is synthesized once, on the first skip, and reused: the text never
    changes, a clean run pays nothing, and a book with two hundred skips does not
    re-render the same sentence two hundred times. If the marker itself cannot be
    synthesized the gap falls back to silence — a failed notice must not escalate into
    a failed run. Rendering is retried on the next skip rather than given up on after
    one blip, since a backend that hiccuped on a chunk may well render this fine, but
    only a few times: silent gaps beat stalling on every skip in a doomed run.
    """

    MAX_RENDER_ATTEMPTS = 3

    def __init__(self, backend: TTSBackend, text: str | None) -> None:
        self.backend = backend
        self.text = (text or "").strip()
        self.spoken = 0
        self._audio: np.ndarray | None = None
        self._attempts = 0
        self._unavailable = not self.text

    def _render(self) -> np.ndarray | None:
        if self._audio is not None or self._unavailable:
            return self._audio
        self._attempts += 1
        try:
            audio = self.backend.synthesize(self.text)
        except (BackendUnavailable, ImportError):
            raise  # the engine is dead — that is the run's problem, not the notice's
        except Exception as exc:  # noqa: BLE001
            self._give_up_after_attempts(f"could not synthesize {self.text!r} ({exc})")
            return None
        if audio is None or not getattr(audio, "size", 0):
            self._give_up_after_attempts(f"{self.text!r} produced no audio")
            return None
        self._audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        return self._audio

    def _give_up_after_attempts(self, reason: str) -> None:
        if self._attempts >= self.MAX_RENDER_ATTEMPTS:
            self._unavailable = True
            log.warning("%s; remaining gaps will be silent", reason)
        else:
            log.debug("%s; will try again at the next skip", reason)

    def append_to(self, writer: PCMWriter) -> int:
        """Write the marker, padded so it does not run into the surrounding prose."""
        audio = self._render()
        if audio is None:
            return 0
        written = writer.append_silence(PAUSE_AROUND_NOTICE)
        written += writer.append(audio)
        written += writer.append_silence(PAUSE_AROUND_NOTICE)
        self.spoken += 1
        return written


def _tags(doc: Document, backend: TTSBackend) -> Tags:
    return Tags(
        title=doc.title,
        artist=doc.author,
        album=doc.title,
        comment=f"Narrated by NarratorTool ({backend.name})",
    )


def _synthesize(backend: TTSBackend, text: str, retries: int):
    """Synthesize one chunk, retrying on failure.

    Returns (audio, attempts, partial_recovery, error). `error` is None on success.
    The last resort before giving up is re-running the chunk sentence by sentence: most
    hiccups are one hostile fragment, and keeping the other sentences beats dropping the
    whole chunk — `partial_recovery` flags that the audio may be missing a sentence.
    """
    last_error = None
    for attempt in range(1, retries + 2):
        try:
            audio = backend.synthesize(text)
            if audio is None:
                raise SynthesisError("backend returned None")
            return audio, attempt, False, None
        except (BackendUnavailable, ImportError):
            raise  # the engine is dead; retrying it thousands of times helps nobody
        except Exception as exc:  # noqa: BLE001 - any backend fault is retryable
            last_error = f"{type(exc).__name__}: {exc}"
            log.debug("synthesis attempt %d failed: %s", attempt, last_error)
            if attempt <= retries:
                time.sleep(RETRY_BACKOFF * attempt)

    attempts = retries + 1
    if len(split_sentences(text)) < 2:
        return None, attempts, False, last_error  # nothing left to try

    audio, recovered_all = _synthesize_piecewise(backend, text)
    attempts += 1
    if audio is not None:
        return audio, attempts, not recovered_all, None
    return None, attempts, False, last_error


def _synthesize_piecewise(backend: TTSBackend, text: str):
    """Retry a failed chunk one sentence at a time. Returns (audio|None, all_survived)."""
    sentences = [s for s in split_sentences(text) if s.strip()]
    if len(sentences) < 2:
        return None, False
    parts, lost = [], 0
    for sentence in sentences:
        try:
            piece = backend.synthesize(sentence)
        except (BackendUnavailable, ImportError):
            raise
        except Exception as exc:  # noqa: BLE001
            lost += 1
            log.debug("sentence-level retry failed: %s", exc)
            continue
        if piece is not None and getattr(piece, "size", 0):
            parts.append(np.asarray(piece, dtype=np.float32).reshape(-1))

    if not parts:
        return None, False
    log.info("recovered chunk sentence-by-sentence (%d/%d sentences)",
             len(sentences) - lost, len(sentences))
    return np.concatenate(parts), lost == 0


def _units(
    doc: Document,
    max_chars: int,
    announce_chapters: bool,
    strip_boilerplate: bool,
    strip_figures: bool = True,
) -> Iterable[tuple[int, str]]:
    """Yield (chapter_index, chunk_text) across the whole document."""
    for chapter_index, chapter in enumerate(doc.chapters):
        if announce_chapters and chapter.title:
            yield chapter_index, chapter.title
        body = normalize(chapter.text, strip_boilerplate=strip_boilerplate,
                         strip_figures=strip_figures)
        for piece in chunk(body, max_chars=max_chars):
            yield chapter_index, piece
