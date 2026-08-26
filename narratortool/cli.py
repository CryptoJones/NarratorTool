"""Command-line interface."""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from . import __version__
from .audio import AudioWriteError, FFmpegMissing, format_duration
from .extract import UnsupportedFormat, supported_extensions
from .pipeline import DEFAULT_RETRIES, OMISSION_NOTICE, NarrationAborted
from .textproc import chunk, normalize
from .tts import BackendUnavailable, available_backends, get_backend
from .tts.kokoro_backend import (
    DEFAULT_LANG,
    DEFAULT_SPEED,
    DEFAULT_VOICE,
    PROFILES,
    default_lang_for,
    default_speed_for,
)

# Seconds of speech per character of chunked text, at speed 1.0. Measured over a real
# 619-chunk run (Computer Science Distilled, 229,292 chars, 4:18 of audio at 0.96x):
# least-squares through the origin gives 0.0672 s/char at 0.96x, so 0.0645 at 1.0x.
# Duration scales as 1/speed. The old estimate assumed a flat 4s per chunk and was
# roughly 6x low, because a chunk averages ~370 chars and ~25s, not 4s.
SECONDS_PER_CHAR = 0.0645

# The house voice, and the rest of the Kokoro set worth knowing about. Cast profiles
# (PROFILES) are listed separately by --list-voices: they are blends, not single voices.
KNOWN_VOICES = {
    "bf_emma": "UK female — the CryptoJones house narration voice",
    "bf_isabella": "UK female",
    "bm_george": "UK male",
    "bm_lewis": "UK male",
    "af_heart": "US female, warm",
    "af_bella": "US female",
    "am_michael": "US male",
    "am_fenrir": "US male",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="narrate",
        description="Narrate documents to MP3 with local neural TTS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  narrate book.epub\n"
            "  narrate paper.pdf -o out/paper.mp3 --announce-chapters\n"
            "  narrate notes.md --voice bm_george --speed 1.0\n"
            f"\nsupported input: {', '.join(supported_extensions())}\n"
        ),
    )
    parser.add_argument("source", nargs="?", type=Path, help="document to narrate")
    parser.add_argument("-o", "--output", type=Path, help="output MP3 (default: alongside source)")

    voice = parser.add_argument_group("voice")
    voice.add_argument("--voice", default=DEFAULT_VOICE,
                       help=f"cast profile or Kokoro voice (default: {DEFAULT_VOICE})")
    voice.add_argument("--speed", type=float, default=None,
                       help=f"speech rate (default: the voice's own, {DEFAULT_SPEED} for "
                            f"{DEFAULT_VOICE})")
    voice.add_argument("--lang", default=None,
                       help="Kokoro language code; inferred from the voice prefix if omitted")
    voice.add_argument("--backend", default="kokoro", choices=available_backends())
    voice.add_argument("--device", default=None, help="cuda | mps | cpu (default: autodetect)")

    text = parser.add_argument_group("text handling")
    text.add_argument("--max-chars", type=int, default=480,
                      help="max characters per synthesis chunk (default: 480)")
    text.add_argument("--announce-chapters", action="store_true",
                      help="speak each chapter title before its body")
    text.add_argument("--keep-boilerplate", action="store_true",
                      help="do not strip Project Gutenberg headers/footers")
    text.add_argument("--keep-figures", action="store_true",
                      help="read figure captions and the text inside charts (PDF)")

    reliability = parser.add_argument_group("reliability")
    reliability.add_argument("--retries", type=int, default=DEFAULT_RETRIES,
                             help=f"retries per chunk before it is lost (default: {DEFAULT_RETRIES})")
    reliability.add_argument("--on-error", choices=("skip", "abort"), default="skip",
                             help="a chunk that will not synthesize: skip it (default) or stop")
    reliability.add_argument("--omission-notice", default=OMISSION_NOTICE, metavar="TEXT",
                             help=f"spoken where a chunk was skipped (default: {OMISSION_NOTICE!r})")
    reliability.add_argument("--no-omission-notice", action="store_true",
                             help="leave a silent gap instead of announcing a skip")
    reliability.add_argument("--log", type=Path, default=None,
                             help="run log path (default: <output>.narration.jsonl)")
    reliability.add_argument("--no-log", action="store_true", help="do not write a run log")
    reliability.add_argument("--strict", action="store_true",
                             help="exit non-zero if any chunk was skipped")

    other = parser.add_argument_group("other")
    other.add_argument("--dry-run", action="store_true",
                       help="parse and chunk, report stats, synthesize nothing")
    other.add_argument("--list-voices", action="store_true", help="list known voices and exit")
    other.add_argument("-q", "--quiet", action="store_true")
    other.add_argument("-V", "--version", action="version", version=f"narratortool {__version__}")
    return parser


def infer_lang(voice: str, explicit: str | None) -> str:
    """Kokoro voice names encode their language: bf_emma is British female.

    Getting this wrong is silent — a British voice under lang_code 'a' still produces
    audio, just with American phonemes — so it is inferred rather than defaulted.

    A cast profile is a blend and has no single prefix to read, so it states its own
    language; 'nia'[0] would otherwise infer the nonexistent language 'n'.
    """
    if explicit:
        return explicit
    from_profile = default_lang_for(voice)
    if from_profile:
        return from_profile
    return voice[0] if voice and voice[0].isalpha() else DEFAULT_LANG


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
    )

    if args.list_voices:
        print("cast profiles (blends, with their own speed and pitch):")
        for name, profile in PROFILES.items():
            marker = " *" if name == DEFAULT_VOICE else "  "
            blend = " + ".join(f"{w:.0%} {v}" for v, w in profile["voices"].items())
            print(f"{marker} {name:<16} {profile['description']}")
            print(f"    {'':<16} {blend}, {profile['speed']}x, "
                  f"{profile['pitch_semitones']:+g}st")
        print("\nstock Kokoro voices:")
        for name, desc in KNOWN_VOICES.items():
            marker = " *" if name == DEFAULT_VOICE else "  "
            print(f"{marker} {name:<16} {desc}")
        print("\n* = default. Prefix: a=American, b=British; f=female, m=male.")
        return 0

    if args.source is None:
        build_parser().print_help()
        return 2

    if args.retries < 0:
        print("error: --retries cannot be negative", file=sys.stderr)
        return 2
    if args.max_chars < 1:
        print("error: --max-chars must be at least 1", file=sys.stderr)
        return 2
    if args.speed is not None and not 0.1 <= args.speed <= 3.0:
        print(f"error: --speed {args.speed} is outside the usable range 0.1-3.0", file=sys.stderr)
        return 2

    try:
        if args.dry_run:
            return _dry_run(args)
        return _narrate(args)
    except NarrationAborted as exc:
        print(file=sys.stderr)
        print(f"error: narration stopped — {exc}", file=sys.stderr)
        if exc.partial_output:
            print(f"salvaged audio so far: {exc.partial_output}", file=sys.stderr)
        else:
            print("no audio had been produced yet; nothing to salvage", file=sys.stderr)
        if exc.log_path:
            print(f"run log: {exc.log_path}", file=sys.stderr)
        return 1
    except (UnsupportedFormat, FFmpegMissing, FileNotFoundError, ValueError,
            AudioWriteError, BackendUnavailable, PermissionError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        # Only reachable outside synthesis; inside it the pipeline salvages and
        # re-raises as NarrationAborted.
        print("\ninterrupted", file=sys.stderr)
        return 130


def _dry_run(args) -> int:
    from .extract import extract
    from .textproc import find_figure_lines

    doc = extract(args.source)
    total = 0
    spoken_chars = 0
    suppressed: list[str] = []
    print(f"title   : {doc.title}")
    print(f"author  : {doc.author or '(unknown)'}")
    print(f"chapters: {len(doc.chapters)}")
    for i, ch in enumerate(doc.chapters):
        if not args.keep_figures:
            suppressed += find_figure_lines(ch.text, not args.keep_boilerplate)
        pieces = chunk(
            normalize(ch.text, not args.keep_boilerplate, not args.keep_figures),
            args.max_chars,
        )
        total += len(pieces)
        spoken_chars += sum(len(p) for p in pieces)
        label = ch.title or f"(untitled {i})"
        print(f"  {i:>3}. {label[:58]:<58} {len(ch.text):>8,} chars  {len(pieces):>5} chunks")
    print(f"\ntotal chunks: {total:,}")
    if suppressed:
        # Worth showing before a long run: this is the one edit that silently removes
        # source text, so --dry-run is where you check it took nothing it should not.
        print(f"figure/diagram lines suppressed: {len(suppressed):,} "
              f"(--keep-figures to narrate them)")
        for line in suppressed[:5]:
            print(f"    - {line[:70]}")
        if len(suppressed) > 5:
            print(f"    ... and {len(suppressed) - 5:,} more")
    speed = args.speed if args.speed is not None else default_speed_for(args.voice)
    seconds = spoken_chars * SECONDS_PER_CHAR / speed
    print(f"estimated audio: ~{format_duration(seconds)} "
          f"({spoken_chars:,} spoken chars at {speed}x)")
    return 0


class _ProgressLine:
    """The \r progress line and the logger both write to the terminal.

    Without coordination a warning lands halfway through the progress line and both
    become unreadable — which matters here, because warnings are how a skipped chunk
    announces itself. This closes the pending line before any log record is emitted.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.pending = False

    def update(self, index: int, total: int, text: str) -> None:
        if not self.enabled:
            return
        pct = 100 * index / total
        preview = text[:48].replace("\n", " ")
        print(f"\r[{index:>5}/{total}] {pct:5.1f}%  {preview:<48}", end="", flush=True)
        self.pending = True

    def clear(self) -> None:
        if self.pending:
            print(flush=True)
            self.pending = False

    def install(self) -> None:
        """Make every log record clear the progress line first."""
        progress = self

        class _Handler(logging.StreamHandler):
            def emit(self, record):
                progress.clear()
                super().emit(record)

        handler = _Handler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(message)s"))
        root = logging.getLogger()
        root.handlers = [handler]


def _narrate(args) -> int:
    from .pipeline import narrate_file

    started = time.time()
    backend = get_backend(
        args.backend,
        voice=args.voice,
        lang_code=infer_lang(args.voice, args.lang),
        speed=args.speed,
        device=args.device,
    )

    line = _ProgressLine(enabled=not args.quiet)
    line.install()

    result = narrate_file(
        args.source,
        output=args.output,
        backend=backend,
        max_chars=args.max_chars,
        announce_chapters=args.announce_chapters,
        strip_boilerplate=not args.keep_boilerplate,
        strip_figures=not args.keep_figures,
        progress=line.update,
        retries=args.retries,
        on_error=args.on_error,
        omission_notice=None if args.no_omission_notice else args.omission_notice,
        log_path=args.log,
        write_log=not args.no_log,
    )
    line.clear()
    print(result.summary())
    print(f"wrote {result.output}  (in {format_duration(time.time() - started)})")
    if result.log_path:
        print(f"run log: {result.log_path}")

    if result.failures:
        _report_failures(result)
        if args.strict:
            return 1
    return 0


def _report_failures(result) -> None:
    """List the chunks that never synthesized, with where to hear the gap."""
    print(f"\n{len(result.failures)} chunk(s) could not be synthesized and were skipped:",
          file=sys.stderr)
    for failure in result.failures[:10]:
        print(f"  chunk {failure.index} (chapter {failure.chapter}): {failure.preview}",
              file=sys.stderr)
        print(f"    {failure.error}", file=sys.stderr)
    if len(result.failures) > 10:
        print(f"  ... and {len(result.failures) - 10} more — full detail in the run log",
              file=sys.stderr)
    if result.notices_spoken == len(result.failures):
        print("Each gap is announced in the narration; the MP3 is otherwise complete.",
              file=sys.stderr)
    elif result.notices_spoken:
        print(f"{result.notices_spoken} of the gaps are announced in the narration; "
              "the rest are silent.", file=sys.stderr)
    else:
        print("The MP3 is complete apart from these passages.", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
