"""Command-line interface."""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from . import __version__
from .audio import FFmpegMissing, format_duration
from .extract import UnsupportedFormat, supported_extensions
from .textproc import chunk, normalize
from .tts import available_backends, get_backend
from .tts.kokoro_backend import DEFAULT_LANG, DEFAULT_SPEED, DEFAULT_VOICE

# The house voice, and the rest of the Kokoro set worth knowing about.
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
    voice.add_argument("--voice", default=DEFAULT_VOICE, help=f"Kokoro voice (default: {DEFAULT_VOICE})")
    voice.add_argument("--speed", type=float, default=DEFAULT_SPEED,
                       help=f"speech rate (default: {DEFAULT_SPEED})")
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
    """
    if explicit:
        return explicit
    return voice[0] if voice and voice[0].isalpha() else DEFAULT_LANG


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
    )

    if args.list_voices:
        for name, desc in KNOWN_VOICES.items():
            marker = " *" if name == DEFAULT_VOICE else "  "
            print(f"{marker} {name:<16} {desc}")
        print("\n* = default. Prefix: a=American, b=British; f=female, m=male.")
        return 0

    if args.source is None:
        build_parser().print_help()
        return 2

    try:
        if args.dry_run:
            return _dry_run(args)
        return _narrate(args)
    except (UnsupportedFormat, FFmpegMissing, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


def _dry_run(args) -> int:
    from .extract import extract

    doc = extract(args.source)
    total = 0
    print(f"title   : {doc.title}")
    print(f"author  : {doc.author or '(unknown)'}")
    print(f"chapters: {len(doc.chapters)}")
    for i, ch in enumerate(doc.chapters):
        pieces = chunk(normalize(ch.text, not args.keep_boilerplate), args.max_chars)
        total += len(pieces)
        label = ch.title or f"(untitled {i})"
        print(f"  {i:>3}. {label[:58]:<58} {len(ch.text):>8,} chars  {len(pieces):>5} chunks")
    print(f"\ntotal chunks: {total:,}")
    # Kokoro on a 4060 runs comfortably faster than realtime; this is a rough floor.
    print(f"estimated audio: ~{format_duration(total * 4.0)} (at ~4s per chunk)")
    return 0


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

    def progress(index: int, total: int, text: str) -> None:
        if args.quiet:
            return
        pct = 100 * index / total
        preview = text[:48].replace("\n", " ")
        print(f"\r[{index:>5}/{total}] {pct:5.1f}%  {preview:<48}", end="", flush=True)

    result = narrate_file(
        args.source,
        output=args.output,
        backend=backend,
        max_chars=args.max_chars,
        announce_chapters=args.announce_chapters,
        strip_boilerplate=not args.keep_boilerplate,
        progress=progress,
    )
    if not args.quiet:
        print()
    print(result.summary())
    print(f"wrote {result.output}  (in {format_duration(time.time() - started)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
