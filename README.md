# NarratorTool

Narrate documents to MP3 with local neural TTS. Point it at a `.txt`, `.md`, `.pdf`,
`.epub`, `.docx`, or `.html` file and get back a single narrated audio file.

Everything runs locally. No API keys, no per-character billing, no upload of whatever
you happen to be narrating.

```bash
narrate book.epub                      # -> book.mp3
narrate paper.pdf -o out/paper.mp3
narrate notes.md --voice bm_george --speed 1.0
narrate huge.txt --dry-run             # parse + chunk, report stats, synthesize nothing
```

## Quick start

```bash
./INSTALL.sh                 # venv + package + extras, then prints how to narrate
.venv/bin/narrate book.epub  # -> book.epub's text, read aloud, as book.mp3
```

`narrate` lives inside the project's `.venv`, so the bare name only works in a shell
where that venv is active. Either call it by path as above, or activate once:

```bash
source .venv/bin/activate
narrate book.epub
```

To type `narrate` from anywhere without activating, run `./INSTALL.sh --link`, which
symlinks it into `~/.local/bin`.

## Voice

The default is **`bf_emma` at speed 0.88** — Emma, UK female. This is the house
narration voice used across the OpenCourseWare courses and the Math-for-ML video, so
new narrations match the existing catalogue without passing any flags.

`narrate --list-voices` shows the rest. Voice names encode their language: the first
letter is the language (`a` American, `b` British) and the second the gender (`f`, `m`).

**The language code is inferred from the voice prefix**, so `--voice bm_george` selects
British automatically. This matters because getting it wrong is *silent* — a British
voice under an American language code still produces audio, just with the wrong
phonemes. Override with `--lang` only if you know you need to.

## Install

```bash
./INSTALL.sh
```

The installer picks an interpreter Kokoro supports, builds `.venv` against it (using
`uv` if you have it), installs the package with every extra, checks for `ffmpeg` and
`espeak-ng`, and finishes by printing the exact command to narrate a file.

| Flag | Effect |
|---|---|
| `--extras kokoro` | TTS + plain text only, instead of everything |
| `--extras pdf,epub` | parsers only — skips the torch download |
| `--dev` | also install `pytest` and `ruff` |
| `--link` | symlink `narrate` into `~/.local/bin` |
| `--recreate` | rebuild `.venv` from scratch |

It is safe to re-run: an existing `.venv` is reused unless you pass `--recreate`.

### By hand

```bash
pip install -e ".[all]"      # every format + Kokoro
pip install -e ".[kokoro]"   # TTS + plain text only
```

Parsers are extras, so a text-only install does not drag in `pypdf`, `ebooklib`, or
`python-docx`. Install just what you need:

| Extra | Enables |
|---|---|
| `kokoro` | speech synthesis (pulls torch) |
| `pdf` | `.pdf` |
| `epub` | `.epub` |
| `docx` | `.docx` |
| `html` | `.html`, and better `.epub` chapter text |

**Python version:** the parsers and CLI run on any Python ≥ 3.10, but the `kokoro`
extra requires **`>=3.10,<3.13`** — Kokoro publishes no wheels for 3.13+. `INSTALL.sh`
picks a supported interpreter for you and refuses to build against one that is too new.
Doing it by hand on a system Python that is too new:

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
```

**System packages:** `ffmpeg` for MP3 encoding, and `espeak-ng` for any non-American
voice — including the default British one — because Kokoro falls back to espeak for
grapheme-to-phoneme on out-of-dictionary words.

```bash
apt install ffmpeg espeak-ng      # or: brew install ffmpeg espeak-ng
```

## How it works

```
extract  ->  normalize  ->  chunk  ->  synthesize  ->  concatenate  ->  encode
```

**extract** dispatches on file extension to a per-format parser, producing a common
`Document` of chapters. EPUB reads the spine in reading order; PDF maps its outline
onto page ranges; DOCX uses Word's Heading styles.

**normalize** removes what sounds wrong read aloud: PDF line-break hyphenation
(`narra-\ntion`), bare page numbers, dot leaders from tables of contents, and Project
Gutenberg's legal boilerplate.

**chunk** splits on sentence boundaries, never mid-sentence. A chunk edge inside a
sentence is audible — the model drops the falling intonation of a sentence ending, so
the seam between two audio segments clicks. Abbreviations (`Dr.`, `e.g.`) are protected
from being read as sentence ends, and dialogue punctuation is handled: `"Stop!" she
cried.` is one sentence, not two.

**synthesize** runs each chunk through the TTS backend.

**concatenate** streams samples to a raw PCM file on disk rather than accumulating
arrays in memory — a full audiobook at 24 kHz float32 would be gigabytes in RAM, so
this keeps memory flat regardless of book length.

**encode** hands the PCM to ffmpeg once, writing a VBR MP3 tagged with the document's
title and author.

## Backends

Kokoro ([hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)) is the default
and currently the only implementation. The backend is a small protocol —
`sample_rate`, `name`, `synthesize(text) -> np.ndarray` — so another engine is a single
new class plus a registry entry.

## Scanned PDFs

A PDF with no text layer raises `NoTextLayer` rather than silently producing an empty
narration. OCR it first:

```bash
ocrmypdf scanned.pdf searchable.pdf && narrate searchable.pdf
```

`.doc` (pre-2007 binary Word) is not supported — convert with
`libreoffice --headless --convert-to docx file.doc`.

## Development

```bash
./INSTALL.sh --dev
.venv/bin/pytest
```

The test suite covers text processing and extraction without needing the TTS model
installed, so it runs fast and in CI.

## License

MIT. Copyright CryptoJones <cryptojones@owasp.org>.
