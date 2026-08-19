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
(`narra-\ntion`), bare page numbers, dot leaders from tables of contents, Project
Gutenberg's legal boilerplate, and the text inside figures
(see [Images, diagrams and figures](#images-diagrams-and-figures)).

**chunk** splits on sentence boundaries, never mid-sentence. A chunk edge inside a
sentence is audible — the model drops the falling intonation of a sentence ending, so
the seam between two audio segments clicks. Abbreviations (`Dr.`, `e.g.`) are protected
from being read as sentence ends, and dialogue punctuation is handled: `"Stop!" she
cried.` is one sentence, not two.

**synthesize** runs each chunk through the TTS backend, retrying a chunk that fails
(see [When the narrator hiccups](#when-the-narrator-hiccups)).

**concatenate** streams samples to a raw PCM file on disk rather than accumulating
arrays in memory — a full audiobook at 24 kHz float32 would be gigabytes in RAM, so
this keeps memory flat regardless of book length.

**encode** hands the PCM to ffmpeg once, writing a VBR MP3 tagged with the document's
title and author.

## Images, diagrams and figures

Images are never narrated — there is no text in them to extract. What *is* a problem is
the text drawn **inside** a diagram. A PDF's text layer contains everything on the page,
so pypdf hands back a chart's axis ticks, axis names and legend keys as loose fragments,
and an unfiltered narration reads them out:

> *"…the results are summarised below. Accuracy one point zero, zero point eight, zero
> point six, zero point four, zero point two, zero ten twenty thirty forty, Epochs,
> baseline, ours, ablation. Figure three point two colon, accuracy versus epochs…"*

`normalize` suppresses three things:

| | Recognised by | Example |
|---|---|---|
| Captions | a figure label | `Figure 3.2: Accuracy versus epochs`, `Fig. 4 —`, `Table 2.` |
| Tick numbers | short lines that are all numbers and symbols | `0.2 0.4 0.6 0.8`, `10%  20%  30%` |
| Axis names, legend keys | a bare word or two sitting *against* the tick numbers | `Accuracy`, `Epochs`, `baseline` |

The third is the interesting one. `Accuracy` and `Epochs` are ordinary words — nothing
about them reads as chart text on its own, and a rule broad enough to catch them in
isolation would eat every heading in the book. What gives them away is position:
immediately against a run of tick numbers, with no blank line between. Only the numbers
seed that sweep, never a caption, so a section heading following a caption survives; the
sweep is capped at six lines per side so a stray tick line cannot eat a column of prose.

Everything here errs towards keeping a line, because the failure modes are not
symmetric: a wrongly kept axis label is a moment of nonsense, while a wrongly dropped
line is content the listener never learns was there. So prose that merely mentions
numbers is safe — `In 1901 he sailed for two years`, `January 1, 1901`, `Page 3 of 12`
— as is `Figure 3 shows that accuracy improves`, which is a sentence, not a caption.

Check what a document would lose before committing to a long run:

```bash
narrate paper.pdf --dry-run     # lists the suppressed lines, narrates nothing
narrate paper.pdf --keep-figures  # read them aloud after all
```

The count and the first fifty suppressed lines are recorded in the run log, so an
omission is auditable after the fact as well as before it.

The other formats need less of this: EPUB and HTML drop `<figure>`, `<table>` and image
alt text at extraction, Markdown strips `![…](…)`, and DOCX images and text boxes never
reach the paragraph stream. PDF is where the leakage happens.

## When the narrator hiccups

Neural TTS is not perfectly reliable over thousands of chunks: a fragment of odd
punctuation trips the phonemizer, CUDA drops an allocation, the model returns an empty
tensor. A four-hour narration should not be lost to one bad paragraph, so failures are
contained rather than fatal.

A chunk that fails is retried (`--retries`, default 2), then retried **one sentence at
a time** — most hiccups are a single hostile fragment, and keeping the other sentences
beats dropping the paragraph. Only if that also fails is the chunk given up on: by
default it is skipped and narration continues, and every skip is reported at the end
with its text and its position.

**A skipped passage is announced aloud.** The narrator says *"Passage Omitted."* in her
own voice where the audio would otherwise contain an unremarkable gap — a silent seam
is indistinguishable from a pause in the reading, so an omission that nobody notices is
the worst outcome. The marker is synthesized once and reused, so a clean run pays
nothing for it, and if the marker itself will not synthesize the gap falls back to
silence rather than failing the run.

```bash
narrate book.epub --retries 4         # more patient with a flaky GPU
narrate book.epub --on-error abort    # stop at the first unrecoverable chunk instead
narrate book.epub --strict            # exit non-zero if anything was skipped (for CI)
narrate book.epub --omission-notice "Text missing."   # say something else
narrate book.epub --no-omission-notice                # leave the gap silent
```

Whatever stops a run — `--on-error abort`, Ctrl-C, a full disk, a dead backend — the
audio produced so far is encoded to `book.partial.mp3` instead of being thrown away
with the temp directory. Failures that retrying cannot fix stop the run immediately:
a backend that will not load, and five consecutive chunk failures, which means the
engine is broken rather than the text.

Two silent-corruption cases are treated as errors rather than shipped: non-finite
samples from a diverged model (which cast to int16 as a burst of full-scale static) are
replaced with silence and the chunk retried, and a run where *every* chunk came out
empty fails loudly instead of writing a silent MP3 that looks like success.

## The run log

Every narration writes a JSONL sidecar next to the MP3 — `book.narration.jsonl` — with
one record per chunk plus a header and footer. Nobody listens to four hours of audio to
confirm it came out clean, so each record carries the offset into the finished MP3
where that chunk starts, which makes any claim about the output checkable by seeking:

```json
{"event": "chunk", "index": 412, "chapter": 7, "chars": 431, "text": "…",
 "status": "skipped", "attempts": 4, "offset_seconds": 5218.44, "error": "…"}
```

The header record carries `figure_lines_suppressed` and the first fifty of those lines.
`status` is `ok`, `retried`, `recovered` (rebuilt sentence-by-sentence, so possibly
missing one), `silent` (nothing to say), or `skipped` (lost). A skipped record also
carries `notice` — whether the omission was announced aloud — and `notice_seconds`, the
length of that announcement. The footer records totals and the indexes of every skipped
chunk.

```bash
jq -r 'select(.status=="skipped") | "\(.offset_seconds)s  \(.text)"' book.narration.jsonl
jq -s 'map(select(.event=="chunk")) | group_by(.status) | map({(.[0].status): length}) | add' book.narration.jsonl
```

Records are flushed as they are written, so the log of a killed run is still usable —
`narratortool.runlog.read_log()` reads one back, tolerating a truncated final line.
Use `--log PATH` to put it elsewhere, or `--no-log` to skip it.

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

The test suite covers text processing, extraction, and the failure paths — retries,
skips, aborts, partial salvage, and the run log — using a fake backend, so it needs
neither the TTS model nor a GPU and runs fast in CI. `tests/test_reliability.py` is
where a new failure mode gets pinned down.

## License

MIT. Copyright CryptoJones <cryptojones@owasp.org>.
