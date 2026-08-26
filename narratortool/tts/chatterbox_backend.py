"""Chatterbox TTS backend (Resemble AI, MIT licence).

Chatterbox is a 0.5B zero-shot cloning model at 24 kHz. It clones from a short
reference clip rather than selecting a named voice, so `--voice` here names a
reference WAV instead of a Kokoro style vector. The house reference is a banked
bf_emma clip, which is what keeps the back catalogue sounding like itself across
the engine change.

It was chosen over Kokoro for **inter-take voice consistency** — the same speaker
measured 7.3% f0 IQR across takes against VibeVoice-1.5B's 19.7%. Over a book of
several thousand chunks that is the number that decides whether the narrator sounds
like one person.

Three engine quirks are handled here, all of them learned the expensive way on a
3,563-cue render:

  * **Short inputs come out as the wrong words entirely.** A one-word input is
    misread 79% of the time, two words 51%, three 33% — and the model is fluent and
    confident about it, so "Preface" becomes a sentence of unrelated speech rather
    than anything that reads as an error. Temperature does not help and neither does
    seed hunting; the model simply has too little context to work from. See
    `_CARRIER` below for the fix.
  * **There is no speed parameter.** Pacing is done as a time stretch on the finished
    audio via `audio_filter`, in the same single ffmpeg pass that encodes the MP3.
  * **`perth` 1.0.0 exports `PerthImplicitWatermarker` as None**, and Chatterbox
    imports it unconditionally, so the model will not load without the patch in
    `_import_chatterbox`.

Chatterbox's habit of writing float32 WAVs (which breaks anything reading with the
`wave` module) does not reach us: this backend hands numpy straight to `PCMWriter`,
which converts to 16-bit itself. Nothing here ever writes a WAV.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .base import BackendUnavailable, SynthesisError

log = logging.getLogger(__name__)

SAMPLE_RATE = 24_000

# Reference clips shipped with the package. A profile name here beats a path, so
# `--voice house` works from any directory.
VOICE_DIR = Path(__file__).resolve().parent.parent / "voices"
DEFAULT_VOICE = "house"

# Chatterbox reads at 17.12 characters per second, measured over the Neuromancer
# render. Kokoro reads 15.5 at 1.0x, so a pace check or duration estimate calibrated
# for one engine is wrong for the other by a third.
CHARS_PER_SECOND = 17.12

# The finished audio is stretched to this by `audio_filter`. Chatterbox at 1.0x is
# faster than the house Kokoro read (0.88x of 15.5 c/s = 13.6 c/s), and 0.9x is the
# stretch CJ listened to a full episode at and called perfect. Matching the legacy
# pace exactly would want roughly 0.8; that is available as --speed and deliberately
# not the default, because 0.9 is the value with a listener behind it.
DEFAULT_SPEED = 0.9

# Generation settings as shipped on the Neuromancer render. repetition_penalty is
# raised from the library default of 1.2: at 1.2 the model loops syllables on long
# chunks, which is exactly the shape of input a book is made of.
DEFAULT_EXAGGERATION = 0.5
DEFAULT_CFG_WEIGHT = 0.5
DEFAULT_TEMPERATURE = 0.8
DEFAULT_REPETITION_PENALTY = 1.4
DEFAULT_SEED = 42

# --- the carrier phrase -------------------------------------------------------------
#
# A short input is misread because the model has no context to condition on. Giving it
# a sentence to say and then cutting that sentence back off is the fix.
#
# The carrier ends in a full stop so that the model puts a real pause between it and
# the target text. That pause is what `_trim_carrier` looks for — it is how the join is
# found without running an ASR pass to get word timestamps.
#
# MEASURED, on the 12 short inputs a book actually produces (chapter headings and the
# omission notice), transcribed back with Whisper:
#
#   no carrier                                   7/12
#   "He looked at me and said."                 10/12
#   "The next section of the book is titled."   11/12   <- this one
#
# The last two are one item apart on a corpus of twelve, which is not a real margin.
# The tiebreak is that this carrier frames a heading and the other frames speech, and
# the inputs here are headings; it also fixed both of the other's failures. Retuning it
# on a larger corpus is #6.
#
# DO NOT add a trailing carrier as well. Tried, and it scored 5/12 — worse than no
# carrier at all. Bracketing the target made the model swallow it (renders of 0.07-0.12
# seconds), and one degenerated into a loop of "bye-bye". The right context does not
# help; it crowds the target out.

_CARRIER = "The next section of the book is titled."
# Inputs shorter than this many words get the carrier; anything at or above it carries
# enough of its own context and is rendered plain. The measured misread rate is 9% at
# four words and 1% at five, so five is where the carrier stops earning its cost.
_CARRIER_BELOW_WORDS = 5

# Where to look for the pause, as a multiple of the carrier's estimated duration. The
# estimate is only good to within about a third, so the window is generous; it is
# narrow enough that a pause between the target's own sentences cannot be mistaken for
# the join.
_JOIN_WINDOW = (0.55, 1.9)
_FRAME_SECONDS = 0.01
# A frame is silence at this fraction of the clip's robust peak. Chatterbox's pauses
# are true near-silence, so this does not need to be generous.
_SILENCE_RATIO = 0.08
# Cut this far back into the pause rather than exactly at the word onset. The onset is
# where the estimate is least certain, and a hair of leading silence is inaudible where
# a clipped first consonant is not.
_ONSET_BACKOFF_SECONDS = 0.03
# Below this, the trim took everything and something went wrong; treat it as a failed
# chunk so the pipeline retries rather than writing a click into the book.
_MIN_TRIMMED_SECONDS = 0.05


def resolve_voice(voice: str) -> Path:
    """A bundled reference name, or a path to a WAV.

    Names win over paths so that `--voice house` means the same thing regardless of
    the working directory, and so a stray `house.wav` nearby cannot shadow it.
    """
    bundled = VOICE_DIR / f"{voice}.wav"
    if bundled.is_file():
        return bundled
    path = Path(voice).expanduser()
    if path.is_file():
        return path
    known = sorted(p.stem for p in VOICE_DIR.glob("*.wav")) if VOICE_DIR.is_dir() else []
    raise BackendUnavailable(
        f"no reference clip for voice {voice!r}: not a bundled reference "
        f"({', '.join(known) or 'none installed'}) and not a readable file"
    )


def available_voices() -> list[str]:
    return sorted(p.stem for p in VOICE_DIR.glob("*.wav")) if VOICE_DIR.is_dir() else []


def atempo_chain(speed: float) -> str | None:
    """An ffmpeg filter chain that plays the audio at `speed`, pitch preserved.

    ffmpeg's atempo only accepts 0.5-2.0, so anything outside that is reached by
    chaining stages — 0.4 becomes 0.5 then 0.8. Returns None at 1.0, so the encode
    stays a plain copy when no stretch is wanted.
    """
    if abs(speed - 1.0) < 1e-6:
        return None
    if speed <= 0:
        raise ValueError(f"speed must be positive, got {speed}")

    stages: list[float] = []
    remaining = float(speed)
    while remaining < 0.5:
        stages.append(0.5)
        remaining /= 0.5
    while remaining > 2.0:
        stages.append(2.0)
        remaining /= 2.0
    stages.append(remaining)
    return ",".join(f"atempo={s:.6f}" for s in stages)


def _frame_levels(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    """Per-frame RMS, and the frame length in samples."""
    frame = max(1, int(sample_rate * _FRAME_SECONDS))
    usable = (audio.size // frame) * frame
    if usable == 0:
        return np.zeros(0, dtype=np.float32), frame
    frames = audio[:usable].reshape(-1, frame).astype(np.float32)
    return np.sqrt((frames * frames).mean(axis=1)), frame


def _longest_silent_run(silent: np.ndarray, lo: int, hi: int) -> tuple[int, int] | None:
    """The longest run of silent frames within [lo, hi), as (start, end) or None."""
    best: tuple[int, int] | None = None
    best_len = 0
    run_start = None
    for index in range(lo, hi):
        if silent[index]:
            if run_start is None:
                run_start = index
        elif run_start is not None:
            if index - run_start > best_len:
                best_len, best = index - run_start, (run_start, index)
            run_start = None
    if run_start is not None and hi - run_start > best_len:
        best = (run_start, hi)
    return best


def _last_speech_onset(levels: np.ndarray, frame: int, total_samples: int) -> int:
    """Where the final run of speech begins, in samples, or `total_samples` if unknown.

    A ceiling on the fallback cut, so an estimate that overshoots a short render cannot
    take the target text with it along with the carrier.

    Returns `total_samples` — i.e. no ceiling — when the final run reaches back to the
    start of the clip. There is no boundary to protect in that case, and clamping to
    zero would block the trim entirely and leave the carrier audible.
    """
    if not levels.size:
        return total_samples
    peak = float(np.percentile(levels, 95))
    spoken = levels >= peak * _SILENCE_RATIO
    if not spoken.any():
        return total_samples
    index = int(np.flatnonzero(spoken)[-1])
    while index > 0 and spoken[index - 1]:
        index -= 1
    return index * frame if index > 0 else total_samples


def _trim_carrier(audio: np.ndarray, sample_rate: int, carrier_seconds: float) -> np.ndarray:
    """Cut the carrier phrase off the front of `audio`.

    The carrier ends in a full stop, so the model leaves a pause before the target
    text. Finding the longest pause in a window around where the carrier is expected
    to end locates the join without an ASR pass — which matters, because the whole
    point of the carrier is to avoid depending on anything heavier than the model
    that is already loaded.

    Falls back to cutting at the estimate when no pause is found. That is worse than
    a located cut but far better than leaving the carrier audible, which would put
    "He looked at me and said" in front of every chapter heading in the book.
    """
    levels, frame = _frame_levels(audio, sample_rate)
    estimate = int(carrier_seconds * sample_rate)
    found = None

    if levels.size:
        peak = float(np.percentile(levels, 95))
        silent = levels < peak * _SILENCE_RATIO
        spoken = np.flatnonzero(~silent)
        lo = min(levels.size, max(0, int(carrier_seconds * _JOIN_WINDOW[0] / _FRAME_SECONDS)))
        hi = min(levels.size, int(carrier_seconds * _JOIN_WINDOW[1] / _FRAME_SECONDS))
        if spoken.size:
            # Stop before the trailing silence. A short target leaves more silence
            # after itself than the carrier's full stop leaves before it, so without
            # this the tail wins on length and the cut lands past the end of the word
            # — trimming the target away along with the carrier. Found on 'Notes',
            # where it took the whole chunk.
            hi = min(hi, int(spoken[-1]))
        found = _longest_silent_run(silent, lo, hi) if hi > lo else None

    if found is None:
        log.debug("no pause found after the carrier; cutting at the %.2fs estimate",
                  carrier_seconds)
        # The estimate is good to about a third, so on a short render it can overshoot
        # the whole clip. Never let it cut past the start of the last run of speech:
        # a little carrier left audible is recoverable, a chunk trimmed to nothing is
        # a heading the listener never hears.
        cut = min(estimate, _last_speech_onset(levels, frame, audio.size))
    else:
        run_start, run_end = found
        backoff = int(_ONSET_BACKOFF_SECONDS / _FRAME_SECONDS)
        cut = max(run_start, run_end - backoff) * frame

    return audio[cut:] if cut < audio.size else np.zeros(0, dtype=np.float32)


def _import_chatterbox():
    """Import ChatterboxTTS, working around perth 1.0.0.

    perth 1.0.0 exports `PerthImplicitWatermarker` as None. Chatterbox imports the
    name and calls it, so the model fails to load with a bare `TypeError: 'NoneType'
    object is not callable` that says nothing about watermarking. Pointing the name at
    the library's own no-op watermarker is the documented workaround.
    """
    try:
        import perth
    except ImportError:
        perth = None
    if perth is not None and getattr(perth, "PerthImplicitWatermarker", None) is None:
        dummy = getattr(perth, "DummyWatermarker", None)
        if dummy is not None:
            perth.PerthImplicitWatermarker = dummy
            log.debug("patched perth.PerthImplicitWatermarker to DummyWatermarker")

    try:
        from chatterbox.tts import ChatterboxTTS
    except ImportError as exc:
        raise BackendUnavailable(
            "Chatterbox backend needs `chatterbox-tts` "
            "(pip install 'narratortool[chatterbox]')"
        ) from exc
    return ChatterboxTTS


class ChatterboxBackend:
    def __init__(
        self,
        voice: str = DEFAULT_VOICE,
        speed: float | None = None,
        device: str | None = None,
        exaggeration: float = DEFAULT_EXAGGERATION,
        cfg_weight: float = DEFAULT_CFG_WEIGHT,
        temperature: float = DEFAULT_TEMPERATURE,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        seed: int = DEFAULT_SEED,
        lang_code: str | None = None,
    ) -> None:
        # lang_code is accepted and ignored: the CLI infers one for Kokoro and passes
        # it to whichever backend is selected. Chatterbox has no language codes.
        self.voice = voice
        self.speed = float(speed) if speed is not None else DEFAULT_SPEED
        self.exaggeration = exaggeration
        self.cfg_weight = cfg_weight
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.seed = seed
        self._device = device
        self._model = None  # built lazily so --list-voices etc. stay fast
        self._load_error: str | None = None

        # Resolved eagerly: a missing reference clip is a typo worth reporting before
        # the run starts, not after several minutes of model download.
        self.reference = resolve_voice(voice)

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    @property
    def name(self) -> str:
        return f"chatterbox:{self.voice}@{self.speed}x"

    @property
    def audio_filter(self) -> str | None:
        """The time stretch, applied by the pipeline's single MP3 encode pass.

        Chatterbox has no speed parameter, so pacing has to be a post-process. Doing
        it per chunk would mean thousands of ffmpeg invocations and a possible seam at
        every join; the encode pass already runs over the whole book, so the stretch
        rides along with it for free.
        """
        return atempo_chain(self.speed)

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = _autodetect_device()
        return self._device

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        if self._load_error:
            raise BackendUnavailable(self._load_error)

        ChatterboxTTS = _import_chatterbox()
        log.info("loading Chatterbox (device=%s, reference=%s)", self.device, self.reference)
        try:
            model = ChatterboxTTS.from_pretrained(device=self.device)
            model.prepare_conditionals(str(self.reference), exaggeration=self.exaggeration)
        except Exception as exc:  # noqa: BLE001 - missing weights, no VRAM, bad clip
            self._load_error = (
                f"could not load Chatterbox (device={self.device}, "
                f"reference={self.reference}): {exc}"
            )
            raise BackendUnavailable(self._load_error) from exc
        self._model = model
        return model

    def synthesize(self, text: str) -> np.ndarray:
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.float32)

        model = self._ensure_model()
        carried = len(text.split()) < _CARRIER_BELOW_WORDS
        prompt = f"{_CARRIER} {text}" if carried else text

        try:
            self._seed_rng()
            wav = model.generate(
                prompt,
                exaggeration=self.exaggeration,
                cfg_weight=self.cfg_weight,
                temperature=self.temperature,
                repetition_penalty=self.repetition_penalty,
            )
        except BackendUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - any generation fault is retryable
            raise SynthesisError(f"Chatterbox failed on {text[:60]!r}: {exc}") from exc

        audio = _to_numpy(wav)
        if audio.size == 0:
            raise SynthesisError(f"Chatterbox returned no audio for {text[:60]!r}")

        if carried:
            audio = _trim_carrier(audio, SAMPLE_RATE, len(_CARRIER) / CHARS_PER_SECOND)
            if audio.size < _MIN_TRIMMED_SECONDS * SAMPLE_RATE:
                raise SynthesisError(
                    f"carrier trim left nothing speakable for {text[:60]!r}"
                )
        return audio

    def _seed_rng(self) -> None:
        """Fix the seed per chunk so a rerun of the same book is the same audio."""
        try:
            import torch

            torch.manual_seed(self.seed)
        except ImportError:  # pragma: no cover - torch is a chatterbox dependency
            pass


def _autodetect_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _to_numpy(wav) -> np.ndarray:
    """Chatterbox returns a torch tensor shaped (1, samples); the pipeline wants mono."""
    if hasattr(wav, "detach"):
        wav = wav.detach().to("cpu").numpy()
    return np.asarray(wav, dtype=np.float32).reshape(-1)
