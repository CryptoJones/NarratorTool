"""Kokoro TTS backend (hexgrad/Kokoro-82M).

Kokoro is an 82M-parameter open-weight model at 24 kHz. It is small enough that the
model load dominates startup and per-chunk synthesis is fast, so the pipeline keeps
one instance alive for the whole run.

Language codes are single letters: 'a' American English, 'b' British English,
'e' Spanish, 'f' French, 'h' Hindi, 'i' Italian, 'j' Japanese, 'p' Portuguese,
'z' Mandarin. Voice names are prefixed by language and gender — af_heart, am_michael,
bf_emma, and so on.

Non-English languages route through espeak-ng for grapheme-to-phoneme conversion, so
those need the `espeak-ng` system package. American English does not.

PROFILES
    A profile is a cast voice rather than a stock one: a weighted blend of several
    Kokoro voices, its own speed, and an optional pitch shift. `--voice nia` selects
    one; `--voice bm_george` still means the plain stock voice it always did.
"""
from __future__ import annotations

import logging
import shutil
import subprocess

import numpy as np

from .base import BackendUnavailable, SynthesisError

log = logging.getLogger(__name__)

SAMPLE_RATE = 24_000

# Cast profiles. `voices` weights must sum to 1.0 — the blend is a weighted sum of the
# per-voice style vectors, so weights that do not sum to one change the loudness and
# character of the result rather than failing.
PROFILES: dict[str, dict] = {
    # Imani Nia Baptiste, from Antigua Runners' kokoro_voice_cast.json (the copy on
    # makemake, which is authoritative — the local clone's cast is behind). Both base
    # voices are af_*, so this is an American-English profile and needs no espeak-ng.
    "nia": {
        "description": "Imani Nia Baptiste — Antigua Runners cast, for one-off books",
        "voices": {"af_nova": 0.5625, "af_heart": 0.4375},
        "speed": 0.96,
        "pitch_semitones": -0.2,
        "preserve_formants": True,
        "lang": "a",
    },
}

# CryptoJones's house narration voice, used across every OCW course and the
# Math-for-ML video: Emma (UK female) at 0.88 speed. Keep these in sync — bf_* voices
# require lang_code 'b', and loading a British voice under 'a' produces mangled
# pronunciation rather than an error.
DEFAULT_VOICE = "bf_emma"
DEFAULT_LANG = "b"
DEFAULT_SPEED = 0.88

# What a stock (non-profile) voice reads at when --speed is not given. Held separately
# from DEFAULT_SPEED so that making a *profile* the default later cannot silently
# change the speed of every stock voice.
STOCK_DEFAULT_SPEED = 0.88


def profile_for(voice: str) -> dict | None:
    """The cast profile named `voice`, or None if it is a plain Kokoro voice."""
    return PROFILES.get(voice)


def default_speed_for(voice: str) -> float:
    """A profile carries its own speed; a stock voice keeps the historical 0.88.

    Stock voices must NOT inherit the default profile's speed. Every stock voice
    defaulted to 0.88 back when bf_emma was the house voice, and `--voice bf_emma`
    with no --speed is precisely how the back catalogue gets re-rendered — handing it
    Nia's 0.96 would produce a subtly faster read that still sounds fine, which is the
    silent-wrong-output failure this module goes out of its way to avoid elsewhere.
    """
    profile = profile_for(voice)
    return float(profile["speed"]) if profile else STOCK_DEFAULT_SPEED


def default_lang_for(voice: str) -> str | None:
    """Profiles state their language outright; stock voices encode it in the prefix."""
    profile = profile_for(voice)
    return str(profile["lang"]) if profile else None


class KokoroBackend:
    def __init__(
        self,
        voice: str = DEFAULT_VOICE,
        lang_code: str = DEFAULT_LANG,
        speed: float | None = None,
        device: str | None = None,
    ) -> None:
        self.voice = voice
        self.lang_code = lang_code
        self.profile = profile_for(voice)
        self.speed = float(speed) if speed is not None else default_speed_for(voice)
        self._device = device
        self._pipeline = None  # built lazily so --list-voices etc. stay fast
        self._style = None  # blended style vector, built with the pipeline
        self._load_error: str | None = None

        if self.profile:
            self.pitch_semitones = float(self.profile["pitch_semitones"])
            self.preserve_formants = bool(self.profile["preserve_formants"])
        else:
            self.pitch_semitones = 0.0
            self.preserve_formants = True

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    @property
    def name(self) -> str:
        if self.profile and self.pitch_semitones:
            return f"kokoro:{self.voice}@{self.speed}x{self.pitch_semitones:+g}st"
        return f"kokoro:{self.voice}@{self.speed}x"

    @property
    def audio_filter(self) -> str | None:
        """An ffmpeg filter the caller must apply to the finished audio, or None.

        The pitch shift is deliberately NOT applied per chunk. Rubberband has edge
        behaviour at buffer boundaries, and a book is thousands of chunks: shifting
        each one separately would spawn thousands of processes and risk a faint seam
        at every join. The pipeline already runs one ffmpeg pass to encode the MP3,
        so the shift rides along with that — one pass, no seams, no extra processes.
        """
        if not self.pitch_semitones:
            return None
        formant = "preserved" if self.preserve_formants else "shifted"
        # ffmpeg's rubberband takes a frequency ratio, not semitones.
        ratio = 2 ** (self.pitch_semitones / 12.0)
        return f"rubberband=pitch={ratio:.6f}:formant={formant}:pitchq=quality"

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = _pick_device()
        return self._device

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        if self._load_error is not None:
            # Loading already failed once; every chunk after would fail identically.
            raise BackendUnavailable(self._load_error)
        try:
            from kokoro import KPipeline
        except ImportError as exc:
            raise ImportError(
                "Kokoro backend needs `kokoro` (pip install 'narratortool[kokoro]')"
            ) from exc

        # A profile that cannot be pitch-shifted is not the voice it claims to be, so
        # this fails at load rather than quietly rendering the whole book untreated.
        if self.audio_filter:
            self._require_rubberband()

        log.info("loading Kokoro (lang=%s, voice=%s, device=%s)",
                 self.lang_code, self.voice, self.device)
        try:
            try:
                self._pipeline = KPipeline(lang_code=self.lang_code, device=self.device)
            except TypeError:
                # Older kokoro releases do not accept `device`.
                self._pipeline = KPipeline(lang_code=self.lang_code)
        except Exception as exc:  # noqa: BLE001 - weights download, bad lang code, dead GPU
            self._load_error = (
                f"could not load Kokoro (lang={self.lang_code!r}, device={self.device}): {exc}"
            )
            raise BackendUnavailable(self._load_error) from exc

        if self.profile:
            self._style = self._blended_style()
        return self._pipeline

    def _blended_style(self):
        """The weighted sum of the profile's per-voice style vectors."""
        blended = None
        for name, weight in self.profile["voices"].items():
            try:
                vector = self._pipeline.load_single_voice(name) * float(weight)
            except Exception as exc:  # noqa: BLE001 - a typo'd voice name lands here
                self._load_error = f"profile {self.voice!r}: cannot load voice {name!r}: {exc}"
                raise BackendUnavailable(self._load_error) from exc
            blended = vector if blended is None else blended + vector
        return blended

    def _require_rubberband(self) -> None:
        if shutil.which("ffmpeg") is None:
            self._load_error = (
                f"profile {self.voice!r} needs a {self.pitch_semitones:+g} semitone shift, "
                "which requires ffmpeg"
            )
            raise BackendUnavailable(self._load_error)
        try:
            filters = subprocess.run(
                ["ffmpeg", "-hide_banner", "-filters"],
                capture_output=True, text=True, check=False,
            ).stdout
        except OSError as exc:
            self._load_error = f"could not run ffmpeg to check its filters: {exc}"
            raise BackendUnavailable(self._load_error) from exc
        if "rubberband" not in filters:
            self._load_error = (
                f"profile {self.voice!r} needs a {self.pitch_semitones:+g} semitone shift, "
                "but this ffmpeg was built without librubberband (no `rubberband` filter). "
                "Install an ffmpeg with librubberband, or pass --voice bf_emma --speed 0.88 "
                "for the legacy house voice, which needs no shift."
            )
            raise BackendUnavailable(self._load_error)

    def synthesize(self, text: str) -> np.ndarray:
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.float32)

        pipeline = self._ensure_pipeline()
        voice = self._style if self.profile else self.voice
        try:
            # KPipeline yields (graphemes, phonemes, audio) per internal segment.
            segments = [
                _to_numpy(item[2])
                for item in pipeline(text, voice=voice, speed=self.speed)
            ]
        except BackendUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced with context below
            raise SynthesisError(f"Kokoro failed on {text[:60]!r}: {exc}") from exc

        segments = [s for s in segments if s is not None and s.size]
        if not segments:
            # Not an error: some inputs (a lone symbol, a stray bullet) have nothing to
            # say. The pipeline logs it as a silent chunk and moves on.
            log.warning("Kokoro produced no audio for %r", text[:60])
            return np.zeros(0, dtype=np.float32)

        audio = np.concatenate(segments).astype(np.float32, copy=False)
        # A NaN/inf run means the model diverged on this text — worth a retry, and worth
        # failing loudly rather than writing a burst of static into the book.
        if not np.isfinite(audio).all():
            raise SynthesisError(
                f"Kokoro produced non-finite audio for {text[:60]!r}"
            )
        return audio


def _to_numpy(audio) -> np.ndarray | None:
    if audio is None:
        return None
    if hasattr(audio, "detach"):  # torch tensor
        audio = audio.detach().cpu().numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def _pick_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"  # makemake
    return "cpu"
