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
"""
from __future__ import annotations

import logging

import numpy as np

from .base import BackendUnavailable, SynthesisError

log = logging.getLogger(__name__)

SAMPLE_RATE = 24_000

# CryptoJones's house narration voice, used across every OCW course and the
# Math-for-ML video: Emma (UK female) at 0.88 speed. Keep these in sync — bf_* voices
# require lang_code 'b', and loading a British voice under 'a' produces mangled
# pronunciation rather than an error.
DEFAULT_VOICE = "bf_emma"
DEFAULT_LANG = "b"
DEFAULT_SPEED = 0.88


class KokoroBackend:
    def __init__(
        self,
        voice: str = DEFAULT_VOICE,
        lang_code: str = DEFAULT_LANG,
        speed: float = DEFAULT_SPEED,
        device: str | None = None,
    ) -> None:
        self.voice = voice
        self.lang_code = lang_code
        self.speed = speed
        self._device = device
        self._pipeline = None  # built lazily so --list-voices etc. stay fast
        self._load_error: str | None = None

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    @property
    def name(self) -> str:
        return f"kokoro:{self.voice}@{self.speed}x"

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
        return self._pipeline

    def synthesize(self, text: str) -> np.ndarray:
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.float32)

        pipeline = self._ensure_pipeline()
        try:
            # KPipeline yields (graphemes, phonemes, audio) per internal segment.
            segments = [
                _to_numpy(item[2])
                for item in pipeline(text, voice=self.voice, speed=self.speed)
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
