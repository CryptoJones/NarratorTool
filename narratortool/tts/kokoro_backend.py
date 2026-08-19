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

from .base import SynthesisError

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
        try:
            from kokoro import KPipeline
        except ImportError as exc:
            raise ImportError(
                "Kokoro backend needs `kokoro` (pip install 'narratortool[kokoro]')"
            ) from exc

        log.info("loading Kokoro (lang=%s, voice=%s, device=%s)",
                 self.lang_code, self.voice, self.device)
        try:
            self._pipeline = KPipeline(lang_code=self.lang_code, device=self.device)
        except TypeError:
            # Older kokoro releases do not accept `device`.
            self._pipeline = KPipeline(lang_code=self.lang_code)
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
        except Exception as exc:  # noqa: BLE001 - surfaced with context below
            raise SynthesisError(f"Kokoro failed on {text[:60]!r}: {exc}") from exc

        segments = [s for s in segments if s is not None and s.size]
        if not segments:
            log.warning("Kokoro produced no audio for %r", text[:60])
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(segments).astype(np.float32, copy=False)


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
