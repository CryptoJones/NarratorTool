"""TTS backend interface.

Backends are swappable so the pipeline does not care which engine renders audio.
Kokoro is the default; the fleet also runs VibeVoice on pluto, which is a natural
second implementation of this same protocol.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class TTSBackend(Protocol):
    """Renders text to mono float32 audio in [-1, 1]."""

    @property
    def sample_rate(self) -> int: ...

    @property
    def name(self) -> str: ...

    def synthesize(self, text: str) -> np.ndarray:
        """Return mono float32 samples for `text`. Must never return None; an
        unspeakable input should return an empty array rather than raise."""
        ...


class SynthesisError(Exception):
    pass
