"""TTS backend registry."""
from __future__ import annotations

from types import ModuleType

from .base import BackendUnavailable, SynthesisError, TTSBackend

_BACKENDS = {
    "chatterbox": "chatterbox_backend:ChatterboxBackend",
    "kokoro": "kokoro_backend:KokoroBackend",
}

# Chatterbox holds a voice steady across takes roughly three times better than the
# alternatives measured, which over a book of thousands of chunks is what decides
# whether the narrator sounds like one person. Kokoro stays available: it is far
# lighter, it needs no GPU, and it is what the back catalogue was rendered with.
DEFAULT_BACKEND = "chatterbox"


def available_backends() -> list[str]:
    return sorted(_BACKENDS)


def _module(name: str) -> ModuleType:
    target = _BACKENDS.get(name)
    if target is None:
        raise ValueError(
            f"unknown TTS backend {name!r}; available: {', '.join(available_backends())}"
        )
    module_name, _, _ = target.partition(":")
    return __import__(f"{__name__}.{module_name}", fromlist=["_"])


def default_voice(name: str) -> str:
    """The voice a backend uses when none is given.

    Backends do not share a voice namespace — Kokoro's are style vectors it ships,
    Chatterbox's are reference clips to clone from — so `--voice` cannot have one
    default across both. Asking the selected backend is what keeps `--backend kokoro`
    from being handed a Chatterbox reference name and failing on it.
    """
    return str(_module(name).DEFAULT_VOICE)


def chars_per_second(name: str) -> float:
    """Characters of text the backend speaks per second at 1.0x.

    Engines differ by a third here, which is enough to make a duration estimate
    calibrated for one useless for the other.
    """
    return float(_module(name).CHARS_PER_SECOND)


def get_backend(name: str, **kwargs) -> TTSBackend:
    module = _module(name)
    class_name = _BACKENDS[name].partition(":")[2]
    try:
        return getattr(module, class_name)(**kwargs)
    except TypeError as exc:  # a bad kwarg here is a caller bug worth naming clearly
        raise ValueError(f"cannot construct backend {name!r}: {exc}") from exc


__all__ = [
    "TTSBackend",
    "SynthesisError",
    "BackendUnavailable",
    "DEFAULT_BACKEND",
    "get_backend",
    "available_backends",
    "default_voice",
    "chars_per_second",
]
