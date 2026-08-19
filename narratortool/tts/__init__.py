"""TTS backend registry."""
from __future__ import annotations

from .base import BackendUnavailable, SynthesisError, TTSBackend

_BACKENDS = {"kokoro": "kokoro_backend:KokoroBackend"}


def available_backends() -> list[str]:
    return sorted(_BACKENDS)


def get_backend(name: str, **kwargs) -> TTSBackend:
    target = _BACKENDS.get(name)
    if target is None:
        raise ValueError(f"unknown TTS backend {name!r}; available: {', '.join(available_backends())}")
    module_name, class_name = target.split(":")
    module = __import__(f"{__name__}.{module_name}", fromlist=[class_name])
    try:
        return getattr(module, class_name)(**kwargs)
    except TypeError as exc:  # a bad kwarg here is a caller bug worth naming clearly
        raise ValueError(f"cannot construct backend {name!r}: {exc}") from exc


__all__ = [
    "TTSBackend",
    "SynthesisError",
    "BackendUnavailable",
    "get_backend",
    "available_backends",
]
