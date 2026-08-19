"""Audio assembly and MP3 encoding.

Synthesized chunks are appended to a single raw PCM file on disk rather than held in
a list of arrays. A full-length audiobook at 24 kHz float32 would be gigabytes in RAM;
streaming to disk keeps memory flat regardless of book length, and ffmpeg encodes the
result in one pass at the end.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class FFmpegMissing(Exception):
    pass


def require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise FFmpegMissing(
            "ffmpeg is required to write MP3. Install it (apt install ffmpeg / brew install ffmpeg)."
        )
    return path


@dataclass
class Tags:
    title: str | None = None
    artist: str | None = None
    album: str | None = None
    comment: str | None = None

    def as_ffmpeg_args(self) -> list[str]:
        args: list[str] = []
        for key, value in (
            ("title", self.title),
            ("artist", self.artist),
            ("album", self.album),
            ("comment", self.comment),
        ):
            if value:
                args += ["-metadata", f"{key}={value}"]
        return args


class PCMWriter:
    """Appends mono float32 audio to a 16-bit little-endian PCM file."""

    def __init__(self, path: Path, sample_rate: int) -> None:
        self.path = Path(path)
        self.sample_rate = sample_rate
        self._fh = self.path.open("wb")
        self._samples = 0

    @property
    def duration_seconds(self) -> float:
        return self._samples / self.sample_rate if self.sample_rate else 0.0

    def append(self, audio: np.ndarray) -> None:
        if audio is None or audio.size == 0:
            return
        # Clip before converting; Kokoro occasionally overshoots [-1, 1] slightly and
        # wrapping instead of clipping turns that into a loud click.
        clipped = np.clip(audio.astype(np.float32, copy=False), -1.0, 1.0)
        self._fh.write((clipped * 32767.0).astype("<i2").tobytes())
        self._samples += clipped.size

    def append_silence(self, seconds: float) -> None:
        count = int(seconds * self.sample_rate)
        if count > 0:
            self._fh.write(np.zeros(count, dtype="<i2").tobytes())
            self._samples += count

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "PCMWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def encode_mp3(
    pcm_path: Path,
    out_path: Path,
    sample_rate: int,
    tags: Tags | None = None,
    quality: int = 2,
) -> Path:
    """Encode raw PCM to MP3. `quality` is LAME VBR 0 (best) - 9 (worst)."""
    ffmpeg = require_ffmpeg()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", str(pcm_path),
        "-codec:a", "libmp3lame", "-qscale:a", str(quality),
    ]
    if tags:
        cmd += tags.as_ffmpeg_args()
    cmd.append(str(out_path))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({result.returncode}): {result.stderr.strip()[:500]}")
    return out_path


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:d}:{secs:02d}"
