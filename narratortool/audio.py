"""Audio assembly and MP3 encoding.

Synthesized chunks are appended to a single raw PCM file on disk rather than held in
a list of arrays. A full-length audiobook at 24 kHz float32 would be gigabytes in RAM;
streaming to disk keeps memory flat regardless of book length, and ffmpeg encodes the
result in one pass at the end.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class FFmpegMissing(Exception):
    pass


class AudioWriteError(Exception):
    """Raised when audio cannot be written or encoded (full disk, bad PCM, ffmpeg)."""


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
        try:
            self._fh = self.path.open("wb")
        except OSError as exc:
            raise AudioWriteError(f"cannot open {self.path} for writing: {exc}") from exc
        self._samples = 0
        self.nonfinite_chunks = 0

    @property
    def duration_seconds(self) -> float:
        return self._samples / self.sample_rate if self.sample_rate else 0.0

    def append(self, audio: np.ndarray) -> int:
        """Write `audio`, returning the number of samples written."""
        if audio is None or audio.size == 0:
            return 0
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)

        # A model that goes off the rails emits NaN/inf rather than raising. Casting
        # those to int16 is undefined and lands as full-scale noise in the MP3, so they
        # are zeroed here — silence is a recoverable defect, a burst of static is not.
        if not np.isfinite(audio).all():
            bad = int((~np.isfinite(audio)).sum())
            self.nonfinite_chunks += 1
            log.warning("replacing %d non-finite sample(s) with silence", bad)
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        # Clip before converting; Kokoro occasionally overshoots [-1, 1] slightly and
        # wrapping instead of clipping turns that into a loud click.
        clipped = np.clip(audio, -1.0, 1.0)
        self._write((clipped * 32767.0).astype("<i2").tobytes())
        self._samples += clipped.size
        return int(clipped.size)

    def append_silence(self, seconds: float) -> int:
        count = int(seconds * self.sample_rate)
        if count > 0:
            self._write(np.zeros(count, dtype="<i2").tobytes())
            self._samples += count
            return count
        return 0

    def _write(self, payload: bytes) -> None:
        try:
            self._fh.write(payload)
        except OSError as exc:  # disk full is the realistic one on a long book
            raise AudioWriteError(f"writing audio to {self.path} failed: {exc}") from exc

    def flush(self) -> None:
        if not self._fh.closed:
            self._fh.flush()

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
    audio_filter: str | None = None,
) -> Path:
    """Encode raw PCM to MP3. `quality` is LAME VBR 0 (best) - 9 (worst).

    `audio_filter` is an ffmpeg -af chain applied during this one pass. Voice profiles
    use it for their pitch shift: the encode already runs ffmpeg over the whole book,
    so treating it here costs nothing and avoids per-chunk seams.
    """
    ffmpeg = require_ffmpeg()
    out_path = Path(out_path)
    pcm_path = Path(pcm_path)

    if not pcm_path.is_file():
        raise AudioWriteError(f"no PCM data to encode at {pcm_path}")
    if pcm_path.stat().st_size == 0:
        raise AudioWriteError(
            "no audio was synthesized — every chunk failed or produced silence"
        )

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AudioWriteError(f"cannot create {out_path.parent}: {exc}") from exc

    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", str(pcm_path),
        "-codec:a", "libmp3lame", "-qscale:a", str(quality),
    ]
    if audio_filter:
        cmd += ["-af", audio_filter]
    if tags:
        cmd += tags.as_ffmpeg_args()
    cmd.append(str(out_path))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as exc:
        raise AudioWriteError(f"could not run ffmpeg: {exc}") from exc
    if result.returncode != 0:
        raise AudioWriteError(
            f"ffmpeg failed ({result.returncode}): {result.stderr.strip()[:500]}"
        )
    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise AudioWriteError(f"ffmpeg reported success but {out_path} is empty")
    return out_path


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:d}:{secs:02d}"
