"""Structured run log.

Every narration writes a JSONL sidecar next to the MP3: one record per synthesized
chunk, plus a header and a footer. The point is verifiability after the fact — a
four-hour audiobook is not something anyone listens to end-to-end to confirm it came
out clean, so each record carries the chunk's offset into the finished audio, which
chapter it came from, and whether it needed retries or was skipped entirely. Grepping
for `"status": "skipped"` and seeking to that offset is the whole workflow.

JSONL rather than a prose log because it stays greppable and machine-checkable while
still being readable in a pager.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

LOG_SUFFIX = ".narration.jsonl"


def default_log_path(output: Path) -> Path:
    """Sidecar path for an output MP3: book.mp3 -> book.narration.jsonl."""
    return output.with_suffix("").with_name(output.stem + LOG_SUFFIX)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RunLog:
    """Append-only JSONL writer. Never raises: a broken log must not kill a run."""

    def __init__(self, path: Path | None) -> None:
        self.path = Path(path) if path else None
        self._fh = None
        self._failed = False
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("w", encoding="utf-8")
        except (OSError, ValueError) as exc:  # ValueError: null byte in the path
            log.warning("could not open run log %s: %s", self.path, exc)
            self.path = None
            self._failed = True

    @property
    def enabled(self) -> bool:
        return self._fh is not None

    def write(self, event: str, **fields: Any) -> None:
        if self._fh is None:
            return
        record = {"event": event, "time": _now(), **fields}
        try:
            self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            self._fh.flush()  # flushed per record so a killed run still leaves a usable log
        except (OSError, TypeError, ValueError) as exc:
            if not self._failed:
                log.warning("run log write failed, continuing without it: %s", exc)
                self._failed = True
            self.close()

    def close(self) -> None:
        if self._fh is not None and not self._fh.closed:
            try:
                self._fh.close()
            except OSError:
                pass
        self._fh = None

    def __enter__(self) -> "RunLog":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_log(path: str | Path) -> list[dict]:
    """Parse a run log back into records, tolerating a truncated final line."""
    records = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                break  # truncated tail from a killed run
    return records
