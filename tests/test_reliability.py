"""Failure-path tests: what happens when the narrator hiccups."""
import json

import numpy as np
import pytest

from narratortool import pipeline
from narratortool.audio import AudioWriteError, PCMWriter
from narratortool.pipeline import NarrationAborted, narrate_file
from narratortool.runlog import read_log
from narratortool.tts.base import BackendUnavailable, SynthesisError

pytestmark = pytest.mark.skipif(
    not __import__("shutil").which("ffmpeg"), reason="ffmpeg not installed"
)

SR = 24_000


class FakeBackend:
    """Synthesizes a fixed tone, failing on chunks whose index is in `fail_on`."""

    def __init__(self, fail_on=(), error=SynthesisError, always=False, sentence_ok=False):
        self.fail_on = set(fail_on)
        self.error = error
        self.always = always
        self.sentence_ok = sentence_ok  # succeed when handed a single sentence
        self.calls = 0

    sample_rate = SR
    name = "fake"

    def synthesize(self, text):
        self.calls += 1
        short = self.sentence_ok and len(text) < 40
        if (self.always or self.calls in self.fail_on) and not short:
            raise self.error(f"synthetic failure on call {self.calls}")
        return np.full(int(0.1 * SR), 0.25, dtype=np.float32)


@pytest.fixture
def book(tmp_path):
    src = tmp_path / "book.txt"
    src.write_text("\n\n".join(f"Paragraph number {i} of the test document." for i in range(6)))
    return src


def _run(book, tmp_path, backend, **kw):
    return narrate_file(book, output=tmp_path / "out.mp3", backend=backend, max_chars=60, **kw)


class TestRetries:
    def test_transient_failure_is_retried_and_recorded(self, book, tmp_path, monkeypatch):
        monkeypatch.setattr(pipeline, "RETRY_BACKOFF", 0)
        backend = FakeBackend(fail_on=[2])  # second call fails, retry succeeds
        result = _run(book, tmp_path, backend)

        assert result.output.exists()
        assert result.retried == 1
        assert not result.failures
        statuses = [r["status"] for r in read_log(result.log_path) if r["event"] == "chunk"]
        assert "retried" in statuses

    def test_chunk_that_never_works_is_skipped_not_fatal(self, book, tmp_path, monkeypatch):
        monkeypatch.setattr(pipeline, "RETRY_BACKOFF", 0)

        class OneBadChunk(FakeBackend):
            def synthesize(self, text):
                if "number 3" in text:
                    raise SynthesisError("this passage always fails")
                return np.full(int(0.1 * SR), 0.25, dtype=np.float32)

        result = _run(book, tmp_path, OneBadChunk(), retries=1)

        assert result.output.exists()
        assert len(result.failures) == 1
        assert "number 3" in result.failures[0].text
        assert not result.ok

    def test_sentence_level_recovery_saves_the_chunk(self, book, tmp_path, monkeypatch):
        monkeypatch.setattr(pipeline, "RETRY_BACKOFF", 0)
        src = tmp_path / "two.txt"
        src.write_text("First sentence here. Second sentence here.")
        backend = FakeBackend(always=True, sentence_ok=True)

        result = narrate_file(src, output=tmp_path / "two.mp3", backend=backend, max_chars=400)

        assert result.output.exists()
        assert not result.failures
        assert result.degraded == 0 or result.duration_seconds > 0


class TestOmissionNotice:
    """A skipped chunk is announced aloud so the gap is not silently invisible."""

    @staticmethod
    def _skips_one(marker_len=0.2):
        class OneBadChunk(FakeBackend):
            def synthesize(self, text):
                if text == pipeline.OMISSION_NOTICE:
                    return np.full(int(marker_len * SR), 0.3, dtype=np.float32)
                if "number 2" in text:
                    raise SynthesisError("nope")
                return np.full(int(0.1 * SR), 0.25, dtype=np.float32)
        return OneBadChunk()

    def test_skipped_chunk_is_announced_in_the_audio(self, book, tmp_path):
        result = _run(book, tmp_path, self._skips_one(), retries=0)

        assert len(result.failures) == 1
        assert result.notices_spoken == 1
        record = next(r for r in read_log(result.log_path) if r.get("status") == "skipped")
        assert record["notice"] == "spoken"
        assert record["notice_seconds"] > 0

    def test_notice_is_synthesized_once_and_reused(self, book, tmp_path):
        class TwoBadChunks(FakeBackend):
            def __init__(self):
                super().__init__()
                self.notice_calls = 0

            def synthesize(self, text):
                if text == pipeline.OMISSION_NOTICE:
                    self.notice_calls += 1
                    return np.full(int(0.2 * SR), 0.3, dtype=np.float32)
                if "number 1" in text or "number 3" in text:
                    raise SynthesisError("nope")
                return np.full(int(0.1 * SR), 0.25, dtype=np.float32)

        backend = TwoBadChunks()
        result = _run(book, tmp_path, backend, retries=0)

        assert result.notices_spoken == 2
        assert backend.notice_calls == 1  # rendered once, reused for the second gap

    def test_clean_run_never_synthesizes_the_notice(self, book, tmp_path):
        class Counting(FakeBackend):
            seen = []

            def synthesize(self, text):
                Counting.seen.append(text)
                return np.full(int(0.1 * SR), 0.25, dtype=np.float32)

        _run(book, tmp_path, Counting())
        assert pipeline.OMISSION_NOTICE not in Counting.seen

    def test_unsayable_notice_falls_back_to_silence(self, book, tmp_path):
        class NoticeAlwaysFails(FakeBackend):
            def synthesize(self, text):
                if text == pipeline.OMISSION_NOTICE:
                    raise SynthesisError("cannot say it")
                if "number 2" in text:
                    raise SynthesisError("nope")
                return np.full(int(0.1 * SR), 0.25, dtype=np.float32)

        result = _run(book, tmp_path, NoticeAlwaysFails(), retries=0)

        assert result.output.exists()  # a failed notice must not fail the run
        assert result.notices_spoken == 0
        record = next(r for r in read_log(result.log_path) if r.get("status") == "skipped")
        assert record["notice"] == "silent"

    def test_notice_retries_on_a_later_gap_after_a_blip(self, book, tmp_path):
        class BlipOnce(FakeBackend):
            def __init__(self):
                super().__init__()
                self.notice_calls = 0

            def synthesize(self, text):
                if text == pipeline.OMISSION_NOTICE:
                    self.notice_calls += 1
                    if self.notice_calls == 1:
                        raise SynthesisError("transient")
                    return np.full(int(0.2 * SR), 0.3, dtype=np.float32)
                if "number 1" in text or "number 3" in text:
                    raise SynthesisError("nope")
                return np.full(int(0.1 * SR), 0.25, dtype=np.float32)

        result = _run(book, tmp_path, BlipOnce(), retries=0)
        assert result.notices_spoken == 1  # silent for the first gap, spoken for the second

    def test_disabling_the_notice_leaves_a_silent_gap(self, book, tmp_path):
        result = _run(book, tmp_path, self._skips_one(), retries=0, omission_notice=None)

        assert result.notices_spoken == 0
        record = next(r for r in read_log(result.log_path) if r.get("status") == "skipped")
        assert record["notice"] == "silent"

    def test_notice_is_customisable(self, book, tmp_path):
        said = []

        class Custom(FakeBackend):
            def synthesize(self, text):
                said.append(text)
                if "number 2" in text:
                    raise SynthesisError("nope")
                return np.full(int(0.1 * SR), 0.25, dtype=np.float32)

        _run(book, tmp_path, Custom(), retries=0, omission_notice="Text missing here.")
        assert "Text missing here." in said

    def test_aborting_run_gets_no_notice(self, book, tmp_path, monkeypatch):
        monkeypatch.setattr(pipeline, "RETRY_BACKOFF", 0)
        said = []

        class Failing(FakeBackend):
            def synthesize(self, text):
                said.append(text)
                if "number 2" in text:
                    raise SynthesisError("nope")
                return np.full(int(0.3 * SR), 0.25, dtype=np.float32)

        with pytest.raises(NarrationAborted):
            _run(book, tmp_path, Failing(), retries=0, on_error="abort")
        assert pipeline.OMISSION_NOTICE not in said

    def test_notice_alone_does_not_count_as_a_narrated_book(self, book, tmp_path, monkeypatch):
        """Every chunk failing must not pass as success just because gaps were announced."""
        monkeypatch.setattr(pipeline, "RETRY_BACKOFF", 0)
        monkeypatch.setattr(pipeline, "MAX_CONSECUTIVE_FAILURES", 99)

        class OnlyNotice(FakeBackend):
            def synthesize(self, text):
                if text == pipeline.OMISSION_NOTICE:
                    return np.full(int(0.2 * SR), 0.3, dtype=np.float32)
                raise SynthesisError("everything fails")

        with pytest.raises(NarrationAborted, match="no audio was synthesized"):
            _run(book, tmp_path, OnlyNotice(), retries=0)


class TestAbort:
    def test_abort_policy_salvages_partial_audio(self, book, tmp_path, monkeypatch):
        monkeypatch.setattr(pipeline, "RETRY_BACKOFF", 0)

        class LateFailure(FakeBackend):
            def synthesize(self, text):
                if "number 4" in text:
                    raise SynthesisError("boom")
                return np.full(int(0.5 * SR), 0.25, dtype=np.float32)

        with pytest.raises(NarrationAborted) as caught:
            _run(book, tmp_path, LateFailure(), retries=0, on_error="abort")

        partial = caught.value.partial_output
        assert partial is not None and partial.exists() and partial.stat().st_size > 0
        assert partial.name == "out.partial.mp3"
        assert not (tmp_path / "out.mp3").exists()

    def test_runaway_backend_aborts_even_when_skipping(self, book, tmp_path, monkeypatch):
        monkeypatch.setattr(pipeline, "RETRY_BACKOFF", 0)
        monkeypatch.setattr(pipeline, "MAX_CONSECUTIVE_FAILURES", 3)

        with pytest.raises(NarrationAborted, match="in a row"):
            _run(book, tmp_path, FakeBackend(always=True), retries=0, on_error="skip")

    def test_dead_backend_is_not_retried(self, book, tmp_path, monkeypatch):
        monkeypatch.setattr(pipeline, "RETRY_BACKOFF", 0)
        backend = FakeBackend(always=True, error=BackendUnavailable)

        with pytest.raises(NarrationAborted, match="BackendUnavailable"):
            _run(book, tmp_path, backend, retries=5)
        assert backend.calls == 1  # no retry storm against a dead engine

    def test_interrupt_salvages_audio(self, book, tmp_path):
        class Interrupting(FakeBackend):
            def synthesize(self, text):
                self.calls += 1
                if self.calls > 3:
                    raise KeyboardInterrupt
                return np.full(int(0.4 * SR), 0.25, dtype=np.float32)

        with pytest.raises(NarrationAborted, match="interrupted"):
            _run(book, tmp_path, Interrupting())
        assert (tmp_path / "out.partial.mp3").exists()


class TestRunLog:
    def test_log_records_every_chunk_with_offsets(self, book, tmp_path):
        result = _run(book, tmp_path, FakeBackend())
        records = read_log(result.log_path)

        assert records[0]["event"] == "run_start"
        assert records[-1]["event"] == "run_end"
        chunks = [r for r in records if r["event"] == "chunk"]
        assert len(chunks) == result.chunk_count
        offsets = [c["offset_seconds"] for c in chunks]
        assert offsets == sorted(offsets)  # offsets locate each chunk in the finished MP3
        assert all("text" in c and "chapter" in c for c in chunks)

    def test_log_names_the_skipped_chunks(self, book, tmp_path, monkeypatch):
        monkeypatch.setattr(pipeline, "RETRY_BACKOFF", 0)

        class OneBadChunk(FakeBackend):
            def synthesize(self, text):
                if "number 2" in text:
                    raise SynthesisError("nope")
                return np.full(int(0.1 * SR), 0.25, dtype=np.float32)

        result = _run(book, tmp_path, OneBadChunk(), retries=0)
        end = read_log(result.log_path)[-1]

        assert end["status"] == "completed_with_skips"
        assert end["skipped"] == 1
        assert end["skipped_indexes"] == [f.index for f in result.failures]

    def test_log_survives_an_aborted_run(self, book, tmp_path, monkeypatch):
        monkeypatch.setattr(pipeline, "RETRY_BACKOFF", 0)
        with pytest.raises(NarrationAborted) as caught:
            _run(book, tmp_path, FakeBackend(always=True), retries=0, on_error="abort")

        records = read_log(caught.value.log_path)
        assert records[-1]["event"] == "run_end"
        assert records[-1]["status"] == "aborted"

    def test_no_log_flag_writes_nothing(self, book, tmp_path):
        result = _run(book, tmp_path, FakeBackend(), write_log=False)
        assert result.log_path is None
        assert not list(tmp_path.glob("*.jsonl"))

    def test_unwritable_log_does_not_kill_the_run(self, book, tmp_path):
        result = _run(book, tmp_path, FakeBackend(),
                      log_path=tmp_path / "no" / "such\x00path" / "run.jsonl")
        assert result.output.exists()


class TestAudioGuards:
    def test_nonfinite_audio_becomes_silence_not_static(self, tmp_path):
        writer = PCMWriter(tmp_path / "a.pcm", SR)
        writer.append(np.array([np.nan, np.inf, -np.inf, 0.5], dtype=np.float32))
        writer.close()

        samples = np.frombuffer((tmp_path / "a.pcm").read_bytes(), dtype="<i2")
        assert list(samples[:3]) == [0, 0, 0]
        assert writer.nonfinite_chunks == 1

    def test_encoding_nothing_is_a_clear_error(self, tmp_path):
        from narratortool.audio import encode_mp3

        (tmp_path / "empty.pcm").write_bytes(b"")
        with pytest.raises(AudioWriteError, match="no audio was synthesized"):
            encode_mp3(tmp_path / "empty.pcm", tmp_path / "out.mp3", SR)

    def test_totally_silent_run_fails_loudly(self, book, tmp_path):
        class Silent(FakeBackend):
            def synthesize(self, text):
                return np.zeros(0, dtype=np.float32)

        with pytest.raises(NarrationAborted, match="no audio was synthesized"):
            _run(book, tmp_path, Silent())


class TestValidation:
    def test_bad_on_error_policy_is_rejected(self, book, tmp_path):
        with pytest.raises(ValueError, match="on_error"):
            _run(book, tmp_path, FakeBackend(), on_error="explode")

    def test_json_log_is_parseable_line_by_line(self, book, tmp_path):
        result = _run(book, tmp_path, FakeBackend())
        for line in result.log_path.read_text().splitlines():
            json.loads(line)
