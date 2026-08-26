"""Tests for the Chatterbox backend's pure logic.

Nothing here loads the model. The parts worth testing are the ones that decide what
audio survives — the carrier trim and the time-stretch chain — and both are ordinary
numpy and string work that a GPU would tell us nothing extra about.
"""
from __future__ import annotations

import numpy as np
import pytest

from narratortool.tts import DEFAULT_BACKEND, available_backends, chars_per_second, default_voice
from narratortool.tts.base import BackendUnavailable
from narratortool.tts.chatterbox_backend import (
    _CARRIER,
    CHARS_PER_SECOND,
    DEFAULT_VOICE,
    VOICE_DIR,
    _trim_carrier,
    atempo_chain,
    available_voices,
    resolve_voice,
)

SR = 24_000
CARRIER_SECONDS = len(_CARRIER) / CHARS_PER_SECOND


def speech(seconds: float, level: float = 0.5) -> np.ndarray:
    """A stand-in for speech: a tone loud enough to sit well above the silence floor."""
    t = np.arange(int(seconds * SR), dtype=np.float32) / SR
    return (level * np.sin(2 * np.pi * 180.0 * t)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


class TestTrimCarrier:
    def test_cuts_at_the_pause_after_the_carrier(self):
        audio = np.concatenate([speech(CARRIER_SECONDS), silence(0.3), speech(2.0)])
        out = _trim_carrier(audio, SR, CARRIER_SECONDS)
        # The target survives whole, plus at most the backoff of leading silence.
        assert 2.0 <= out.size / SR <= 2.1

    def test_the_cut_lands_in_the_pause_not_in_the_word(self):
        """A cut past the word onset clips the first consonant, which is audible."""
        audio = np.concatenate([speech(CARRIER_SECONDS), silence(0.3), speech(2.0)])
        out = _trim_carrier(audio, SR, CARRIER_SECONDS)
        assert float(np.abs(out[: int(0.02 * SR)]).max()) < 1e-6

    def test_no_carrier_audio_survives(self):
        """The whole point: "He looked at me and said" must not reach the listener."""
        audio = np.concatenate([speech(CARRIER_SECONDS, level=0.5), silence(0.3),
                                speech(2.0, level=0.5)])
        out = _trim_carrier(audio, SR, CARRIER_SECONDS)
        assert audio.size - out.size >= int(CARRIER_SECONDS * SR)

    def test_a_pause_early_in_the_window_is_taken_over_a_later_one(self):
        """The join is the longest pause in the window, not merely the first found."""
        audio = np.concatenate([
            speech(1.0), silence(0.05),          # a breath inside the carrier
            speech(0.4), silence(0.35),          # the real join
            speech(2.0),
        ])
        out = _trim_carrier(audio, SR, CARRIER_SECONDS)
        assert 2.0 <= out.size / SR <= 2.1

    def test_trailing_silence_does_not_win_over_the_join(self):
        """Found live on 'Notes', which trimmed to nothing.

        A short target leaves a longer tail of silence after itself than the carrier's
        full stop leaves before it, so on length alone the tail beats the real join
        and the cut lands past the end of the word.
        """
        audio = np.concatenate([
            speech(CARRIER_SECONDS), silence(0.25),   # the join
            speech(0.35),                             # a one-word target
            silence(0.9),                             # a longer tail
        ])
        out = _trim_carrier(audio, SR, CARRIER_SECONDS)
        assert float(np.abs(out).max()) > 0.1, "the target must survive the trim"

    def test_a_short_target_survives_an_overshooting_estimate(self):
        """With no pause to find, the estimate must not cut past the target."""
        audio = np.concatenate([speech(1.0), silence(0.02), speech(0.3)])
        out = _trim_carrier(audio, SR, CARRIER_SECONDS)
        assert float(np.abs(out).max()) > 0.1

    def test_falls_back_to_the_estimate_when_no_pause_is_found(self):
        """Better an approximate cut than the carrier left audible in every heading."""
        audio = speech(4.0)
        out = _trim_carrier(audio, SR, CARRIER_SECONDS)
        assert out.size == audio.size - int(CARRIER_SECONDS * SR)

    def test_a_pause_far_outside_the_window_is_not_mistaken_for_the_join(self):
        """A pause between the target's own sentences must not be taken as the join."""
        audio = np.concatenate([speech(6.0), silence(0.4), speech(2.0)])
        out = _trim_carrier(audio, SR, CARRIER_SECONDS)
        assert out.size == audio.size - int(CARRIER_SECONDS * SR)

    def test_audio_shorter_than_the_carrier_trims_to_empty(self):
        out = _trim_carrier(speech(0.5), SR, CARRIER_SECONDS)
        assert out.size == 0

    def test_empty_audio_is_handled(self):
        out = _trim_carrier(np.zeros(0, dtype=np.float32), SR, CARRIER_SECONDS)
        assert out.size == 0


class TestAtempoChain:
    def test_no_filter_at_normal_speed(self):
        assert atempo_chain(1.0) is None

    def test_a_single_stage_inside_ffmpegs_range(self):
        assert atempo_chain(0.9) == "atempo=0.900000"

    @pytest.mark.parametrize("speed", [0.1, 0.24, 0.4, 0.5, 0.75, 1.5, 2.0, 2.5, 3.0])
    def test_the_stages_multiply_back_to_the_requested_speed(self, speed):
        chain = atempo_chain(speed)
        product = 1.0
        for stage in chain.split(","):
            product *= float(stage.split("=")[1])
        assert product == pytest.approx(speed, rel=1e-5)

    @pytest.mark.parametrize("speed", [0.1, 0.24, 0.4, 2.5, 3.0, 4.0])
    def test_every_stage_is_within_ffmpegs_accepted_range(self, speed):
        """ffmpeg's atempo rejects anything outside 0.5-2.0, hence the chaining."""
        for stage in atempo_chain(speed).split(","):
            assert 0.5 <= float(stage.split("=")[1]) <= 2.0

    def test_a_nonpositive_speed_is_rejected(self):
        with pytest.raises(ValueError):
            atempo_chain(0.0)


class TestResolveVoice:
    def test_the_house_reference_ships_with_the_package(self):
        """The default backend has no voice at all without this clip."""
        assert DEFAULT_VOICE in available_voices()
        assert resolve_voice(DEFAULT_VOICE) == VOICE_DIR / f"{DEFAULT_VOICE}.wav"

    def test_a_path_to_a_clip_is_accepted(self, tmp_path):
        clip = tmp_path / "someone.wav"
        clip.write_bytes(b"RIFF....WAVE")
        assert resolve_voice(str(clip)) == clip

    def test_a_bundled_name_wins_over_a_file_of_the_same_name(self, tmp_path, monkeypatch):
        """`--voice house` must mean the same thing from any working directory."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / f"{DEFAULT_VOICE}.wav").write_bytes(b"RIFF....WAVE")
        assert resolve_voice(DEFAULT_VOICE) == VOICE_DIR / f"{DEFAULT_VOICE}.wav"

    def test_an_unknown_voice_names_what_is_available(self):
        with pytest.raises(BackendUnavailable, match="no reference clip"):
            resolve_voice("not-a-voice")


class TestRegistry:
    def test_chatterbox_is_the_default_backend(self):
        assert DEFAULT_BACKEND == "chatterbox"
        assert set(available_backends()) == {"chatterbox", "kokoro"}

    def test_each_backend_names_its_own_default_voice(self):
        """The two do not share a voice namespace, so neither default may leak."""
        assert default_voice("chatterbox") == "house"
        assert default_voice("kokoro") == "bf_emma"

    def test_each_backend_carries_its_own_measured_pace(self):
        """A duration estimate calibrated for one engine is wrong for the other."""
        assert chars_per_second("chatterbox") != chars_per_second("kokoro")

    def test_an_unknown_backend_is_named_clearly(self):
        with pytest.raises(ValueError, match="unknown TTS backend"):
            default_voice("vibevoice")
