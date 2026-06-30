"""Unit tests for Rung 1 — the pure, hardware-free parts of rung1_audio.

We deliberately do NOT test record() or sd.play(): those touch the mic/speaker, which
don't exist in a test runner and would block. The testable contract is the WAV round-trip
(what I save is bit-for-bit what I load) and the audio_stats math.

`tmp_path` is a pytest fixture: a fresh temp directory per test, auto-cleaned, so tests
never collide on disk or leave files behind.
"""

import numpy as np
import pytest

from rung1_audio import SAMPLE_RATE, audio_stats, load_wav, save_wav


def test_round_trip_preserves_samples(tmp_path):
    """The whole promise of WAV: int16 samples come back exactly as written, no drift."""
    # Arrange: hand-build a tiny signal that includes the extremes of int16
    # (32767 and -32768) so we'd catch any clipping or overflow at the boundaries.
    audio = np.array([[0], [1000], [-1000], [32767], [-32768]], dtype="int16")
    path = str(tmp_path / "rt.wav")

    # Act: write it to disk, then read it back into a brand-new array.
    save_wav(audio, path, SAMPLE_RATE)
    reloaded, _ = load_wav(path)

    # Assert: every sample is identical. We use exact equality (not approximate) because
    # PCM int16 is lossless, so any difference at all is a real bug, not rounding.
    assert np.array_equal(reloaded.reshape(audio.shape), audio)
    # And the format must still be int16, not silently widened to float.
    assert reloaded.dtype == np.int16


def test_round_trip_preserves_sample_rate(tmp_path):
    """The header must carry the rate back, or downstream STT would misread pitch/speed."""
    # Arrange: contents don't matter here, only the rate, so silence is fine.
    audio = np.zeros((100, 1), dtype="int16")
    path = str(tmp_path / "sr.wav")

    # Act: save at SAMPLE_RATE, then ask load_wav what rate the file reports.
    save_wav(audio, path, SAMPLE_RATE)
    _, sr = load_wav(path)

    # Assert: the rate we wrote is the rate we get back.
    assert sr == SAMPLE_RATE


def test_save_creates_file(tmp_path):
    """save_wav must actually produce a real file on disk, not just run without error."""
    # Arrange + Act: save a short array to a path that doesn't exist yet.
    path = tmp_path / "exists.wav"
    save_wav(np.zeros((10, 1), dtype="int16"), str(path), SAMPLE_RATE)

    # Assert: the file now exists AND has bytes in it (size > 0 rules out an empty file).
    assert path.exists() and path.stat().st_size > 0


def test_audio_stats_reports_correct_values():
    """Stats are just min/max/shape — lock the math so a refactor can't silently break it."""
    # Arrange: a 3-sample array whose min (-5), max (42), and shape we know by hand.
    audio = np.array([[-5], [0], [42]], dtype="int16")

    # Act: compute the stats (pure function, no disk or hardware involved).
    stats = audio_stats(audio)

    # Assert: each reported value matches what we can see in the input by eye.
    assert stats["shape"] == (3, 1)   # 3 frames, 1 channel (mono)
    assert stats["min"] == -5
    assert stats["max"] == 42
    assert stats["dtype"] == "int16"


def test_silence_round_trips_and_reports_zero(tmp_path):
    """Edge case: all-zeros (silence) must survive the round trip and report min=max=0,
    which is exactly the signal we use to detect a dead mic."""
    # Arrange: a block of pure silence (every sample is 0).
    audio = np.zeros((50, 1), dtype="int16")
    path = str(tmp_path / "silence.wav")

    # Act: round-trip it through disk, then compute stats on the reloaded copy.
    save_wav(audio, path, SAMPLE_RATE)
    reloaded, _ = load_wav(path)
    stats = audio_stats(reloaded)

    # Assert: silence stays silent (min == max == 0)...
    assert stats["min"] == 0 and stats["max"] == 0
    # ...and the round trip didn't invent any non-zero samples.
    assert np.array_equal(reloaded.reshape(audio.shape), audio)
