"""Unit tests for Rung 2, piece 1 — the key loader.

We test load_api_key() by passing a fake env dict, so these tests need no real key,
no .env file, and never touch the network. (The default path that reads the real
environment is the untestable I/O part, like record() in Rung 1 — we don't test it.)
"""

import numpy as np
import pytest

from rung1_audio import SAMPLE_RATE, load_wav, save_wav
from rung2_tts import load_api_key, pcm16_to_array


def test_returns_key_when_present():
    """The happy path: a key in the env mapping comes back unchanged."""
    # Arrange + Act: hand the loader a fake environment holding a key.
    key = load_api_key({"DEEPGRAM_API_KEY": "abc123"})

    # Assert: we get exactly that key back.
    assert key == "abc123"


def test_raises_when_missing():
    """A totally empty env must fail loudly, not return None for an HTTP call to choke on."""
    # Assert: an env with no key raises, so a forgotten .env is caught immediately.
    with pytest.raises(RuntimeError):
        load_api_key({})


def test_raises_when_empty_string():
    """A present-but-blank key (e.g. `DEEPGRAM_API_KEY=` in .env) is just as broken as missing."""
    # Assert: empty string is treated as "not set", not as a valid key.
    with pytest.raises(RuntimeError):
        load_api_key({"DEEPGRAM_API_KEY": ""})


def test_pcm16_to_array_reconstructs_known_samples():
    """Known bytes -> known samples. We include the int16 extremes (32767, -32768) so any
    byte-order or overflow bug at the boundaries would show up."""
    # Arrange: pick samples by hand, then turn them INTO bytes the way Deepgram's wire format
    # does (.tobytes() on an int16 array is exactly a linear16 byte stream).
    expected = np.array([0, 1000, -1000, 32767, -32768], dtype=np.int16)
    raw = expected.tobytes()

    # Act: convert those raw bytes back into an array.
    audio = pcm16_to_array(raw)

    # Assert: the samples come back exactly, in the (frames, 1) mono shape Rung 1 expects.
    assert audio.dtype == np.int16
    assert audio.shape == (5, 1)
    assert np.array_equal(audio.reshape(-1), expected)


def test_pcm16_to_array_handles_empty():
    """Edge case: no audio bytes -> an empty (0, 1) array, not a crash."""
    audio = pcm16_to_array(b"")
    assert audio.shape == (0, 1)


def test_pcm16_to_array_output_round_trips_through_wav(tmp_path):
    """The conversion's output must work with Rung 1: save it as a WAV and reload it,
    and the samples must survive — proving piece 3 really does feed piece-1 functions."""
    # Arrange: bytes -> array via the function under test.
    expected = np.array([5, -5, 20000, -20000], dtype=np.int16)
    audio = pcm16_to_array(expected.tobytes())
    path = str(tmp_path / "tts.wav")

    # Act: round-trip the array through Rung 1's save/load.
    save_wav(audio, path, SAMPLE_RATE)
    reloaded, sr = load_wav(path)

    # Assert: samples and rate survive the trip.
    assert np.array_equal(reloaded.reshape(-1), expected)
    assert sr == SAMPLE_RATE
