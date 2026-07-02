"""Unit tests for Rung 2 — the pure, network-free parts.

We test the key loader (via a fake env dict), the batch bytes->array conversion, and the
streaming carry-over buffer. None of these need a real key or the network; the parts that
do (synthesize / synthesize_stream / stream_and_play) are verified by a live run instead.
"""

import numpy as np
import pytest

from rung1_audio import SAMPLE_RATE, load_wav, save_wav
from rung2_tts import load_api_key, pcm16_to_array, take_complete_samples


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


def test_take_complete_samples_even_buffer_has_no_leftover():
    """An even byte count is all whole samples: everything converts, nothing carries over."""
    expected = np.array([100, -100, 32767], dtype=np.int16)  # 6 bytes, 3 samples
    samples, leftover = take_complete_samples(expected.tobytes())

    assert leftover == b""
    assert samples.shape == (3, 1)
    assert np.array_equal(samples.reshape(-1), expected)


def test_take_complete_samples_odd_buffer_keeps_trailing_byte():
    """An odd byte count converts the whole prefix and hands back exactly the last byte."""
    # 4 bytes (2 samples) + 1 stray byte = 5 bytes.
    two_samples = np.array([1234, 5678], dtype=np.int16).tobytes()
    buf = two_samples + b"\xAB"

    samples, leftover = take_complete_samples(buf)

    assert samples.shape == (2, 1)                     # only the 2 whole samples came out
    assert np.array_equal(samples.reshape(-1), np.array([1234, 5678], dtype=np.int16))
    assert leftover == b"\xAB"                          # the stray byte is preserved, not dropped


def test_take_complete_samples_carry_over_reconstructs_split_sample():
    """The whole point: a sample split across two chunks must reassemble exactly.

    We slice a known 2-sample stream at an ODD offset so the second sample is torn in half,
    then feed the pieces the way stream_and_play does (leftover + next chunk)."""
    expected = np.array([0x1234, 0x5678], dtype=np.int16)
    stream = expected.tobytes()                        # 4 bytes total
    chunk1, chunk2 = stream[:3], stream[3:]            # 3 bytes then 1 byte — splits sample 2

    # First chunk: one whole sample out, one byte held back.
    s1, leftover = take_complete_samples(chunk1)
    assert s1.shape == (1, 1)
    assert leftover == stream[2:3]

    # Second chunk: prepend the leftover, and the torn sample comes back whole.
    s2, leftover = take_complete_samples(leftover + chunk2)
    assert leftover == b""

    # Reassembled stream equals the original — no invented or misaligned samples.
    combined = np.concatenate([s1, s2]).reshape(-1)
    assert np.array_equal(combined, expected)


def test_take_complete_samples_empty_buffer():
    """No bytes -> no samples, no leftover — the safe base case for the very first chunk."""
    samples, leftover = take_complete_samples(b"")
    assert samples.shape == (0, 1)
    assert leftover == b""
