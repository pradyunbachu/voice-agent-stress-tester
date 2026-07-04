"""Unit tests for Rung 3 — the pure JSON extraction.

We test extract_transcript() with hand-built dicts shaped like Deepgram's response, so these
tests need no key, no audio, and no network. The transcribe() call that actually hits Deepgram
is the untestable I/O part (like synthesize() in Rung 2) — verified by a live run instead.
"""

from rung3_stt import extract_transcript


def _response(transcript: str) -> dict:
    """Build a minimal Deepgram-shaped response carrying one transcript."""
    return {"results": {"channels": [{"alternatives": [{"transcript": transcript}]}]}}


def test_extracts_transcript_from_well_formed_response():
    """The happy path: pull the string out of results->channels[0]->alternatives[0]->transcript."""
    response = _response("welcome to tony's pizza")
    assert extract_transcript(response) == "welcome to tony's pizza"


def test_empty_alternatives_returns_empty_string():
    """Silence gives an empty alternatives list — we want "", not an IndexError."""
    response = {"results": {"channels": [{"alternatives": []}]}}
    assert extract_transcript(response) == ""


def test_no_channels_returns_empty_string():
    """A response with no channels at all must also degrade to "", not crash."""
    response = {"results": {"channels": []}}
    assert extract_transcript(response) == ""


def test_missing_results_key_returns_empty_string():
    """A totally unexpected/empty payload (e.g. an error shape) still yields "" safely."""
    assert extract_transcript({}) == ""


def test_transcript_field_missing_returns_empty_string():
    """An alternative with no transcript field defaults to "" rather than KeyError."""
    response = {"results": {"channels": [{"alternatives": [{"confidence": 0.9}]}]}}
    assert extract_transcript(response) == ""
