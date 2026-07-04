"""Unit tests for Rung 3 — the pure JSON extraction.

We test extract_transcript() with hand-built dicts shaped like Deepgram's response, so these
tests need no key, no audio, and no network. The transcribe() call that actually hits Deepgram
is the untestable I/O part (like synthesize() in Rung 2) — verified by a live run instead.
"""

from rung3_stt import extract_transcript, parse_stt_message


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


# --- streaming message parser (parse_stt_message) ---

def _stream_msg(transcript: str, is_final: bool, speech_final: bool) -> dict:
    """Build a minimal Deepgram *streaming* message (note: channel is singular here)."""
    return {
        "channel": {"alternatives": [{"transcript": transcript}]},
        "is_final": is_final,
        "speech_final": speech_final,
    }


def test_parse_interim_message():
    """An interim guess: text present, is_final False (may still change)."""
    result = parse_stt_message(_stream_msg("welcome to", is_final=False, speech_final=False))
    assert result.text == "welcome to"
    assert result.is_final is False
    assert result.speech_final is False


def test_parse_final_message():
    """A settled span: same text but is_final True."""
    result = parse_stt_message(_stream_msg("welcome to tony's", is_final=True, speech_final=False))
    assert result.text == "welcome to tony's"
    assert result.is_final is True


def test_parse_endpoint_message():
    """speech_final True is the endpointing signal — the speaker paused/stopped."""
    result = parse_stt_message(_stream_msg("welcome to tony's pizza", is_final=True, speech_final=True))
    assert result.speech_final is True


def test_parse_non_transcript_message_has_empty_text():
    """Metadata/keep-alive messages have no channel: text is "", flags default to False."""
    result = parse_stt_message({"type": "Metadata"})
    assert result.text == ""
    assert result.is_final is False
    assert result.speech_final is False


def test_parse_empty_alternatives_has_empty_text():
    """A Results message with an empty alternatives list yields "" text, not an IndexError."""
    result = parse_stt_message({"channel": {"alternatives": []}, "is_final": True})
    assert result.text == ""
    assert result.is_final is True
