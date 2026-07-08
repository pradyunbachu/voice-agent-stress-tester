"""Unit tests for Rung 4 — the pure, network-free parts.

We test the key loader (via a fake env dict), the message builder, and the reply extractor
with hand-built response dicts. The chat_completion() call that hits Groq is the untestable
I/O part (like transcribe in Rung 3) — verified by a live run instead.
"""

import pytest

from rung4_llm import build_messages, extract_reply, load_groq_key, parse_sse_line


# --- key loader ---

def test_load_groq_key_returns_key_when_present():
    assert load_groq_key({"GROQ_API_KEY": "gsk_abc"}) == "gsk_abc"


def test_load_groq_key_raises_when_missing():
    with pytest.raises(RuntimeError):
        load_groq_key({})


def test_load_groq_key_raises_when_empty():
    with pytest.raises(RuntimeError):
        load_groq_key({"GROQ_API_KEY": ""})


# --- message builder ---

def test_build_messages_shapes_the_conversation():
    """system prompt and user text land in the right roles, in order."""
    messages = build_messages("be brief", "hello")
    assert messages == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello"},
    ]


# --- reply extractor ---

def _response(content: str) -> dict:
    """Build a minimal Groq-shaped chat-completion response."""
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def test_extract_reply_from_well_formed_response():
    assert extract_reply(_response("We have a veggie pizza.")) == "We have a veggie pizza."


def test_extract_reply_empty_choices_returns_empty_string():
    """No choices (e.g. an error/empty payload) -> "", not an IndexError."""
    assert extract_reply({"choices": []}) == ""


def test_extract_reply_missing_content_returns_empty_string():
    """A choice with no message content defaults to "" rather than KeyError."""
    assert extract_reply({"choices": [{"message": {}}]}) == ""


def test_extract_reply_missing_choices_key_returns_empty_string():
    assert extract_reply({}) == ""


# --- SSE line parser (streaming) ---

def test_parse_sse_line_extracts_token():
    """A normal data line yields its delta content token."""
    line = 'data: {"choices":[{"delta":{"content":"Yes"}}]}'
    assert parse_sse_line(line) == "Yes"


def test_parse_sse_line_done_sentinel_returns_empty():
    """The end-of-stream marker isn't JSON and carries no token."""
    assert parse_sse_line("data: [DONE]") == ""


def test_parse_sse_line_blank_line_returns_empty():
    """Blank separator lines between events yield nothing."""
    assert parse_sse_line("") == ""


def test_parse_sse_line_role_only_delta_returns_empty():
    """The first chunk often has a role but no content -> "", not a crash."""
    line = 'data: {"choices":[{"delta":{"role":"assistant"}}]}'
    assert parse_sse_line(line) == ""


def test_parse_sse_line_null_content_returns_empty():
    """A delta whose content is null (e.g. the finish chunk) collapses to ""."""
    line = 'data: {"choices":[{"delta":{"content":null}}]}'
    assert parse_sse_line(line) == ""


def test_parse_sse_line_non_data_line_returns_empty():
    """SSE keep-alive comment lines start with ':' and carry no token."""
    assert parse_sse_line(": keep-alive") == ""
