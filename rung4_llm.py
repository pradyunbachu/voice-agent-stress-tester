"""Stage A, Rung 4 — LLM alone (Groq), the "brain".

Give the model a list of messages, get a reply back. Same request/response shape as the
Deepgram rungs, with one new idea: the input is a *conversation* — a list of role-tagged
messages (system / user / assistant), not a single prompt string.

  batch:  chat_completion() -> extract_reply()   (this file)
  stream: tokens as they generate                (next piece)
"""

import json
import os
from collections.abc import Mapping

import requests                  # HTTP client: POSTs the messages, returns the JSON reply
from dotenv import load_dotenv  # reads a .env file's KEY=value lines into the environment

# Groq speaks the OpenAI chat-completions API. llama-3.3-70b-versatile is a solid general model;
# swap to llama-3.1-8b-instant later if the live loop needs lower latency.
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
LLM_MODEL = "llama-3.3-70b-versatile"


def load_groq_key(env: Mapping[str, str] | None = None) -> str:
    """Return the Groq API key, or raise if it isn't set.

    Same pure/impure seam as Rung 2's Deepgram loader: by default (env=None) we read .env and
    os.environ; tests pass a fake dict to exercise the logic without a real key.
    """
    if env is None:
        load_dotenv()
        env = os.environ
    key = env.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY not found. Copy .env.example to .env and add your key.")
    return key


def build_messages(system_prompt: str, user_text: str) -> list[dict]:
    """Build the two-message conversation the API expects: standing instructions + the user turn.

    Pure (strings in, list out). This is the shape you grow over a real dialog by appending
    the assistant's reply and the next user turn — the API is stateless, so the messages list
    IS the conversation memory.
    """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]


def chat_completion(messages: list[dict], api_key: str) -> dict:
    """POST the conversation to Groq and return the parsed JSON reply.

    Touches the network, so (like transcribe) it's verified by a live run, not a unit test.
    Note the Bearer auth scheme here vs Deepgram's "Token".
    """
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": LLM_MODEL, "messages": messages}

    response = requests.post(GROQ_URL, headers=headers, json=body)
    response.raise_for_status()  # bad key / bad request -> raise instead of parsing an error body
    return response.json()


def extract_reply(response: dict) -> str:
    """Dig the model's text out of Groq's nested JSON: choices[0] -> message -> content.

    Pure (dict in, string out), so this is the unit-tested core. Returns "" for an empty or
    malformed response instead of raising, the same way extract_transcript does.
    """
    choices = response.get("choices", [])
    if not choices:
        return ""
    return choices[0].get("message", {}).get("content", "")


def parse_sse_line(line: str) -> str:
    """Pull the token out of one Server-Sent-Events line, or "" if there's nothing to emit.

    Streaming lines look like `data: {json}`, and the new token is at choices[0].delta.content
    (delta = the increment, vs batch's whole `message`). Returns "" for empty lines, the
    `data: [DONE]` sentinel, keep-alive comments, and chunks with no content (e.g. the opening
    role delta). Pure (string in, string out), so this is the streaming pass's unit-tested core.
    """
    if not line.startswith("data:"):
        return ""                          # blank lines and ":" keep-alive comments
    payload = line[len("data:"):].strip()  # drop the "data:" prefix
    if not payload or payload == "[DONE]":
        return ""                          # end-of-stream sentinel isn't JSON
    choices = json.loads(payload).get("choices", [])
    if not choices:
        return ""
    return choices[0].get("delta", {}).get("content") or ""  # content can be null -> ""


def stream_completion(messages: list[dict], api_key: str):
    """Yield reply tokens as the model generates them (a generator).

    Same POST as chat_completion but with stream=True, so the reply comes back as SSE lines
    we parse one at a time. Network-touching, so verified by a live run, not a unit test.
    """
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": LLM_MODEL, "messages": messages, "stream": True}

    with requests.post(GROQ_URL, headers=headers, json=body, stream=True) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines():   # iterate the SSE lines as they arrive
            token = parse_sse_line(raw_line.decode("utf-8")) if raw_line else ""
            if token:
                yield token


# Streaming demo: print the reply token-by-token as the model generates it (typewriter effect).
if __name__ == "__main__":
    key = load_groq_key()
    system = "You are a concise food-ordering assistant for Tony's Pizza. Keep replies short."
    messages = build_messages(system, "Hi, do you have any vegetarian options?")

    print("Streaming reply: ", end="", flush=True)
    for token in stream_completion(messages, key):
        print(token, end="", flush=True)   # end="" + flush so tokens appear live, not buffered
    print()  # final newline once the stream ends
