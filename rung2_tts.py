"""Stage A, Rung 2 — TTS alone (Deepgram Aura), batch version.

Hand Deepgram some text, get back spoken audio as raw linear16 PCM, turn those bytes
into the same int16 array from Rung 1, then save/inspect/play with Rung 1's functions.

This file is being built one piece at a time. Piece 1 (here): load the API key safely.
"""

import os
from collections.abc import Mapping

import numpy as np               # the int16 array Deepgram's bytes become
import requests                  # HTTP client: sends the request to Deepgram, returns the response
import sounddevice as sd         # play the result so your ears confirm it's speech
from dotenv import load_dotenv  # reads a .env file's KEY=value lines into the environment

# Reuse Rung 1: the one sample rate (so "ask Deepgram for" == "save_wav at"), plus the
# functions that already know how to inspect/save the array.
from rung1_audio import SAMPLE_RATE, inspect, save_wav

# Deepgram's text-to-speech endpoint and the voice we want.
DEEPGRAM_TTS_URL = "https://api.deepgram.com/v1/speak"
TTS_MODEL = "aura-2-thalia-en"


def load_api_key(env: Mapping[str, str] | None = None) -> str:
    """Return the Deepgram API key, or raise a clear error if it isn't set.

    Pure/impure split (same idea as Rung 1's audio_stats vs record): the value-checking
    logic is pure and testable. By default (env=None) we do the real I/O — read .env and
    look at os.environ. Tests instead pass a fake dict, so they never need a real key.

    Args:
        env: a mapping to read the key from. Defaults to the real environment after
             loading .env. Tests pass their own dict to exercise the logic in isolation.

    Returns:
        The API key string.

    Raises:
        RuntimeError: if DEEPGRAM_API_KEY is missing or empty — fail loudly now, not
                      with a confusing 401 deep inside an HTTP call later.
    """
    if env is None:
        load_dotenv()       # populate os.environ from .env (no-op if the file is absent)
        env = os.environ
    key = env.get("DEEPGRAM_API_KEY")
    if not key:             # catches both "missing" and "present but empty"
        raise RuntimeError(
            "DEEPGRAM_API_KEY not found. Copy .env.example to .env and add your key."
        )
    return key


def synthesize(text: str, api_key: str) -> bytes:
    """Ask Deepgram to speak `text` and return the audio as raw linear16 PCM bytes.

    Touches the network, so (like record() in Rung 1) it's verified by a live run, not a
    unit test. The bytes carry no header — just sample values at SAMPLE_RATE; turning them
    into an array is the next, pure, testable piece.
    """
    # Query params spell out the format we want back: raw int16 PCM at our one sample rate.
    params = {"model": TTS_MODEL, "encoding": "linear16", "sample_rate": SAMPLE_RATE}
    # The key authenticates us; "Token <key>" is Deepgram's required header format.
    headers = {"Authorization": f"Token {api_key}", "Content-Type": "application/json"}

    response = requests.post(DEEPGRAM_TTS_URL, params=params, headers=headers, json={"text": text})
    # On a bad key or bad request Deepgram returns an error JSON, not audio. Raising here
    # stops us from treating that error body as if it were sound.
    response.raise_for_status()
    return response.content  # the audio itself: raw linear16 bytes


def pcm16_to_array(raw: bytes) -> np.ndarray:
    """Turn raw linear16 PCM bytes into a (frames, 1) int16 array — the shape Rung 1 wants.

    Pure (bytes in, array out), so this is the unit-tested heart of the rung. It's the exact
    inverse of writing int16 samples to disk: frombuffer reads the stream two bytes at a time
    back into samples; reshape turns the flat list into the mono (frames, 1) column shape.
    """
    samples = np.frombuffer(raw, dtype=np.int16)  # reinterpret bytes as int16 samples
    return samples.reshape(-1, 1)                 # flat (N,) -> mono (N, 1)


# End-to-end demo: text -> Deepgram -> bytes -> array -> inspect/save/play (the last three
# are Rung 1, fed by a brand-new source). Your ears + the stats confirm it's real speech.
if __name__ == "__main__":
    key = load_api_key()
    print(f"DEEPGRAM_API_KEY loaded ({len(key)} chars)")

    audio_bytes = synthesize("Welcome to Tony's Pizza. What can I get started for you?", key)
    print(f"Got {len(audio_bytes)} bytes of linear16 audio from Deepgram.")

    audio = pcm16_to_array(audio_bytes)
    inspect(audio, "tts")                  # same stats line as Rung 1: shape/dtype/min/max
    save_wav(audio, "tts_output.wav", SAMPLE_RATE)
    print("Saved tts_output.wav")

    print("Playing back...")
    sd.play(audio, SAMPLE_RATE)
    sd.wait()
    print("Done.")
