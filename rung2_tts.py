"""Stage A, Rung 2 — TTS alone (Deepgram Aura).

Hand Deepgram some text, get back spoken audio as raw linear16 PCM, turn those bytes
into the same int16 array from Rung 1, then save/inspect/play with Rung 1's functions.

Two ways to receive the audio:
  - batch:  synthesize()        — wait for the whole clip, then use it (pcm16_to_array).
  - stream: synthesize_stream() — get chunks as they're made and play them immediately,
            so the first sound comes out in ~tens of ms instead of after the whole clip.
"""

import os
import time
from collections.abc import Mapping

import numpy as np               # the int16 array Deepgram's bytes become
import requests                  # HTTP client: sends the request to Deepgram, returns the response
import sounddevice as sd         # play the result so your ears confirm it's speech
from dotenv import load_dotenv  # reads a .env file's KEY=value lines into the environment

# Reuse Rung 1: the audio format constants (so "ask Deepgram for" == "play/save at"), plus
# the functions that already know how to inspect/save the array.
from rung1_audio import CHANNELS, DTYPE, SAMPLE_RATE, inspect, save_wav

# Deepgram's text-to-speech endpoint and the voice we want.
DEEPGRAM_TTS_URL = "https://api.deepgram.com/v1/speak"
TTS_MODEL = "aura-2-thalia-en"

# Bytes to pull per network read while streaming. The carry-over buffer (take_complete_
# samples) makes correctness independent of this size, so it's just a latency/throughput knob.
CHUNK_SIZE = 4096


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


def synthesize_stream(text: str, api_key: str):
    """Yield linear16 audio in chunks as Deepgram synthesizes it (a generator).

    Same request as synthesize(), but stream=True tells requests not to download the whole
    body first — iter_content hands us pieces as they arrive, so playback can start before
    the clip is finished. Network-touching, so verified by a live run, not a unit test.
    """
    params = {"model": TTS_MODEL, "encoding": "linear16", "sample_rate": SAMPLE_RATE}
    headers = {"Authorization": f"Token {api_key}", "Content-Type": "application/json"}
    # `with` closes the connection even if playback errors out mid-stream.
    with requests.post(DEEPGRAM_TTS_URL, params=params, headers=headers,
                       json={"text": text}, stream=True) as response:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:            # iter_content can emit empty keep-alive chunks; skip them
                yield chunk


def take_complete_samples(buf: bytes) -> tuple[np.ndarray, bytes]:
    """Split a byte buffer into (complete int16 samples, leftover byte).

    linear16 is 2 bytes/sample, but a network chunk can end mid-sample. We convert only the
    largest even prefix and hand back any trailing odd byte, which the caller prepends to the
    next chunk. That carry-over is why no sample is ever invented or misaligned. Pure, so it's
    this pass's unit-tested core.

    Returns:
        (samples, leftover): samples is a (frames, 1) int16 array; leftover is b"" or 1 byte.
    """
    n_whole = len(buf) - (len(buf) % 2)                      # largest even byte count
    samples = np.frombuffer(buf[:n_whole], dtype=np.int16).reshape(-1, 1)
    leftover = buf[n_whole:]                                 # the odd byte, if any
    return samples, leftover


def stream_and_play(text: str, api_key: str) -> np.ndarray:
    """Stream audio from Deepgram and play each chunk as it arrives; return the full array.

    Live playback (the point of streaming) goes through a persistent OutputStream we write()
    into. We also collect the chunks so the caller can save/inspect the whole clip afterward
    — that collection is just for parity with the batch pass, not part of the playback path.
    """
    leftover = b""            # bytes carried over from a chunk that ended mid-sample
    collected: list[np.ndarray] = []
    first_ms = None
    start = time.perf_counter()

    # OutputStream keeps the speaker open across many small writes (vs sd.play's one-shot).
    with sd.OutputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE) as stream:
        for chunk in synthesize_stream(text, api_key):
            if first_ms is None:  # measure time-to-first-audio: the whole reason to stream
                first_ms = (time.perf_counter() - start) * 1000
                print(f"First audio after {first_ms:.0f} ms")
            samples, leftover = take_complete_samples(leftover + chunk)
            stream.write(samples)      # feed the speaker immediately
            collected.append(samples)  # keep a copy so we can save/inspect later

    return np.concatenate(collected) if collected else np.empty((0, 1), dtype=np.int16)


# Streaming demo: play Deepgram's speech as it arrives, then save/inspect the assembled clip
# with Rung 1's functions (same array, just delivered live instead of all at once).
if __name__ == "__main__":
    key = load_api_key()
    print(f"DEEPGRAM_API_KEY loaded ({len(key)} chars)")

    print("Streaming from Deepgram...")
    audio = stream_and_play("Welcome to Tony's Pizza. What can I get started for you?", key)

    inspect(audio, "tts-stream")           # same stats line as Rung 1: shape/dtype/min/max
    save_wav(audio, "tts_output.wav", SAMPLE_RATE)
    print("Saved tts_output.wav. Done.")
