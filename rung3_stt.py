"""Stage A, Rung 3 — STT alone (Deepgram), batch version.

The mirror of Rung 2: instead of text -> audio, we send audio -> get text back.
Read a WAV file's bytes, POST them to Deepgram's /listen endpoint, and dig the
transcript string out of the JSON response.

  transcribe()        — POST audio bytes, return the parsed JSON (network, live-verified).
  extract_transcript() — pull results->channels[0]->alternatives[0]->transcript out of that
                         JSON (pure, unit-tested).
"""

import asyncio
import json
import ssl
import sys
from typing import NamedTuple
from urllib.parse import urlencode

import certifi     # bundled CA certificates — see SSL_CONTEXT below for why we need them
import requests    # HTTP client for batch: sends the audio, returns the JSON response
import websockets  # async websocket client for streaming: the persistent wss:// connection

# This Python build ships without CA root certificates in its default SSL store, so verifying
# a wss:// (TLS) server fails with CERTIFICATE_VERIFY_FAILED. requests dodges this by bundling
# certifi; websockets uses the empty default store, so we build a context from certifi ourselves
# and pass it to connect(). (Batch STT over https worked precisely because requests bundles this.)
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

# Reuse the same Deepgram key loader we wrote for TTS — it's the same key. (If this cross-rung
# import grows, the clean fix later is a shared config module.) Also reuse Rung 1's WAV loader
# and audio-format constants so streaming sends the exact format it claims to.
from rung1_audio import CHANNELS, SAMPLE_RATE, load_wav
from rung2_tts import load_api_key

# Deepgram's speech-to-text endpoint and the model. nova-3 is their current best English model;
# smart_format adds punctuation/capitalization so the transcript reads like real text.
DEEPGRAM_STT_URL = "https://api.deepgram.com/v1/listen"
DEEPGRAM_STT_WS_URL = "wss://api.deepgram.com/v1/listen"  # streaming uses wss:// (websocket), not https
STT_MODEL = "nova-3"

# Streaming pacing: send this many milliseconds of audio, then sleep that long, so Deepgram
# receives the WAV at ~real-time speed. Without the sleep there'd be no interim results or
# endpointing (see design notes) — the whole file would arrive at once and only finals come back.
STREAM_CHUNK_MS = 100


def transcribe(audio_bytes: bytes, api_key: str) -> dict:
    """POST raw WAV bytes to Deepgram and return the parsed JSON response.

    The flip from TTS: here the request *body* is the audio itself (data=audio_bytes), and
    Content-Type: audio/wav tells Deepgram to read the WAV header for the rate/format. Touches
    the network, so like synthesize() it's verified by a live run, not a unit test.
    """
    params = {"model": STT_MODEL, "smart_format": "true"}
    headers = {"Authorization": f"Token {api_key}", "Content-Type": "audio/wav"}

    response = requests.post(DEEPGRAM_STT_URL, params=params, headers=headers, data=audio_bytes)
    response.raise_for_status()  # bad key / bad audio -> raise instead of parsing an error body
    return response.json()       # Deepgram's structured result (see extract_transcript for shape)


def extract_transcript(response: dict) -> str:
    """Dig the transcript string out of Deepgram's nested JSON.

    Deepgram nests results by channel (multi-channel audio) and by alternative (ranked
    guesses, best first). For our mono, single-guess case the text always lives at
    results -> channels[0] -> alternatives[0] -> transcript. Pure (dict in, string out), so
    this is the unit-tested core.

    Returns "" when there's no transcript (e.g. silence gives an empty alternatives list),
    so a dead-air WAV yields a clean empty string instead of an IndexError.
    """
    channels = response.get("results", {}).get("channels", [])
    if not channels:
        return ""
    alternatives = channels[0].get("alternatives", [])
    if not alternatives:
        return ""
    return alternatives[0].get("transcript", "")


class STTResult(NamedTuple):
    """One parsed streaming transcript message.

    text:         the words Deepgram heard for this span.
    is_final:     False = a live guess that may still change; True = this span is settled.
    speech_final: True when Deepgram detects the speaker paused/stopped (endpointing) — the
                  "your turn to respond" signal.
    """
    text: str
    is_final: bool
    speech_final: bool


def parse_stt_message(msg: dict) -> STTResult:
    """Pull text + the two flags out of one Deepgram streaming message.

    Streaming nests differently from batch: the transcript is at channel.alternatives[0]
    .transcript (channel is singular here), and is_final/speech_final sit at the top level.
    Non-transcript messages (Metadata, SpeechStarted, ...) have no channel, so text is "".

    Pure (dict in, STTResult out), so this is the streaming pass's unit-tested core.
    """
    alternatives = msg.get("channel", {}).get("alternatives", [])
    text = alternatives[0].get("transcript", "") if alternatives else ""
    return STTResult(
        text=text,
        is_final=bool(msg.get("is_final", False)),
        speech_final=bool(msg.get("speech_final", False)),
    )


async def send_audio(ws, path: str) -> None:
    """Stream a WAV's linear16 bytes up in ~real-time chunks, then signal end-of-audio.

    Reuses Rung 1's load_wav to get the int16 array, then .tobytes() back to the raw PCM
    Deepgram expects. The asyncio.sleep between chunks is what paces it to real time — the
    reason interim results and endpointing exist at all (see STREAM_CHUNK_MS).
    """
    audio, _ = load_wav(path)                       # WAV -> (frames, 1) int16 array
    raw = audio.tobytes()                           # -> raw linear16 byte stream
    bytes_per_chunk = int(SAMPLE_RATE * CHANNELS * 2 * STREAM_CHUNK_MS / 1000)  # 2 bytes/sample

    for start in range(0, len(raw), bytes_per_chunk):
        await ws.send(raw[start:start + bytes_per_chunk])  # a binary frame = audio
        await asyncio.sleep(STREAM_CHUNK_MS / 1000)        # pause -> lets receive_results run

    # A text frame telling Deepgram no more audio is coming, so it flushes the last final.
    await ws.send(json.dumps({"type": "CloseStream"}))


async def receive_results(ws) -> None:
    """Print each transcript as it arrives, marking interim vs final and endpoints.

    `async for` yields messages until Deepgram closes the socket (after CloseStream). Each
    message goes through Piece 1's parse_stt_message, so all the JSON-shape logic is reused.
    """
    async for message in ws:
        result = parse_stt_message(json.loads(message))
        if not result.text:                 # skip metadata / keep-alive messages
            continue
        marker = "FINAL  " if result.is_final else "interim"
        print(f"[{marker}] {result.text}")
        if result.speech_final:             # endpointing: Deepgram thinks the speaker stopped
            print("           -- endpoint (speaker paused) --")


async def stream_wav(path: str, api_key: str) -> None:
    """Open the websocket and run the send + receive coroutines concurrently on it."""
    params = {
        "model": STT_MODEL,
        "encoding": "linear16",     # we send raw PCM, so we must declare the format/rate
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "interim_results": "true",  # ask for the live guesses, not just finals
        "smart_format": "true",
    }
    url = f"{DEEPGRAM_STT_WS_URL}?{urlencode(params)}"
    headers = {"Authorization": f"Token {api_key}"}  # same auth as the batch REST call

    # additional_headers is the websockets>=14 name for per-connection headers; ssl points at
    # certifi's CA bundle so TLS verification succeeds on this Python build.
    async with websockets.connect(url, additional_headers=headers, ssl=SSL_CONTEXT) as ws:
        # gather runs both coroutines on one event loop: while one awaits the network, the
        # other runs. send_audio finishes first; receive_results runs until the socket closes.
        await asyncio.gather(send_audio(ws, path), receive_results(ws))


# Streaming demo: send a WAV (default recording.wav, or pass one as an arg) to Deepgram over a
# websocket and watch the transcript build live — interim guesses, then finals, then endpoints.
# Try `python rung3_stt.py tts_output.wav` to stream Deepgram its own A2 voice.
if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "recording.wav"

    key = load_api_key()
    print(f"Streaming {path} to Deepgram (interim + final results)...")
    asyncio.run(stream_wav(path, key))  # asyncio.run starts the event loop and drives it
    print("Done.")
