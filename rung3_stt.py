"""Stage A, Rung 3 — STT alone (Deepgram), batch version.

The mirror of Rung 2: instead of text -> audio, we send audio -> get text back.
Read a WAV file's bytes, POST them to Deepgram's /listen endpoint, and dig the
transcript string out of the JSON response.

  transcribe()        — POST audio bytes, return the parsed JSON (network, live-verified).
  extract_transcript() — pull results->channels[0]->alternatives[0]->transcript out of that
                         JSON (pure, unit-tested).
"""

import sys

import requests  # HTTP client: sends the audio to Deepgram, returns the JSON response

# Reuse the same Deepgram key loader we wrote for TTS — it's the same key. (If this cross-rung
# import grows, the clean fix later is a shared config module.)
from rung2_tts import load_api_key

# Deepgram's speech-to-text endpoint and the model. nova-3 is their current best English model;
# smart_format adds punctuation/capitalization so the transcript reads like real text.
DEEPGRAM_STT_URL = "https://api.deepgram.com/v1/listen"
STT_MODEL = "nova-3"


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


# Batch demo: read a WAV (default recording.wav, or pass one as an arg), transcribe it, print
# the text. Try `python rung3_stt.py tts_output.wav` to feed Deepgram its own A2 voice.
if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "recording.wav"

    key = load_api_key()
    with open(path, "rb") as f:      # send the WAV file's raw bytes, header and all
        audio_bytes = f.read()
    print(f"Transcribing {path} ({len(audio_bytes)} bytes)...")

    result = transcribe(audio_bytes, key)
    transcript = extract_transcript(result)
    print(f"Transcript: {transcript!r}")
