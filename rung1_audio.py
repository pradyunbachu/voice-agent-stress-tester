"""Stage A, Rung 1 — "Audio is numbers".

Record 3s from the mic, inspect the raw array, save it to a WAV, reload it, play it back.
The whole point: prove that sound is just a long array of integers sampled fast, and that
a WAV is that array plus a tiny header.
"""

# Three libraries, three doors the audio moves through. The audio is always the same
# int16 array; these just carry it between memory, hardware, and disk.
import numpy as np       # memory:   holds the samples as an ndarray
import sounddevice as sd # hardware: mic in, speaker out
import soundfile as sf   # disk:     reads/writes WAV (samples + header)

# --- Comparability constants (move into config later, design decision #5) ---
# These three fully describe the number-stream. Everything downstream (STT, TTS) must
# agree on them or the same bytes get misread.
SAMPLE_RATE = 16000   # Hz. 16 kHz is the speech-API standard (Deepgram expects it).
CHANNELS = 1          # mono — voice is one stream of numbers, not stereo.
DTYPE = "int16"       # PCM: each sample is the wave's amplitude as an int in [-32768, 32767].
SECONDS = 3           # demo-only: how long main() records.
WAV_PATH = "recording.wav"  # demo-only: where main() writes.


def record(seconds: int) -> np.ndarray:
    """Record from the mic into an int16 array of shape (frames, channels).

    Touches hardware, so it's not unit-tested (no mic in a test runner, and it blocks).
    """
    frames = int(seconds * SAMPLE_RATE)  # samples needed = duration * rate
    print(f"Recording {seconds}s... speak now.")
    # sd.rec returns immediately and fills the buffer on a background thread; sd.wait()
    # blocks until it's done, so we don't read a half-empty buffer.
    audio = sd.rec(frames, samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE)
    sd.wait()
    print("Done recording.")
    return audio


def audio_stats(audio: np.ndarray) -> dict:
    """Return {shape, dtype, min, max} — the cheapest "did this work?" check.

    Pure function (no hardware, no disk), which is what makes it unit-testable.
    Reading min/max: near +/-32000 is loud/clipping, a few hundred is quiet,
    ~0 is silence (dead mic, wrong device, or no permission).
    """
    return {
        "shape": audio.shape,
        "dtype": str(audio.dtype),
        "min": int(audio.min()),
        "max": int(audio.max()),
    }


def inspect(audio: np.ndarray, label: str) -> dict:
    """Print the stats and return them too (so tests can read values, not stdout).

    Split on purpose: audio_stats() is the pure, tested math; inspect() is the I/O.
    """
    stats = audio_stats(audio)
    print(f"[{label}] shape={stats['shape']} dtype={stats['dtype']} "
          f"min={stats['min']} max={stats['max']}")
    return stats


def save_wav(audio: np.ndarray, path: str, sr: int) -> None:
    """Write samples plus a WAV header (rate/channels/format) to disk.

    The crux of the rung: the in-memory array doesn't know it's 16 kHz — that fact
    lives in a variable. The header bakes it in so the file is self-describing.
    """
    sf.write(path, audio, sr)


def load_wav(path: str, dtype: str = DTYPE) -> tuple[np.ndarray, int]:
    """Reload a WAV into a fresh array; returns (audio, sr).

    dtype defaults to int16 so the reload matches what we wrote. (soundfile would
    otherwise hand back float64, making the round trip look "different" for a boring
    reason and masking whether the real values survived.)
    """
    audio, sr = sf.read(path, dtype=dtype)
    return audio, sr


def main() -> None:
    """record -> inspect -> save -> reload into a NEW array -> inspect -> play.

    The order is the lesson. Reloading into a separate variable (not reusing `audio`)
    is what makes it a real round-trip through disk, not just a memory replay.
    """
    audio = record(SECONDS)
    inspect(audio, "recorded")

    # Save, then reload into a *new* array, to prove the round trip survives disk.
    save_wav(audio, WAV_PATH, SAMPLE_RATE)
    print(f"Saved {WAV_PATH}")

    reloaded, sr = load_wav(WAV_PATH)
    print(f"Reloaded {WAV_PATH} (samplerate={sr})")
    inspect(reloaded, "reloaded")

    # Play the reloaded array so your ears confirm the round trip survived.
    print("Playing back...")
    sd.play(reloaded, SAMPLE_RATE)
    sd.wait()
    print("Done.")


# Run main() only when executed directly, not when imported. This makes the file
# dual-purpose: a runnable script, and an importable library of pure functions for the
# tests. Without the guard, `from rung1_audio import save_wav` would start recording.
if __name__ == "__main__":
    main()
