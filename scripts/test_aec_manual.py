#!/usr/bin/env python3
"""scripts/test_aec_manual.py — Manual AEC before/after comparison tool.

Plays a WAV file through speakers while recording from the microphone.
Saves three audio files:
  - raw_mic.wav       : unprocessed microphone signal
  - cleaned_aec.wav   : after adaptive filter (PBFDLMS)
  - cleaned_res.wav   : after adaptive filter + residual echo suppression (RES)

Usage
-----
    python scripts/test_aec_manual.py [input.wav] [--duration SECONDS]

If no input WAV is given, a synthetic 440 Hz sine tone is generated and played.

After the run, inspect the three output WAVs with Audacity or any audio player
to verify that the echo is substantially reduced in the cleaned outputs while
speech recorded during playback is preserved.

Requirements
------------
- sounddevice (pip install sounddevice)
- soundfile  (pip install soundfile)
- numpy      (pip install numpy)
"""

from __future__ import annotations

import argparse
import sys
import time
import threading
from pathlib import Path

import numpy as np

# Ensure the project root is on the path when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16_000       # processing rate for AEC
CHUNK = 512                # AEC block size
DEFAULT_DURATION = 5.0     # seconds of recording when no WAV supplied


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(f"[test-aec] {msg}", flush=True)


def _generate_sine(freq: float, duration: float, sr: int, amp: float = 0.5) -> np.ndarray:
    """Generate a sine wave at *freq* Hz for *duration* seconds."""
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _load_wav(path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load a WAV file and resample to target_sr if needed."""
    import soundfile as sf
    data, sr = sf.read(path, dtype='float32', always_2d=False)
    if data.ndim > 1:
        data = data[:, 0]  # take first channel
    if sr != target_sr:
        _log(f"Resampling {sr} Hz → {target_sr} Hz")
        n_out = int(len(data) * target_sr / sr)
        x_old = np.linspace(0, len(data) - 1, len(data))
        x_new = np.linspace(0, len(data) - 1, n_out)
        data = np.interp(x_new, x_old, data).astype(np.float32)
    return data


def _save_wav(path: str, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    """Save a mono float32 array to a WAV file."""
    import soundfile as sf
    sf.write(path, audio, sr)
    _log(f"Saved: {path} ({len(audio) / sr:.2f}s)")


# ---------------------------------------------------------------------------
# Playback thread
# ---------------------------------------------------------------------------

def _play_audio(audio: np.ndarray, sr: int, stop_event: threading.Event) -> None:
    """Play *audio* through the default speaker (blocking until done or stopped)."""
    import sounddevice as sd

    try:
        with sd.OutputStream(samplerate=sr, channels=1, dtype='float32') as stream:
            chunk_size = 1024
            offset = 0
            while offset < len(audio) and not stop_event.is_set():
                chunk = audio[offset:offset + chunk_size]
                if len(chunk) == 0:
                    break
                stream.write(chunk.reshape(-1, 1))
                offset += len(chunk)
    except Exception as exc:
        _log(f"WARNING: Playback error: {exc}")


# ---------------------------------------------------------------------------
# Main recording + AEC pipeline
# ---------------------------------------------------------------------------

def run(
    tts_audio: np.ndarray,
    duration: float,
    out_dir: Path,
) -> None:
    """Play tts_audio through speakers while recording + AEC-processing the mic.

    Parameters
    ----------
    tts_audio:
        Audio to play (float32, 16kHz mono). This is the known TTS reference.
    duration:
        How many seconds to record (should be >= len(tts_audio) / SAMPLE_RATE).
    out_dir:
        Directory to write output WAV files.
    """
    import sounddevice as sd
    from lazy_claude.aec import ReferenceBuffer, EchoCanceller

    out_dir.mkdir(parents=True, exist_ok=True)

    n_chunks = int(np.ceil(duration * SAMPLE_RATE / CHUNK))

    # ── AEC pipeline setup ──
    ref_buf = ReferenceBuffer(
        capacity=max(65536, len(tts_audio) * 2),
        write_sr=SAMPLE_RATE,
        read_sr=SAMPLE_RATE,
    )
    ec_plain = EchoCanceller(
        filter_length=4800, mu=0.05, chunk_size=CHUNK, enable_res=False
    )
    ec_res = EchoCanceller(
        filter_length=4800, mu=0.05, chunk_size=CHUNK, enable_res=True
    )

    # Write full TTS reference into buffer before playback starts.
    # In a real system this happens chunk by chunk during playback;
    # for the manual test we pre-fill to ensure alignment.
    ref_buf.write(tts_audio)

    # Storage
    raw_mic_chunks: list[np.ndarray] = []
    cleaned_aec_chunks: list[np.ndarray] = []
    cleaned_res_chunks: list[np.ndarray] = []

    done_event = threading.Event()
    chunk_count = [0]

    def mic_callback(indata: np.ndarray, frames: int, _time_info, status) -> None:
        if status:
            _log(f"  mic status: {status}")
        mono = indata[:, 0].copy()

        # Ensure exactly CHUNK samples
        if len(mono) != CHUNK:
            return

        ref_chunk = ref_buf.read(CHUNK)

        raw_mic_chunks.append(mono)
        cleaned_aec_chunks.append(ec_plain.cancel(mono.copy(), ref_chunk.copy()))
        cleaned_res_chunks.append(ec_res.cancel(mono.copy(), ref_chunk))

        chunk_count[0] += 1
        if chunk_count[0] >= n_chunks:
            done_event.set()

    # ── Start playback in a background thread ──
    stop_event = threading.Event()
    play_thread = threading.Thread(
        target=_play_audio,
        args=(tts_audio, SAMPLE_RATE, stop_event),
        daemon=True,
    )

    _log(f"Recording {duration:.1f}s at {SAMPLE_RATE} Hz (CHUNK={CHUNK})…")
    _log("  Playing reference audio through speakers simultaneously.")
    _log("  Speak while audio plays to test barge-in preservation.\n")

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocksize=CHUNK,
            callback=mic_callback,
        ):
            play_thread.start()
            done_event.wait(timeout=duration + 5.0)
    except sd.PortAudioError as exc:
        _log(f"ERROR: microphone unavailable: {exc}")
        return
    finally:
        stop_event.set()
        play_thread.join(timeout=2.0)

    _log(f"\nCaptured {chunk_count[0]} chunks ({chunk_count[0] * CHUNK / SAMPLE_RATE:.2f}s)")

    if not raw_mic_chunks:
        _log("No audio captured — check microphone permissions.")
        return

    raw_mic = np.concatenate(raw_mic_chunks).astype(np.float32)
    cleaned_aec = np.concatenate(cleaned_aec_chunks).astype(np.float32)
    cleaned_res = np.concatenate(cleaned_res_chunks).astype(np.float32)

    # ── Save outputs ──
    _save_wav(str(out_dir / "raw_mic.wav"), raw_mic)
    _save_wav(str(out_dir / "cleaned_aec.wav"), cleaned_aec)
    _save_wav(str(out_dir / "cleaned_res.wav"), cleaned_res)

    # ── Print diagnostics ──
    _log("\n=== Signal levels (RMS dBFS) ===")
    def rms_db(s: np.ndarray) -> str:
        rms = np.sqrt(np.mean(s.astype(np.float64) ** 2))
        if rms < 1e-12:
            return "-inf"
        return f"{20 * np.log10(rms):.1f}"

    _log(f"  Raw mic:      {rms_db(raw_mic)} dBFS")
    _log(f"  AEC cleaned:  {rms_db(cleaned_aec)} dBFS")
    _log(f"  RES cleaned:  {rms_db(cleaned_res)} dBFS")

    _log("\n=== Estimated echo attenuation (raw → cleaned) ===")
    raw_pwr = float(np.mean(raw_mic.astype(np.float64) ** 2))
    aec_pwr = float(np.mean(cleaned_aec.astype(np.float64) ** 2))
    res_pwr = float(np.mean(cleaned_res.astype(np.float64) ** 2))

    if raw_pwr > 1e-12:
        atten_aec = 10 * np.log10(raw_pwr / (aec_pwr + 1e-12))
        atten_res = 10 * np.log10(raw_pwr / (res_pwr + 1e-12))
        _log(f"  AEC attenuation: {atten_aec:.1f} dB")
        _log(f"  AEC+RES attenuation: {atten_res:.1f} dB")
    else:
        _log("  (Mic was silent — no echo to measure)")

    _log(f"\nOutput files written to: {out_dir.resolve()}")
    _log("Open the WAV files in Audacity to inspect before/after visually.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AEC before/after comparison tool. "
                    "Plays audio through speakers while recording the mic, then saves "
                    "raw + AEC-cleaned + RES-cleaned WAV files for inspection."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Path to input WAV file to play. "
             "If omitted, a synthetic 440 Hz tone is generated.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION,
        help=f"Recording duration in seconds (default: {DEFAULT_DURATION}). "
             "Ignored if an input WAV is provided (duration matches WAV length).",
    )
    parser.add_argument(
        "--out-dir",
        default="aec_test_output",
        help="Directory for output WAV files (default: ./aec_test_output/).",
    )

    args = parser.parse_args()

    if args.input:
        _log(f"Loading: {args.input}")
        tts_audio = _load_wav(args.input, target_sr=SAMPLE_RATE)
        duration = len(tts_audio) / SAMPLE_RATE + 1.0  # record slightly longer
        _log(f"Audio: {len(tts_audio)} samples, {duration - 1.0:.2f}s")
    else:
        _log(f"No input WAV provided — generating {args.duration:.1f}s synthetic 440 Hz tone.")
        tts_audio = _generate_sine(440.0, args.duration, sr=SAMPLE_RATE, amp=0.5)
        duration = args.duration

    run(
        tts_audio=tts_audio,
        duration=duration,
        out_dir=Path(args.out_dir),
    )


if __name__ == "__main__":
    main()
