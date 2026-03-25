"""audio.py — Microphone capture with Silero VAD.

All output (logging, warnings) goes to stderr.  stdout is reserved for the
MCP protocol channel and must stay clean.

Public API
----------
load_vad_model() -> SileroVAD
    Download (first run) and load the Silero VAD ONNX model.  Returns a
    callable SileroVAD object.

record_audio(...) -> np.ndarray | None
    Capture microphone audio until speech ends (or timeout).
    Returns a 16 kHz mono float32 numpy array, or None on no-speech timeout.

VadStateMachine
    State machine that drives WAITING → SPEAKING → TRAILING_SILENCE → DONE.
    Exposed publicly so tests can drive it directly.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16_000          # Hz — the only rate Silero VAD 16kHz model supports
CHUNK_SAMPLES = 512           # samples per VAD frame at 16 kHz  (32 ms)
CONTEXT_SAMPLES = 64          # context prepended to each chunk per Silero spec

_MODELS_DIR = Path(__file__).parent / "models"
_VAD_MODEL_PATH = _MODELS_DIR / "silero_vad.onnx"
_VAD_MODEL_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/master"
    "/src/silero_vad/data/silero_vad.onnx"
)

# Speech probability threshold — above → speech, below → silence
_SPEECH_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# stderr logging helper
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    """Write a log line to stderr (never stdout)."""
    print(f"[lazy-claude audio] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Silero VAD ONNX wrapper (no torch dependency)
# ---------------------------------------------------------------------------

class SileroVAD:
    """Thin ONNX-based wrapper around the Silero VAD model.

    Usage::

        model = SileroVAD(path)
        prob = model(chunk)   # chunk: float32 array of CHUNK_SAMPLES (512)
        model.reset()
    """

    def __init__(self, model_path: Path) -> None:
        import onnxruntime as ort  # lazy import — avoid polluting startup

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3  # suppress ort INFO messages

        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.reset()

    def reset(self) -> None:
        """Reset internal GRU state and context buffer."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)

    def __call__(self, chunk: np.ndarray) -> float:
        """Run VAD on a single chunk.

        Parameters
        ----------
        chunk:
            1-D float32 array of exactly CHUNK_SAMPLES (512) samples.

        Returns
        -------
        float
            Speech probability in [0, 1].
        """
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)
        if chunk.ndim != 1 or chunk.shape[0] != CHUNK_SAMPLES:
            raise ValueError(
                f"chunk must be 1-D with {CHUNK_SAMPLES} samples, got shape {chunk.shape}"
            )

        # Prepend context: shape [1, CONTEXT_SAMPLES + CHUNK_SAMPLES]
        x = np.concatenate([self._context, chunk[np.newaxis, :]], axis=1)

        feed = {
            "input": x,
            "state": self._state,
            "sr": np.array(SAMPLE_RATE, dtype=np.int64),
        }
        output, new_state = self._session.run(None, feed)

        self._state = new_state
        # Update context to last CONTEXT_SAMPLES of the input
        self._context = x[:, -CONTEXT_SAMPLES:]

        return float(output[0, 0])


# ---------------------------------------------------------------------------
# Model loader (download on first use)
# ---------------------------------------------------------------------------

def load_vad_model() -> SileroVAD:
    """Return a ready-to-use SileroVAD instance.

    Downloads the ONNX model on the first call and caches it under
    ``lazy_claude/models/silero_vad.onnx``.
    """
    if not _VAD_MODEL_PATH.exists():
        _log(f"Downloading Silero VAD model → {_VAD_MODEL_PATH} …")
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(_VAD_MODEL_URL, _VAD_MODEL_PATH)
            _log("Download complete.")
        except Exception as exc:
            _log(f"ERROR: failed to download VAD model: {exc}")
            raise

    return SileroVAD(_VAD_MODEL_PATH)


# ---------------------------------------------------------------------------
# VAD State Machine
# ---------------------------------------------------------------------------

State = Literal["WAITING", "SPEAKING", "TRAILING_SILENCE", "DONE"]


class VadStateMachine:
    """Drive audio-capture state based on per-chunk VAD probabilities.

    Parameters
    ----------
    silence_duration:
        Seconds of silence after the last speech chunk required to stop.
    min_speech_duration:
        Minimum accumulated speech seconds before we honour a stop signal.
        Prevents very short bursts (e.g. door click) from ending a recording.
    no_speech_timeout:
        If no speech has started within this many seconds, return done
        (``timed_out`` will be True).
    sample_rate:
        Audio sample rate (default 16000).
    chunk_size:
        Samples per chunk (default 512).
    speech_threshold:
        VAD probability threshold — above is speech.
    """

    def __init__(
        self,
        silence_duration: float = 1.5,
        min_speech_duration: float = 0.5,
        no_speech_timeout: float = 15.0,
        sample_rate: int = SAMPLE_RATE,
        chunk_size: int = CHUNK_SAMPLES,
        speech_threshold: float = _SPEECH_THRESHOLD,
    ) -> None:
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration
        self.no_speech_timeout = no_speech_timeout
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.speech_threshold = speech_threshold

        self.state: State = "WAITING"
        self.timed_out: bool = False

        self._speech_started_at: float | None = None
        self._silence_started_at: float | None = None
        self._accumulated_speech: float = 0.0  # seconds

    def update(self, speech_prob: float, timestamp: float) -> bool:
        """Process one VAD result.

        Parameters
        ----------
        speech_prob:
            Speech probability from the VAD model (0–1).
        timestamp:
            Elapsed time in seconds (from recording start).

        Returns
        -------
        bool
            True when recording should stop (state is DONE).
        """
        is_speech = speech_prob >= self.speech_threshold

        if self.state == "WAITING":
            # No-speech timeout check
            if timestamp >= self.no_speech_timeout:
                self.state = "DONE"
                self.timed_out = True
                return True

            if is_speech:
                self.state = "SPEAKING"
                self._speech_started_at = timestamp
                self._accumulated_speech += self.chunk_size / self.sample_rate

        elif self.state == "SPEAKING":
            if is_speech:
                self._accumulated_speech += self.chunk_size / self.sample_rate
            else:
                self.state = "TRAILING_SILENCE"
                self._silence_started_at = timestamp

        elif self.state == "TRAILING_SILENCE":
            if is_speech:
                # Speech resumed — go back to SPEAKING
                self.state = "SPEAKING"
                self._accumulated_speech += self.chunk_size / self.sample_rate
                self._silence_started_at = None
            else:
                silence_elapsed = timestamp - self._silence_started_at  # type: ignore[operator]
                if (
                    silence_elapsed >= self.silence_duration
                    and self._accumulated_speech >= self.min_speech_duration
                ):
                    self.state = "DONE"
                    return True

        return self.state == "DONE"


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _get_capture_rate() -> tuple[int, bool]:
    """Return (capture_rate, needs_resample).

    Tries 16 kHz first.  If the device doesn't support it, falls back to the
    device's native rate and we'll resample in software.
    """
    import sounddevice as sd  # noqa: PLC0415

    try:
        sd.check_input_settings(samplerate=16_000, channels=1, dtype="float32")
        return 16_000, False
    except Exception:
        pass

    # Query the default input device's default sample rate
    try:
        device_info = sd.query_devices(kind="input")
        native_rate = int(device_info["default_samplerate"])
        _log(
            f"16kHz not supported by input device. "
            f"Capturing at {native_rate} Hz and resampling to 16 kHz."
        )
        return native_rate, True
    except Exception as exc:
        _log(f"WARNING: could not query native sample rate ({exc}), assuming 44100 Hz.")
        return 44_100, True


def _resample_to_16k(audio: np.ndarray, from_rate: int) -> np.ndarray:
    """Simple linear resampling to 16 kHz mono float32."""
    if from_rate == SAMPLE_RATE:
        return audio
    ratio = SAMPLE_RATE / from_rate
    n_out = int(len(audio) * ratio)
    # Use numpy for a simple polyphase-style resampling via linspace
    x_old = np.linspace(0, len(audio) - 1, len(audio))
    x_new = np.linspace(0, len(audio) - 1, n_out)
    return np.interp(x_new, x_old, audio).astype(np.float32)


# ---------------------------------------------------------------------------
# Main recording function
# ---------------------------------------------------------------------------

def record_audio(
    silence_duration: float = 1.5,
    min_speech_duration: float = 0.5,
    no_speech_timeout: float = 15.0,
    vad_model: SileroVAD | None = None,
) -> np.ndarray | None:
    """Record microphone audio until the end-of-speech condition is met.

    Parameters
    ----------
    silence_duration:
        Seconds of post-speech silence to wait before stopping.
    min_speech_duration:
        Minimum seconds of speech required before a stop is honoured.
    no_speech_timeout:
        Stop and return None if no speech starts within this many seconds.
    vad_model:
        Pre-loaded SileroVAD instance.  Loaded automatically if not provided.

    Returns
    -------
    np.ndarray | None
        16 kHz mono float32 audio array, or None if no speech was detected
        before ``no_speech_timeout``.

    Raises
    ------
    RuntimeError
        If microphone permissions are denied or no input device is found.
    """
    import sounddevice as sd  # noqa: PLC0415

    if vad_model is None:
        vad_model = load_vad_model()

    capture_rate, needs_resample = _get_capture_rate()

    # Adjust chunk size proportionally if we're capturing at a different rate
    native_chunk = CHUNK_SAMPLES if not needs_resample else int(
        CHUNK_SAMPLES * capture_rate / SAMPLE_RATE
    )

    vad = VadStateMachine(
        silence_duration=silence_duration,
        min_speech_duration=min_speech_duration,
        no_speech_timeout=no_speech_timeout,
    )
    vad_model.reset()

    recorded_chunks: list[np.ndarray] = []
    done_event = threading.Event()
    start_time = time.monotonic()
    # Running buffer for VAD chunks (needed when native != 16kHz)
    _resample_buffer: list[np.ndarray] = []

    def _callback(indata: np.ndarray, frames: int, _time, status) -> None:
        nonlocal recorded_chunks, _resample_buffer

        if status:
            _log(f"sounddevice status: {status}")

        # indata shape: (frames, channels) — take mono channel
        mono = indata[:, 0].copy()

        if needs_resample:
            # Accumulate until we have enough samples for a 16kHz chunk
            _resample_buffer.append(mono)
            buf = np.concatenate(_resample_buffer)
            # How many native samples correspond to one 16kHz chunk?
            native_per_vad = int(CHUNK_SAMPLES * capture_rate / SAMPLE_RATE)
            while len(buf) >= native_per_vad:
                segment = buf[:native_per_vad]
                buf = buf[native_per_vad:]
                chunk_16k = _resample_to_16k(segment, capture_rate)
                recorded_chunks.append(chunk_16k)
                _run_vad(chunk_16k)
            _resample_buffer = [buf] if len(buf) > 0 else []
        else:
            recorded_chunks.append(mono)
            _run_vad(mono)

    def _run_vad(chunk_16k: np.ndarray) -> None:
        elapsed = time.monotonic() - start_time
        prob = vad_model(chunk_16k)
        done = vad.update(speech_prob=prob, timestamp=elapsed)
        if done:
            done_event.set()

    _log("Listening… (speak now)")

    try:
        with sd.InputStream(
            samplerate=capture_rate,
            channels=1,
            dtype="float32",
            blocksize=native_chunk,
            callback=_callback,
        ):
            done_event.wait(timeout=no_speech_timeout + 5.0)
    except sd.PortAudioError as exc:
        err_str = str(exc).lower()
        if "permission" in err_str or "access" in err_str or "denied" in err_str:
            msg = (
                "Microphone access denied. "
                "Grant permission in System Settings → Privacy & Security → Microphone."
            )
        else:
            msg = f"PortAudio error: {exc}"
        _log(f"ERROR: {msg}")
        raise RuntimeError(msg) from exc

    if vad.timed_out or not recorded_chunks:
        _log("No speech detected within timeout.")
        return None

    audio = np.concatenate(recorded_chunks)
    _log(f"Captured {len(audio) / SAMPLE_RATE:.2f}s of audio.")
    return audio.astype(np.float32)
