"""stt.py — Speech-to-text via whisper.cpp (pywhispercpp).

All output (logging, progress) goes to stderr.  stdout is reserved for the
MCP protocol channel and must stay clean.

Public API
----------
load_model(model_name: str = "base.en") -> Model
    Download (first run) and load a whisper.cpp model.  Returns a
    pywhispercpp.model.Model instance.

transcribe(audio: np.ndarray, model: Model | None = None) -> str
    Transcribe a 16 kHz mono float32 numpy array.  Returns the recognised
    text with artifacts stripped, or an empty string for silent/empty input.

_strip_artifacts(text: str) -> str
    Remove common whisper hallucination tokens and extraneous whitespace.
    Exposed for testing.
"""

from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pywhispercpp.model import Model

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "base.en"

# Minimum number of samples required to attempt transcription.
# Whisper needs at least ~0.1 s of audio to be meaningful.
_MIN_SAMPLES = 1_600  # 0.1 s at 16 kHz

# Known whisper hallucination patterns (case-insensitive).
# These appear when the model receives silence or very short audio.
_ARTIFACT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\[BLANK_AUDIO\]", re.IGNORECASE),
    re.compile(r"\[MUSIC\]", re.IGNORECASE),
    re.compile(r"\[NOISE\]", re.IGNORECASE),
    re.compile(r"\[SILENCE\]", re.IGNORECASE),
    re.compile(r"\( silence \)", re.IGNORECASE),
    # YouTube / podcast hallucinations
    re.compile(r"Thank you for watching\.?", re.IGNORECASE),
    re.compile(r"Subscribe to our channel\.?", re.IGNORECASE),
    re.compile(r"Please subscribe\.?", re.IGNORECASE),
    re.compile(r"Like and subscribe\.?", re.IGNORECASE),
]

# If a single word repeats more than this many consecutive times, treat it as
# a hallucination loop and collapse / drop it.
_MAX_WORD_REPEAT = 4


# ---------------------------------------------------------------------------
# stderr logging helper
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    """Write a log line to stderr (never stdout)."""
    print(f"[lazy-claude stt] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Artifact stripping
# ---------------------------------------------------------------------------


def _strip_artifacts(text: str) -> str:
    """Remove known whisper hallucination tokens and clean up whitespace.

    Parameters
    ----------
    text:
        Raw transcription string from whisper.

    Returns
    -------
    str
        Cleaned text, possibly empty.
    """
    # Apply each hallucination pattern
    for pattern in _ARTIFACT_PATTERNS:
        text = pattern.sub("", text)

    # Collapse repeated words: "hello hello hello hello hello" → drop entirely
    # Match any word repeated more than _MAX_WORD_REPEAT times consecutively.
    def _collapse_repeats(m: re.Match[str]) -> str:
        word = m.group(1)
        # Keep a single occurrence only if it looks like real speech context
        # (i.e. surrounded by other text); in isolation it's noise — drop it.
        return ""

    repeat_re = re.compile(
        r"\b(\w+)(?:\s+\1){" + str(_MAX_WORD_REPEAT) + r",}\b",
        re.IGNORECASE,
    )
    text = repeat_re.sub(_collapse_repeats, text)

    # Strip leading / trailing whitespace
    return text.strip()


# ---------------------------------------------------------------------------
# Model loader (download on first use)
# ---------------------------------------------------------------------------


def load_model(model_name: str = _DEFAULT_MODEL) -> "Model":
    """Return a ready-to-use pywhispercpp Model instance.

    Downloads the GGML model on the first call and caches it in the
    pywhispercpp default models directory (platform user-data dir).

    Parameters
    ----------
    model_name:
        One of the pywhispercpp AVAILABLE_MODELS, e.g. ``"base.en"``,
        ``"small"``, ``"medium.en"``.  Default is ``"base.en"``.

    Returns
    -------
    Model
        A loaded pywhispercpp.model.Model instance, ready for transcription.
    """
    # Import lazily so the module can be imported without loading whisper.
    from pywhispercpp.model import Model  # noqa: PLC0415

    _log(f"Loading whisper model '{model_name}' …")

    # redirect_whispercpp_logs_to=sys.stderr keeps all C++ output off stdout.
    model = Model(
        model_name,
        redirect_whispercpp_logs_to=sys.stderr,
        print_progress=False,
        print_realtime=False,
        print_timestamps=False,
        print_special=False,
    )
    _log(f"Model '{model_name}' ready.")
    return model


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


def transcribe(
    audio: np.ndarray,
    *,
    model: "Model | None" = None,
    model_name: str = _DEFAULT_MODEL,
) -> str:
    """Transcribe a 16 kHz mono float32 audio array.

    Parameters
    ----------
    audio:
        1-D float32 numpy array sampled at 16 kHz.
    model:
        Pre-loaded Model instance.  If not provided, one is loaded via
        ``load_model(model_name)``.
    model_name:
        Model name to use when ``model`` is not provided.

    Returns
    -------
    str
        Recognised text, stripped of artefacts.  Returns ``""`` for
        empty or effectively silent input.
    """
    # Guard: nothing to transcribe
    if audio is None or len(audio) == 0:
        return ""

    audio = np.asarray(audio, dtype=np.float32)

    if len(audio) < _MIN_SAMPLES:
        _log(f"Audio too short ({len(audio)} samples < {_MIN_SAMPLES}), returning empty.")
        return ""

    if model is None:
        model = load_model(model_name)

    _log(f"Transcribing {len(audio) / 16_000:.2f}s of audio …")

    try:
        segments = model.transcribe(
            audio,
            print_progress=False,
            print_realtime=False,
            print_timestamps=False,
            print_special=False,
        )
    except Exception as exc:  # noqa: BLE001
        _log(f"ERROR during transcription: {exc}")
        return ""

    # Join all segment texts
    raw = " ".join(seg.text for seg in segments if seg.text)

    result = _strip_artifacts(raw)
    _log(f"Transcription: {result!r}")
    return result
