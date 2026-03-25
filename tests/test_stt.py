"""Tests for stt.py — Whisper.cpp STT engine.

These tests cover:
- Public API surface
- transcribe() with empty / silent audio returns ""
- transcribe() returns a string for normal audio
- Artifact stripping (BLANK_AUDIO tokens, leading/trailing whitespace)
- Model configurability (model name parameter)
- All output goes to stderr (stdout stays clean)
"""

from __future__ import annotations

import subprocess
import sys
import tempfile

import numpy as np
import pytest

SAMPLE_RATE = 16_000


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _silence(seconds: float = 1.0) -> np.ndarray:
    """Return a silent 16 kHz float32 array."""
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)


def _noise(seconds: float = 0.5) -> np.ndarray:
    """Return low-amplitude noise at 16 kHz float32."""
    rng = np.random.default_rng(0)
    return (rng.standard_normal(int(SAMPLE_RATE * seconds)) * 0.01).astype(np.float32)


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


class TestSTTModuleAPI:
    """Verify the public API of stt.py exists and is callable."""

    def test_transcribe_callable(self):
        from lazy_claude.stt import transcribe
        assert callable(transcribe)

    def test_load_model_callable(self):
        from lazy_claude.stt import load_model
        assert callable(load_model)


# ---------------------------------------------------------------------------
# Empty / silent audio
# ---------------------------------------------------------------------------


class TestTranscribeSilentAudio:
    """transcribe() must return empty string for empty or silent input."""

    def test_empty_array_returns_empty_string(self):
        from lazy_claude.stt import transcribe
        audio = np.array([], dtype=np.float32)
        result = transcribe(audio)
        assert result == "", f"Expected '', got: {result!r}"

    def test_zero_samples_returns_empty_string(self):
        from lazy_claude.stt import transcribe
        audio = np.zeros(0, dtype=np.float32)
        result = transcribe(audio)
        assert result == ""

    def test_silence_returns_string(self):
        """Pure silence should return a string (possibly empty after artifact stripping)."""
        from lazy_claude.stt import transcribe
        audio = _silence(1.0)
        result = transcribe(audio)
        assert isinstance(result, str)

    def test_transcribe_returns_str_type(self):
        from lazy_claude.stt import transcribe
        result = transcribe(_silence(0.5))
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Artifact stripping
# ---------------------------------------------------------------------------


class TestArtifactStripping:
    """Verify that common whisper hallucination tokens are stripped."""

    def test_strip_blank_audio_token(self):
        from lazy_claude.stt import _strip_artifacts
        assert _strip_artifacts("[BLANK_AUDIO]") == ""

    def test_strip_blank_audio_token_with_whitespace(self):
        from lazy_claude.stt import _strip_artifacts
        assert _strip_artifacts("  [BLANK_AUDIO]  ") == ""

    def test_strip_leading_trailing_whitespace(self):
        from lazy_claude.stt import _strip_artifacts
        assert _strip_artifacts("  hello world  ") == "hello world"

    def test_strip_multiple_blank_audio_tokens(self):
        from lazy_claude.stt import _strip_artifacts
        assert _strip_artifacts("[BLANK_AUDIO] [BLANK_AUDIO]") == ""

    def test_strip_music_token(self):
        from lazy_claude.stt import _strip_artifacts
        # Common whisper hallucination for background music
        assert _strip_artifacts("[MUSIC]") == ""

    def test_preserve_normal_text(self):
        from lazy_claude.stt import _strip_artifacts
        assert _strip_artifacts("hello world") == "hello world"

    def test_strip_repeated_phrases(self):
        from lazy_claude.stt import _strip_artifacts
        # Whisper sometimes hallucinates repeated phrases
        result = _strip_artifacts("hello hello hello hello hello")
        # The result should not contain 5 consecutive repetitions of the same word
        assert result != "hello hello hello hello hello" or result == ""

    def test_strip_thank_you_for_watching_artifact(self):
        from lazy_claude.stt import _strip_artifacts
        # Common hallucination at end of audio
        assert _strip_artifacts("Thank you for watching.") == ""

    def test_strip_subscribe_artifact(self):
        from lazy_claude.stt import _strip_artifacts
        assert _strip_artifacts("Subscribe to our channel.") == ""

    def test_normal_speech_not_stripped(self):
        from lazy_claude.stt import _strip_artifacts
        text = "open the file in vim"
        assert _strip_artifacts(text) == text


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


class TestLoadModel:
    """Test load_model() returns a usable model object."""

    def test_load_default_model(self):
        from lazy_claude.stt import load_model
        model = load_model()
        assert model is not None

    def test_load_model_returns_same_type(self):
        from lazy_claude.stt import load_model
        m1 = load_model()
        m2 = load_model()
        assert type(m1) == type(m2)

    def test_load_model_custom_name(self):
        """load_model('base.en') should work the same as default."""
        from lazy_claude.stt import load_model
        model = load_model("base.en")
        assert model is not None


# ---------------------------------------------------------------------------
# Transcribe with a loaded model
# ---------------------------------------------------------------------------


class TestTranscribeWithModel:
    """Pass a pre-loaded model to avoid repeated downloads."""

    def setup_method(self):
        from lazy_claude.stt import load_model
        self.model = load_model()

    def test_transcribe_silence_with_model(self):
        from lazy_claude.stt import transcribe
        result = transcribe(_silence(1.0), model=self.model)
        assert isinstance(result, str)

    def test_transcribe_noise_with_model(self):
        from lazy_claude.stt import transcribe
        result = transcribe(_noise(1.0), model=self.model)
        assert isinstance(result, str)

    def test_transcribe_empty_with_model(self):
        from lazy_claude.stt import transcribe
        result = transcribe(np.array([], dtype=np.float32), model=self.model)
        assert result == ""

    def test_transcribe_does_not_write_to_stdout(self):
        """transcribe() must not emit anything on stdout."""
        code = """
import numpy as np
from lazy_claude.stt import transcribe, load_model
model = load_model()
audio = np.zeros(16000, dtype=np.float32)
result = transcribe(audio, model=model)
"""
        with tempfile.TemporaryFile() as cap:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                stdout=cap,
                stderr=subprocess.DEVNULL,
                cwd="/Users/sonwork/Workspace/lazy-claude",
            )
            cap.seek(0)
            out = cap.read()
        assert proc.returncode == 0, "transcribe() must not crash"
        assert out == b"", f"Expected no stdout, got: {out!r}"


# ---------------------------------------------------------------------------
# stdout cleanliness
# ---------------------------------------------------------------------------


class TestStdoutClean:
    """Importing and using stt module must not pollute stdout."""

    def test_import_stt_produces_no_stdout(self):
        with tempfile.TemporaryFile() as cap:
            proc = subprocess.run(
                [sys.executable, "-c", "from lazy_claude import stt"],
                stdout=cap,
                stderr=subprocess.DEVNULL,
                cwd="/Users/sonwork/Workspace/lazy-claude",
            )
            cap.seek(0)
            out = cap.read()
        assert proc.returncode == 0
        assert out == b"", f"Expected no stdout on import, got: {out!r}"
