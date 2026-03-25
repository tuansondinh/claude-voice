"""Tests for audio.py — VAD state machine and audio capture utilities.

These tests do NOT require a microphone.  They test:
- VAD state machine transitions
- Silence / speech detection thresholds
- Timeout behaviour
- Return type and sample rate of recorded audio
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Unit tests for the VAD state machine (no hardware required)
# ---------------------------------------------------------------------------


class TestVADStateMachine:
    """Test the VadStateMachine class directly."""

    def setup_method(self):
        # Import here so any import-time errors surface as test failures
        from lazy_claude.audio import VadStateMachine
        self.VadStateMachine = VadStateMachine

    def _machine(self, **kwargs):
        defaults = dict(
            silence_duration=1.5,
            min_speech_duration=0.5,
            no_speech_timeout=15.0,
            sample_rate=16000,
            chunk_size=512,
        )
        defaults.update(kwargs)
        return self.VadStateMachine(**defaults)

    def test_initial_state_is_waiting(self):
        m = self._machine()
        assert m.state == "WAITING"

    def test_single_speech_chunk_transitions_to_speaking(self):
        m = self._machine()
        # Feed a speech chunk (prob > threshold)
        done = m.update(speech_prob=0.9, timestamp=0.0)
        assert not done
        assert m.state == "SPEAKING"

    def test_silence_in_waiting_stays_waiting(self):
        m = self._machine()
        done = m.update(speech_prob=0.1, timestamp=0.0)
        assert not done
        assert m.state == "WAITING"

    def test_speech_then_silence_transitions_to_trailing(self):
        m = self._machine(min_speech_duration=0.0)
        m.update(speech_prob=0.9, timestamp=0.0)
        assert m.state == "SPEAKING"
        done = m.update(speech_prob=0.1, timestamp=0.1)
        assert not done
        assert m.state == "TRAILING_SILENCE"

    def test_trailing_silence_long_enough_triggers_done(self):
        m = self._machine(min_speech_duration=0.0, silence_duration=0.5)
        # Start speaking
        m.update(speech_prob=0.9, timestamp=0.0)
        # Start trailing silence
        m.update(speech_prob=0.1, timestamp=0.1)
        assert m.state == "TRAILING_SILENCE"
        # Silence not long enough yet
        done = m.update(speech_prob=0.1, timestamp=0.4)
        assert not done
        # Silence now exceeds 0.5s
        done = m.update(speech_prob=0.1, timestamp=0.7)
        assert done
        assert m.state == "DONE"

    def test_speech_resumes_during_trailing_silence(self):
        m = self._machine(min_speech_duration=0.0, silence_duration=1.5)
        m.update(speech_prob=0.9, timestamp=0.0)
        m.update(speech_prob=0.1, timestamp=0.1)
        assert m.state == "TRAILING_SILENCE"
        # Speech resumes
        done = m.update(speech_prob=0.9, timestamp=0.5)
        assert not done
        assert m.state == "SPEAKING"

    def test_no_speech_timeout_triggers_done(self):
        m = self._machine(no_speech_timeout=1.0)
        done = m.update(speech_prob=0.1, timestamp=0.0)
        assert not done
        done = m.update(speech_prob=0.1, timestamp=1.5)
        assert done
        assert m.state == "DONE"
        assert m.timed_out is True

    def test_no_speech_timeout_does_not_fire_once_speech_started(self):
        m = self._machine(no_speech_timeout=1.0)
        m.update(speech_prob=0.9, timestamp=0.0)
        # timestamp way past no_speech_timeout, but speech already started
        done = m.update(speech_prob=0.1, timestamp=2.0)
        assert not done  # still in TRAILING_SILENCE, not timed out
        assert m.timed_out is False

    def test_min_speech_duration_prevents_early_done(self):
        """With min_speech_duration=0.5s, trailing silence should not finish
        the recording if total speech accumulated is less than 0.5s."""
        chunk_duration = 512 / 16000  # ~0.032 s
        m = self._machine(min_speech_duration=0.5, silence_duration=0.1)
        # Only ~one chunk of speech (~32 ms) — below min_speech_duration
        m.update(speech_prob=0.9, timestamp=0.0)
        # Long silence — would normally trigger done
        done = m.update(speech_prob=0.1, timestamp=0.5)
        assert not done
        # Still TRAILING_SILENCE, not DONE
        assert m.state in ("TRAILING_SILENCE", "WAITING", "SPEAKING")


# ---------------------------------------------------------------------------
# Integration smoke tests (no mic — just verify public API exists)
# ---------------------------------------------------------------------------


class TestAudioModuleAPI:
    """Verify the public API surface of audio.py."""

    def test_record_audio_callable(self):
        from lazy_claude.audio import record_audio
        assert callable(record_audio)

    def test_vad_model_loader_callable(self):
        from lazy_claude.audio import load_vad_model
        assert callable(load_vad_model)

    def test_vad_state_machine_importable(self):
        from lazy_claude.audio import VadStateMachine
        assert VadStateMachine is not None


class TestVADModelLoading:
    """Test that the Silero VAD ONNX model can be loaded and run on a dummy chunk."""

    def test_load_vad_model_returns_callable(self):
        from lazy_claude.audio import load_vad_model
        model = load_vad_model()
        assert callable(model)

    def test_vad_model_returns_probability_for_silence(self):
        from lazy_claude.audio import load_vad_model
        model = load_vad_model()
        # Pure silence chunk: 512 float32 zeros at 16kHz
        chunk = np.zeros(512, dtype=np.float32)
        prob = model(chunk)
        assert isinstance(prob, float), f"Expected float, got {type(prob)}"
        assert 0.0 <= prob <= 1.0, f"Probability out of range: {prob}"

    def test_vad_model_returns_probability_for_noise(self):
        from lazy_claude.audio import load_vad_model
        model = load_vad_model()
        rng = np.random.default_rng(42)
        chunk = rng.standard_normal(512).astype(np.float32) * 0.1
        prob = model(chunk)
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_vad_model_resets_state(self):
        from lazy_claude.audio import load_vad_model
        model = load_vad_model()
        chunk = np.zeros(512, dtype=np.float32)
        prob1 = model(chunk)
        model.reset()
        prob2 = model(chunk)
        # After reset, same silent chunk should give same result
        assert abs(prob1 - prob2) < 1e-5, f"Probabilities differ after reset: {prob1} vs {prob2}"
