"""Tests for Phase 3: Porcupine wake word + wake-word-only mode.

All tests mock pvporcupine so no real Porcupine library or access key is needed.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch, call
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_porcupine(keyword_return: int = 0):
    """Return a mock Porcupine object that returns keyword_return from process()."""
    mock = MagicMock()
    mock.process = MagicMock(return_value=keyword_return)
    mock.delete = MagicMock()
    return mock


def _make_vad_model(speech_prob: float = 0.0):
    """Return a mock VAD model returning a fixed speech probability."""
    mock = MagicMock()
    mock.return_value = speech_prob
    return mock


def _make_listener_with_porcupine(
    porcupine_mock=None,
    vad_speech_prob: float = 0.0,
    wake_word_only: bool | None = None,
):
    """Build a MacOSContinuousListener with Porcupine mocked in.

    Returns (listener, porcupine_mock).
    """
    from lazy_claude.av_audio import MacOSContinuousListener

    if porcupine_mock is None:
        porcupine_mock = _make_mock_porcupine(keyword_return=0)

    backend_mock = MagicMock()
    backend_mock.install_mic_tap = MagicMock()
    backend_mock.play_audio = MagicMock()
    backend_mock.shutdown = MagicMock()

    vad_model = _make_vad_model(vad_speech_prob)

    # Patch pvporcupine.create so __init__ picks it up
    mock_pvporcupine = MagicMock()
    mock_pvporcupine.create = MagicMock(return_value=porcupine_mock)

    with patch.dict('os.environ', {
        'PORCUPINE_ACCESS_KEY': 'test-key',
        'PORCUPINE_MODEL_PATH': '/fake/model.ppn',
    }), patch.dict('sys.modules', {'pvporcupine': mock_pvporcupine}):
        listener = MacOSContinuousListener(vad_model, backend=backend_mock)

    if wake_word_only is not None:
        listener._wake_word_only_mode = wake_word_only

    return listener, porcupine_mock, backend_mock


def _make_listener_without_porcupine(vad_speech_prob: float = 0.0):
    """Build a MacOSContinuousListener without Porcupine configured."""
    from lazy_claude.av_audio import MacOSContinuousListener

    backend_mock = MagicMock()
    backend_mock.install_mic_tap = MagicMock()
    backend_mock.play_audio = MagicMock()
    backend_mock.shutdown = MagicMock()

    vad_model = _make_vad_model(vad_speech_prob)

    # No PORCUPINE_ACCESS_KEY in environment
    with patch.dict('os.environ', {}, clear=False):
        import os
        orig_key = os.environ.pop('PORCUPINE_ACCESS_KEY', None)
        orig_path = os.environ.pop('PORCUPINE_MODEL_PATH', None)
        try:
            listener = MacOSContinuousListener(vad_model, backend=backend_mock)
        finally:
            if orig_key is not None:
                os.environ['PORCUPINE_ACCESS_KEY'] = orig_key
            if orig_path is not None:
                os.environ['PORCUPINE_MODEL_PATH'] = orig_path

    return listener, backend_mock


def _make_chunk(value: float = 0.0, size: int = 512) -> np.ndarray:
    return np.full(size, value, dtype=np.float32)


# ---------------------------------------------------------------------------
# Float32 → int16 conversion correctness
# ---------------------------------------------------------------------------

class TestFloat32ToInt16Conversion:
    def test_conversion_formula_correct(self):
        """(np.array([0.5]) * 32767).clip(-32768, 32767).astype(np.int16)[0] == 16383"""
        arr = np.array([0.5], dtype=np.float32)
        result = (arr * 32767).clip(-32768, 32767).astype(np.int16)[0]
        assert result == 16383, f"Expected 16383, got {result}"

    def test_conversion_not_zero_for_nonzero_input(self):
        """Verify we're not accidentally truncating to zero."""
        arr = np.array([0.5], dtype=np.float32)
        result = (arr * 32767).clip(-32768, 32767).astype(np.int16)[0]
        assert result != 0

    def test_conversion_positive_clamp(self):
        arr = np.array([2.0], dtype=np.float32)
        result = (arr * 32767).clip(-32768, 32767).astype(np.int16)[0]
        assert result == 32767

    def test_conversion_negative_clamp(self):
        arr = np.array([-2.0], dtype=np.float32)
        result = (arr * 32767).clip(-32768, 32767).astype(np.int16)[0]
        assert result == -32768


# ---------------------------------------------------------------------------
# Wake word → active transition
# ---------------------------------------------------------------------------

class TestWakeWordToActiveTransition:
    def test_keyword_detected_switches_mode_to_active(self):
        """When Porcupine returns >= 0, mode switches from wake_word to active."""
        porcupine = _make_mock_porcupine(keyword_return=0)
        listener, _, backend = _make_listener_with_porcupine(porcupine_mock=porcupine)

        # Ensure listener is in wake_word mode
        listener._active.set()
        assert listener._mode == "wake_word"

        chunk = _make_chunk(0.5)
        listener._process_chunk(chunk)

        assert listener._mode == "active"

    def test_keyword_sets_active_since(self):
        """_active_since is set when keyword is detected."""
        porcupine = _make_mock_porcupine(keyword_return=0)
        listener, _, backend = _make_listener_with_porcupine(porcupine_mock=porcupine)
        listener._active.set()

        assert listener._active_since is None
        chunk = _make_chunk(0.5)
        listener._process_chunk(chunk)

        assert listener._active_since is not None

    def test_no_keyword_stays_in_wake_word_mode(self):
        """When Porcupine returns -1 (no keyword), mode stays wake_word."""
        porcupine = _make_mock_porcupine(keyword_return=-1)
        listener, _, backend = _make_listener_with_porcupine(porcupine_mock=porcupine)
        listener._active.set()

        chunk = _make_chunk(0.5)
        listener._process_chunk(chunk)

        assert listener._mode == "wake_word"

    def test_ping_played_on_keyword_detect(self):
        """A ping audio is played through the backend when keyword is detected."""
        porcupine = _make_mock_porcupine(keyword_return=0)
        listener, _, backend = _make_listener_with_porcupine(porcupine_mock=porcupine)
        listener._active.set()

        chunk = _make_chunk(0.5)
        listener._process_chunk(chunk)

        # play_audio should have been called (ping)
        backend.play_audio.assert_called_once()
        played_audio = backend.play_audio.call_args[0][0]
        assert isinstance(played_audio, np.ndarray)
        assert len(played_audio) > 0


# ---------------------------------------------------------------------------
# Active → wake_word after utterance (wake_word_only=True)
# ---------------------------------------------------------------------------

class TestActiveToWakeWordAfterUtterance:
    def _simulate_full_utterance(self, listener):
        """Drive listener through a complete speech → silence → complete utterance cycle."""
        # VAD model returning speech for utterance chunks, then silence
        speech_chunk = _make_chunk(0.8)
        silence_chunk = _make_chunk(0.0)

        # Seed recording with speech
        listener._recording = True
        listener._accumulated_speech = 0.5  # meets min threshold
        listener._utterance_chunks = [speech_chunk]

        # Now inject silence for long enough
        now = time.monotonic()
        listener._silence_started = now - 2.0  # 2s ago — exceeds SILENCE_DURATION
        # Process a silence chunk that triggers completion
        listener._vad.return_value = 0.0
        listener._process_chunk(silence_chunk)

    def test_active_to_wake_word_after_utterance_when_wake_word_only(self):
        """With _wake_word_only_mode=True, after utterance stored, mode → wake_word."""
        porcupine = _make_mock_porcupine(keyword_return=-1)  # won't fire again
        listener, _, backend = _make_listener_with_porcupine(
            porcupine_mock=porcupine, wake_word_only=True
        )
        listener._active.set()
        listener._mode = "active"
        listener._active_since = time.monotonic()

        self._simulate_full_utterance(listener)

        assert listener._mode == "wake_word"

    def test_active_stays_active_after_utterance_when_not_wake_word_only(self):
        """With _wake_word_only_mode=False, after utterance stored, mode stays active."""
        porcupine = _make_mock_porcupine(keyword_return=-1)
        listener, _, backend = _make_listener_with_porcupine(
            porcupine_mock=porcupine, wake_word_only=False
        )
        listener._active.set()
        listener._mode = "active"
        listener._active_since = time.monotonic()

        self._simulate_full_utterance(listener)

        assert listener._mode == "active"


# ---------------------------------------------------------------------------
# 15-second timeout: active → wake_word with no speech
# ---------------------------------------------------------------------------

class TestActiveModeTimeout:
    def test_timeout_switches_to_wake_word(self):
        """After 15s with no speech (WAITING state), mode returns to wake_word."""
        porcupine = _make_mock_porcupine(keyword_return=-1)
        listener, _, backend = _make_listener_with_porcupine(
            porcupine_mock=porcupine, wake_word_only=True
        )
        listener._active.set()
        listener._mode = "active"
        # Set _active_since to 16s ago
        listener._active_since = time.monotonic() - 16.0
        # Ensure we're in WAITING state (not recording)
        listener._recording = False

        # Process a silence chunk — should detect timeout
        chunk = _make_chunk(0.0)
        listener._vad.return_value = 0.0
        listener._process_chunk(chunk)

        assert listener._mode == "wake_word"

    def test_no_timeout_before_15s(self):
        """Within 15s, active mode is preserved even with no speech."""
        porcupine = _make_mock_porcupine(keyword_return=-1)
        listener, _, backend = _make_listener_with_porcupine(
            porcupine_mock=porcupine, wake_word_only=True
        )
        listener._active.set()
        listener._mode = "active"
        listener._active_since = time.monotonic() - 5.0  # only 5s ago
        listener._recording = False

        chunk = _make_chunk(0.0)
        listener._vad.return_value = 0.0
        listener._process_chunk(chunk)

        assert listener._mode == "active"


# ---------------------------------------------------------------------------
# Ping VAD isolation: _tts_active gating
# ---------------------------------------------------------------------------

class TestPingVADIsolation:
    def test_tts_active_true_during_ping_false_after(self):
        """_tts_active is set True before playing ping and restored to False after."""
        porcupine = _make_mock_porcupine(keyword_return=0)
        listener, _, backend = _make_listener_with_porcupine(porcupine_mock=porcupine)
        listener._active.set()

        tts_states = []

        def capture_play_audio(audio_data):
            # Record _tts_active at the moment play_audio is called
            tts_states.append(listener._tts_active)

        backend.play_audio.side_effect = capture_play_audio

        chunk = _make_chunk(0.5)
        listener._process_chunk(chunk)

        # During play_audio, _tts_active should have been True
        assert len(tts_states) >= 1
        assert tts_states[0] is True

        # After _process_chunk returns, _tts_active should be False
        assert listener._tts_active is False


# ---------------------------------------------------------------------------
# set_active(False) resets mode
# ---------------------------------------------------------------------------

class TestSetActiveResetsMode:
    def test_set_active_false_with_porcupine_resets_to_wake_word(self):
        """set_active(False) with Porcupine present → _mode = 'wake_word'."""
        porcupine = _make_mock_porcupine(keyword_return=-1)
        listener, _, backend = _make_listener_with_porcupine(porcupine_mock=porcupine)
        listener._mode = "active"

        listener.set_active(False)

        assert listener._mode == "wake_word"

    def test_set_active_false_without_porcupine_mode_stays_active(self):
        """set_active(False) without Porcupine → _mode stays 'active' (no wake word)."""
        listener, backend = _make_listener_without_porcupine()
        listener._mode = "active"

        listener.set_active(False)

        # Without porcupine, mode should remain 'active' (no wake word mode)
        assert listener._mode == "active"


# ---------------------------------------------------------------------------
# stop() calls porcupine.delete()
# ---------------------------------------------------------------------------

class TestStopDeletesPorcupine:
    def test_stop_calls_porcupine_delete(self):
        """listener.stop() calls porcupine.delete() to release native resources."""
        porcupine = _make_mock_porcupine(keyword_return=-1)
        listener, _, backend = _make_listener_with_porcupine(porcupine_mock=porcupine)

        listener.stop()

        porcupine.delete.assert_called_once()

    def test_stop_without_porcupine_does_not_crash(self):
        """stop() when _porcupine is None must not raise."""
        listener, backend = _make_listener_without_porcupine()
        # Should not raise
        listener.stop()


# ---------------------------------------------------------------------------
# _wake_word_only_mode default logic
# ---------------------------------------------------------------------------

class TestWakeWordOnlyModeDefault:
    def test_wake_word_only_true_when_porcupine_available_and_not_always_on(self):
        """_wake_word_only_mode=True when Porcupine loaded and LAZY_CLAUDE_ALWAYS_ON unset."""
        listener, _, backend = _make_listener_with_porcupine()
        assert listener._wake_word_only_mode is True

    def test_wake_word_only_false_when_always_on_env_set(self):
        """_wake_word_only_mode=False when LAZY_CLAUDE_ALWAYS_ON=1."""
        from lazy_claude.av_audio import MacOSContinuousListener

        backend_mock = MagicMock()
        backend_mock.install_mic_tap = MagicMock()
        backend_mock.play_audio = MagicMock()
        vad_model = _make_vad_model()

        porcupine_mock = _make_mock_porcupine()
        mock_pvporcupine = MagicMock()
        mock_pvporcupine.create = MagicMock(return_value=porcupine_mock)

        with patch.dict('os.environ', {
            'PORCUPINE_ACCESS_KEY': 'test-key',
            'PORCUPINE_MODEL_PATH': '/fake/model.ppn',
            'LAZY_CLAUDE_ALWAYS_ON': '1',
        }), patch.dict('sys.modules', {'pvporcupine': mock_pvporcupine}):
            listener = MacOSContinuousListener(vad_model, backend=backend_mock)

        assert listener._wake_word_only_mode is False

    def test_wake_word_only_false_when_porcupine_not_available(self):
        """_wake_word_only_mode=False when Porcupine not configured."""
        listener, backend = _make_listener_without_porcupine()
        assert listener._wake_word_only_mode is False


# ---------------------------------------------------------------------------
# VoiceServer.set_listening_mode_impl
# ---------------------------------------------------------------------------

def _ensure_server_module_imported():
    """Ensure lazy_claude.server is importable by mocking sounddevice if needed."""
    import sys
    from unittest.mock import MagicMock

    # Ensure sounddevice is mocked before server import (only if not already there)
    if 'sounddevice' not in sys.modules:
        sys.modules['sounddevice'] = MagicMock()
    if 'lazy_claude.server' not in sys.modules:
        # Mock all heavy deps before importing the server module
        for mod in ['sounddevice', 'onnxruntime', 'pywhispercpp']:
            if mod not in sys.modules:
                sys.modules[mod] = MagicMock()
        try:
            import lazy_claude.server  # noqa: F401
        except Exception:
            pass


# Ensure the server module is importable at module load time
_ensure_server_module_imported()


def _make_server_with_listener(listener_mock):
    """Build a VoiceServer with a pre-wired listener mock (no audio hardware)."""
    from unittest.mock import patch, MagicMock

    mock_tts = MagicMock()
    mock_tts.is_speaking = False
    mock_tts.speak = MagicMock()
    mock_tts.stop = MagicMock()

    with patch('lazy_claude.server.TTSEngine', return_value=mock_tts), \
         patch('lazy_claude.server.load_model', return_value=MagicMock()), \
         patch('lazy_claude.server.load_vad_model', return_value=MagicMock()), \
         patch('lazy_claude.server.ReferenceBuffer'), \
         patch('lazy_claude.server.EchoCanceller'), \
         patch('lazy_claude.server.ContinuousListener', return_value=listener_mock):
        from lazy_claude.server import VoiceServer
        server = VoiceServer()
    server.tts = mock_tts
    server._listener = listener_mock
    return server


class TestSetListeningMode:
    def _make_listener_with_porcupine_mock(self):
        """Build a simple mock listener that has _porcupine set."""
        listener = MagicMock()
        listener._porcupine = MagicMock()  # Porcupine is available
        listener._wake_word_only_mode = False
        listener._mode = "active"
        listener._reset_utterance_state = MagicMock()
        return listener

    def _make_listener_without_porcupine_mock(self):
        """Build a simple mock listener with _porcupine=None."""
        listener = MagicMock()
        listener._porcupine = None
        listener._wake_word_only_mode = False
        listener._mode = "active"
        listener._reset_utterance_state = MagicMock()
        return listener

    def test_set_wake_word_mode_with_porcupine_available(self):
        """set_listening_mode('wake_word') with Porcupine → effective_mode='wake_word'."""
        listener = self._make_listener_with_porcupine_mock()
        server = _make_server_with_listener(listener)

        result = server.set_listening_mode_impl(mode="wake_word")

        assert result["mode"] == "wake_word"
        assert result["porcupine_available"] is True

    def test_set_wake_word_mode_without_porcupine_falls_back_to_always_on(self):
        """set_listening_mode('wake_word') without Porcupine → effective_mode='always_on'."""
        listener = self._make_listener_without_porcupine_mock()
        server = _make_server_with_listener(listener)

        result = server.set_listening_mode_impl(mode="wake_word")

        assert result["mode"] == "always_on"
        assert result["porcupine_available"] is False

    def test_set_always_on_mode_with_porcupine(self):
        """set_listening_mode('always_on') with Porcupine → effective_mode='always_on'."""
        listener = self._make_listener_with_porcupine_mock()
        server = _make_server_with_listener(listener)

        result = server.set_listening_mode_impl(mode="always_on")

        assert result["mode"] == "always_on"
        assert result["porcupine_available"] is True

    def test_set_wake_word_mode_updates_listener_mode(self):
        """Switching to wake_word sets listener._mode to 'wake_word'."""
        listener = self._make_listener_with_porcupine_mock()
        server = _make_server_with_listener(listener)

        server.set_listening_mode_impl(mode="wake_word")

        assert listener._mode == "wake_word"

    def test_set_wake_word_mode_calls_reset_utterance_state(self):
        """Switching to wake_word calls _reset_utterance_state()."""
        listener = self._make_listener_with_porcupine_mock()
        server = _make_server_with_listener(listener)

        server.set_listening_mode_impl(mode="wake_word")

        listener._reset_utterance_state.assert_called()

    def test_set_always_on_mode_updates_wake_word_only_flag(self):
        """Switching to always_on sets _wake_word_only_mode=False."""
        listener = self._make_listener_with_porcupine_mock()
        listener._wake_word_only_mode = True
        server = _make_server_with_listener(listener)

        server.set_listening_mode_impl(mode="always_on")

        assert listener._wake_word_only_mode is False


# ---------------------------------------------------------------------------
# set_listening_mode MCP tool registration
# ---------------------------------------------------------------------------

class TestSetListeningModeToolRegistration:
    def test_set_listening_mode_tool_registered(self):
        """set_listening_mode is registered as an MCP tool."""
        import asyncio
        from unittest.mock import patch, MagicMock

        mock_tts = MagicMock()
        mock_tts.is_speaking = False
        mock_listener = MagicMock()
        mock_listener._porcupine = None
        mock_listener._wake_word_only_mode = False
        mock_listener._mode = "active"
        mock_listener._reset_utterance_state = MagicMock()

        with patch('lazy_claude.server.TTSEngine', return_value=mock_tts), \
             patch('lazy_claude.server.load_model', return_value=MagicMock()), \
             patch('lazy_claude.server.load_vad_model', return_value=MagicMock()), \
             patch('lazy_claude.server.ReferenceBuffer'), \
             patch('lazy_claude.server.EchoCanceller'), \
             patch('lazy_claude.server.ContinuousListener', return_value=mock_listener):
            from lazy_claude.server import create_server
            app, voice = create_server()

        tool_names = [t.name for t in asyncio.run(app.list_tools())]
        assert "set_listening_mode" in tool_names
