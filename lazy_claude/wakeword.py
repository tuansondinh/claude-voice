"""Wake-word backend helpers.

This module wraps the concrete detector so the rest of the codebase can keep
the simple ``process(pcm_int16) -> int`` contract that Porcupine used.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np


class OpenWakeWordDetector:
    """Thin adapter around openWakeWord with a Porcupine-like API."""

    _FRAME_SAMPLES = 1_280  # 80 ms at 16 kHz

    def __init__(
        self,
        model_paths: Optional[list[str]] = None,
        threshold: float = 0.75,
        vad_threshold: Optional[float] = None,
    ) -> None:
        import openwakeword  # type: ignore[import-untyped]
        from openwakeword.model import Model  # type: ignore[import-untyped]
        from openwakeword.utils import download_models  # type: ignore[import-untyped]

        kwargs: dict[str, Any] = {}
        if not model_paths:
            default_paths = openwakeword.get_pretrained_model_paths("onnx")
            if default_paths and not all(os.path.exists(path) for path in default_paths):
                download_models()
        if model_paths:
            kwargs["wakeword_models"] = model_paths
        if vad_threshold is not None:
            kwargs["vad_threshold"] = vad_threshold
        kwargs["inference_framework"] = "onnx"

        self._model = Model(**kwargs)
        self._threshold = threshold
        self._buffer = np.array([], dtype=np.int16)

    def process(self, pcm_int16: np.ndarray) -> int:
        """Return 0 on wake-word detection, else -1."""
        frame = np.asarray(pcm_int16, dtype=np.int16).reshape(-1)
        if frame.size == 0:
            return -1

        self._buffer = np.concatenate((self._buffer, frame))
        detected = -1

        while self._buffer.size >= self._FRAME_SAMPLES:
            window = self._buffer[: self._FRAME_SAMPLES].copy()
            self._buffer = self._buffer[self._FRAME_SAMPLES :]
            prediction = self._model.predict(window)
            if self._prediction_above_threshold(prediction):
                detected = 0
                break

        return detected

    def delete(self) -> None:
        """Match the old detector cleanup API."""
        self._buffer = np.array([], dtype=np.int16)

    def _prediction_above_threshold(self, prediction: Any) -> bool:
        if isinstance(prediction, dict):
            return any(float(score) >= self._threshold for score in prediction.values())
        try:
            return float(prediction) >= self._threshold
        except Exception:
            return False


def create_wakeword_detector() -> Optional[OpenWakeWordDetector]:
    """Build and return the configured wake-word detector, if enabled."""
    enabled = os.environ.get("LAZY_CLAUDE_WAKEWORD", "0")
    if enabled != "1" and "OPENWAKEWORD_MODEL_PATH" not in os.environ:
        return None

    model_path = os.environ.get("OPENWAKEWORD_MODEL_PATH")
    if model_path:
        model_paths = [model_path]
    else:
        model_paths = ["hey_jarvis"]
    threshold = float(os.environ.get("OPENWAKEWORD_THRESHOLD", "0.75"))

    vad_threshold_raw = os.environ.get("OPENWAKEWORD_VAD_THRESHOLD")
    vad_threshold = float(vad_threshold_raw) if vad_threshold_raw else 0.5

    return OpenWakeWordDetector(
        model_paths=model_paths,
        threshold=threshold,
        vad_threshold=vad_threshold,
    )
