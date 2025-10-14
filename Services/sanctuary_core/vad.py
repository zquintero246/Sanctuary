"""Simple energy based voice activity detector."""
from __future__ import annotations

import collections
import math
from typing import Deque

from .interfaces import VADInterface


class EnergyVAD(VADInterface):
    """A lightweight energy-based VAD suitable for unit testing."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        silence_ms: int = 300,
        voice_threshold: float = 40.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.silence_frames = max(1, int(silence_ms / frame_ms))
        self.voice_threshold = voice_threshold
        self._recent_energy: Deque[float] = collections.deque(maxlen=self.silence_frames)
        self._silence_run = 0
        self._endpoint = False

    def is_voice(self, pcm_bytes: bytes) -> bool:
        if not pcm_bytes:
            return False
        rms = self._rms(pcm_bytes)
        self._recent_energy.append(rms)
        if rms > self.voice_threshold:
            self._silence_run = 0
            self._endpoint = False
            return True
        self._silence_run += 1
        if self._silence_run >= self.silence_frames:
            self._endpoint = True
        return False

    def endpointed(self) -> bool:
        if self._endpoint:
            self._endpoint = False
            return True
        return False

    def reset(self) -> None:
        self._recent_energy.clear()
        self._silence_run = 0
        self._endpoint = False

    @staticmethod
    def _rms(pcm_bytes: bytes) -> float:
        if not pcm_bytes:
            return 0.0
        count = len(pcm_bytes) // 2
        if count == 0:
            return 0.0
        ints = memoryview(pcm_bytes).cast("h")
        acc = sum(sample * sample for sample in ints)
        mean = acc / count
        return math.sqrt(mean)
