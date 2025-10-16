"""Streaming XTTS adapter built on top of Coqui TTS."""
from __future__ import annotations

import asyncio
import importlib
from typing import AsyncIterator, Iterable, Optional

import numpy as np

_transformers = importlib.import_module("transformers")
if not hasattr(_transformers, "BeamSearchScorer"):
    # Recent versions of ``transformers`` stopped re-exporting ``BeamSearchScorer``
    # from the top-level package, while Coqui TTS still imports it from there.
    # Mirror the old attribute so the dependency keeps working.
    _beam_search = importlib.import_module("transformers.generation.beam_search")
    _transformers.BeamSearchScorer = _beam_search.BeamSearchScorer

from TTS.api import TTS

from Services.sanctuary_core.interfaces import TTSInterface


def _chunk_bytes(data: bytes, chunk_size: int) -> Iterable[bytes]:
    for idx in range(0, len(data), chunk_size):
        yield data[idx : idx + chunk_size]


def _apply_fade_out(pcm: np.ndarray, fade_samples: int) -> np.ndarray:
    if fade_samples <= 0 or pcm.size == 0:
        return pcm
    fade_samples = min(fade_samples, pcm.size)
    fade_curve = np.linspace(1.0, 0.0, fade_samples, endpoint=True, dtype=np.float32)
    pcm_tail = pcm[-fade_samples:]
    pcm[-fade_samples:] = pcm_tail * fade_curve
    return pcm


class XTTSStreamingTTS(TTSInterface):
    """Generate PCM16 audio chunks from text with barge-in friendly fade out."""

    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        *,
        speaker_wav: Optional[str] = None,
        language: str = "es",
        sample_rate: int = 16000,
        jitter_ms: int = 150,
        fade_out_ms: int = 60,
    ) -> None:
        self._tts = TTS(model_name)
        self._speaker_wav = speaker_wav
        self._language = language
        self._native_rate = int(self._tts.synthesizer.output_sample_rate)
        self.sample_rate = sample_rate or self._native_rate
        self._frame_samples = max(1, int(self.sample_rate * jitter_ms / 1000))
        self._fade_samples = int(self.sample_rate * fade_out_ms / 1000)
        self._stop_event = asyncio.Event()

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        self._stop_event.clear()
        loop = asyncio.get_running_loop()

        def _synth() -> np.ndarray:
            return self._tts.tts(
                text=text,
                speaker_wav=self._speaker_wav,
                language=self._language,
            )

        wav = await loop.run_in_executor(None, _synth)
        audio = np.asarray(wav, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if audio.size == 0:
            return
        pcm = np.clip(audio, -1.0, 1.0)
        if self.sample_rate != self._native_rate:
            pcm = self._resample(pcm, self._native_rate, self.sample_rate)
        pcm_int16 = (pcm * 32767.0).astype(np.int16)

        frame_bytes = self._frame_samples * 2
        yielded_final = False
        for chunk in _chunk_bytes(pcm_int16.tobytes(), frame_bytes):
            if self._stop_event.is_set():
                tail = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
                faded = _apply_fade_out(tail, self._fade_samples)
                if faded.size:
                    yield faded.astype(np.int16).tobytes()
                yielded_final = True
                break
            yield chunk

        if self._stop_event.is_set() and not yielded_final:
            # Provide a short silence tail to smooth the cutoff.
            silence = np.zeros(min(self._frame_samples, self._fade_samples), dtype=np.int16)
            yield silence.tobytes()

    async def stop(self) -> None:
        self._stop_event.set()

    @staticmethod
    def _resample(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        if src_rate == dst_rate or pcm.size == 0:
            return pcm
        duration = pcm.size / src_rate
        dst_length = int(duration * dst_rate)
        if dst_length == 0:
            return pcm
        src_times = np.linspace(0.0, duration, num=pcm.size, endpoint=False, dtype=np.float32)
        dst_times = np.linspace(0.0, duration, num=dst_length, endpoint=False, dtype=np.float32)
        return np.interp(dst_times, src_times, pcm).astype(np.float32)
