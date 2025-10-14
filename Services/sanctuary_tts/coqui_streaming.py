"""Streaming TTS adapter built on top of Coqui TTS."""
from __future__ import annotations

import asyncio
from typing import AsyncIterator, Iterable, Optional

import numpy as np
from TTS.api import TTS

from Services.sanctuary_core.interfaces import TTSInterface


def _chunk_bytes(data: bytes, chunk_size: int) -> Iterable[bytes]:
    for idx in range(0, len(data), chunk_size):
        yield data[idx : idx + chunk_size]


class CoquiStreamingTTS(TTSInterface):
    """Generate PCM audio for the provided text and yield it in small frames."""

    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        *,
        speaker_wav: Optional[str] = None,
        language: str = "es",
        chunk_duration_ms: int = 200,
    ) -> None:
        self._tts = TTS(model_name)
        self._speaker_wav = speaker_wav
        self._language = language
        self.sample_rate = int(self._tts.synthesizer.output_sample_rate)
        self._chunk_samples = max(1, int(self.sample_rate * chunk_duration_ms / 1000))
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
        pcm = np.clip(audio, -1.0, 1.0)
        pcm_int16 = (pcm * 32767.0).astype(np.int16).tobytes()
        frame_bytes = self._chunk_samples * 2
        for chunk in _chunk_bytes(pcm_int16, frame_bytes):
            if self._stop_event.is_set():
                break
            yield chunk

    async def stop(self) -> None:
        self._stop_event.set()
