"""Generic streaming TTS helper with jitter buffering."""
from __future__ import annotations

import asyncio
from typing import AsyncIterator, Awaitable, Callable, Optional

from Services.sanctuary_core.interfaces import TTSInterface


Synthesiser = Callable[[str], Awaitable[AsyncIterator[bytes]]]


class StreamingTTS(TTSInterface):
    """A simple streaming TTS wrapper with a jitter buffer."""

    def __init__(
        self,
        synthesiser: Synthesiser,
        *,
        jitter_ms: int = 150,
    ) -> None:
        self._synthesiser = synthesiser
        self._buffer_ms = jitter_ms
        self._stop_event = asyncio.Event()
        self._lock = asyncio.Lock()

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        async with self._lock:
            self._stop_event.clear()
            iterator = await self._synthesiser(text)
            buffer: list[bytes] = []
            budget_ms = self._buffer_ms
            async for chunk in iterator:
                if self._stop_event.is_set():
                    break
                buffer.append(chunk)
                # emulate jitter buffer drain once filled
                if budget_ms <= 0:
                    break
                budget_ms -= self._estimate_chunk_ms(chunk)
            for chunk in buffer:
                if self._stop_event.is_set():
                    break
                yield chunk
            if not self._stop_event.is_set():
                async for chunk in iterator:
                    if self._stop_event.is_set():
                        break
                    yield chunk

    async def stop(self) -> None:
        self._stop_event.set()

    @staticmethod
    def _estimate_chunk_ms(chunk: bytes, sample_rate: int = 24000) -> float:
        if not chunk:
            return 0.0
        frame_count = len(chunk) // 2
        return frame_count / sample_rate * 1000
