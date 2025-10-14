"""Generic streaming STT wrapper with partial aggregation support."""
from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator, Awaitable, Callable, Optional

from Services.sanctuary_core.interfaces import STTInterface, STTPartial


PartialDecoder = Callable[[bytes, int], Awaitable[STTPartial]]
FinalDecoder = Callable[[bytes, int], Awaitable[STTPartial]]


class StreamingSTT(STTInterface):
    """A backend agnostic streaming STT helper.

    The class coordinates buffering of PCM audio, periodically invokes a
    *partial decoder* to obtain incremental transcripts and exposes them
    through :meth:`stream_partials`.
    """

    def __init__(
        self,
        partial_decoder: PartialDecoder,
        final_decoder: Optional[FinalDecoder] = None,
        *,
        partial_interval_ms: int = 150,
        endpoint_silence_ms: int = 300,
    ) -> None:
        self._partial_decoder = partial_decoder
        self._final_decoder = final_decoder or partial_decoder
        self._partial_interval = partial_interval_ms / 1000.0
        self._endpoint_silence = endpoint_silence_ms / 1000.0
        self._buffer = bytearray()
        self._partials: "asyncio.Queue[STTPartial]" = asyncio.Queue()
        self._final: Optional[STTPartial] = None
        self._last_partial_time = 0.0
        self._last_voice_time = time.perf_counter()
        self._lock = asyncio.Lock()

    async def feed(self, pcm_bytes: bytes, sample_rate: int) -> None:
        async with self._lock:
            self._buffer.extend(pcm_bytes)
            now = time.perf_counter()
            if now - self._last_partial_time >= self._partial_interval:
                partial = await self._partial_decoder(bytes(self._buffer), sample_rate)
                partial.setdefault("is_final", False)
                partial.setdefault("maybe_sentence_boundary", False)
                await self._partials.put(partial)
                self._last_partial_time = now
            self._last_voice_time = now

    async def stream_partials(self) -> AsyncIterator[STTPartial]:
        while not self._partials.empty():
            yield await self._partials.get()
            self._partials.task_done()

    async def get_final(self) -> STTPartial:
        async with self._lock:
            if self._final is None:
                self._final = await self._final_decoder(bytes(self._buffer), 0)
                self._final.setdefault("is_final", True)
                self._final.setdefault("maybe_sentence_boundary", True)
            return self._final
