"""Testing aids and scripted implementations for the orchestrator."""
from __future__ import annotations

import asyncio
from collections import deque
from typing import AsyncIterator, Deque, Iterable, List, Optional

from .interfaces import LLMInterface, STTInterface, STTPartial, TTSInterface, VADInterface


class ScriptedSTT(STTInterface):
    """STT implementation that yields scripted partials/finals."""

    def __init__(self, partials: Iterable[STTPartial], final: STTPartial) -> None:
        self._partials: Deque[STTPartial] = deque(partials)
        self._final = final
        self.feed_count = 0

    async def feed(self, pcm_bytes: bytes, sample_rate: int) -> None:  # pragma: no cover - trivial
        self.feed_count += 1

    async def stream_partials(self) -> AsyncIterator[STTPartial]:
        while self._partials:
            yield self._partials.popleft()

    async def get_final(self) -> STTPartial:
        return self._final


class ScriptedLLM(LLMInterface):
    """LLM implementation yielding pre-defined chunks."""

    def __init__(self, chunks: Iterable[str], delay: float = 0.0) -> None:
        self._chunks = list(chunks)
        self.delay = delay

    async def generate_stream(self, prompt: str) -> AsyncIterator[str]:
        for chunk in self._chunks:
            if self.delay:
                await asyncio.sleep(self.delay)
            yield chunk


class ScriptedTTS(TTSInterface):
    """TTS implementation yielding scripted audio payloads."""

    def __init__(self, chunk_map: Optional[dict[str, List[bytes]]] = None) -> None:
        self._chunk_map = chunk_map or {}
        self._stop_event = asyncio.Event()
        self.stop_called = 0

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        self._stop_event.clear()
        for chunk in self._chunk_map.get(text, [b"audio"]):
            if self._stop_event.is_set():
                break
            yield chunk

    async def stop(self) -> None:
        self.stop_called += 1
        self._stop_event.set()


class ScriptedVAD(VADInterface):
    """VAD implementation driven by a boolean script per chunk."""

    def __init__(self, script: Iterable[bool], endpoint_after: int = 0) -> None:
        self._script = deque(script)
        self._endpoint_after = endpoint_after
        self._chunks_seen = 0

    def is_voice(self, pcm_bytes: bytes) -> bool:
        self._chunks_seen += 1
        if self._script:
            return self._script.popleft()
        return False

    def endpointed(self) -> bool:
        return self._chunks_seen >= self._endpoint_after if self._endpoint_after else False
