"""Streaming STT adapter backed by the open-source Whisper models."""
from __future__ import annotations

import asyncio
import time
from asyncio import QueueEmpty
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, Optional

import numpy as np
import whisper

from Services.sanctuary_core.interfaces import STTInterface, STTPartial


def _pcm16_to_float32(pcm: bytes) -> np.ndarray:
    if not pcm:
        return np.array([], dtype=np.float32)
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    audio /= 32768.0
    return audio


class WhisperStreamingSTT(STTInterface):
    """Minimal streaming implementation for Whisper models.

    The class keeps a PCM buffer in memory and periodically spawns decoding jobs
    on a background :class:`ThreadPoolExecutor`.  Partial updates are emitted at
    a configurable cadence while the final result is produced once the caller
    invokes :meth:`get_final`.
    """

    def __init__(
        self,
        model_size: str = "small",
        *,
        language: Optional[str] = "es",
        partial_interval_ms: int = 150,
        endpoint_grace_ms: int = 350,
    ) -> None:
        self._model = whisper.load_model(model_size)
        self._language = language
        self._buffer = bytearray()
        self._partials: "asyncio.Queue[STTPartial]" = asyncio.Queue()
        self._final: Optional[STTPartial] = None
        self._lock = asyncio.Lock()
        self._partial_interval = partial_interval_ms / 1000.0
        self._last_partial_ts = 0.0
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")
        self._pending_partial_task: Optional[asyncio.Task] = None
        self._endpoint_grace = endpoint_grace_ms / 1000.0

    async def feed(self, pcm_bytes: bytes, sample_rate: int) -> None:
        del sample_rate  # Whisper handles resampling internally when needed.
        if not pcm_bytes:
            return
        async with self._lock:
            self._buffer.extend(pcm_bytes)
            now = time.perf_counter()
            if now - self._last_partial_ts >= self._partial_interval:
                self._last_partial_ts = now
                await self._schedule_partial(False)

    async def stream_partials(self) -> AsyncIterator[STTPartial]:
        while not self._partials.empty():
            yield await self._partials.get()
            self._partials.task_done()

    async def get_final(self) -> STTPartial:
        # Give a small grace period so the last chunk can be decoded if the
        # caller immediately calls ``get_final`` after the last audio frame.
        await asyncio.sleep(self._endpoint_grace)
        async with self._lock:
            if self._final is None:
                await self._schedule_partial(True)
            # ``_schedule_partial`` populates ``self._final`` for final calls.
            while self._final is None:
                await asyncio.sleep(0)
            result = self._final
            self._reset_state()
            return result

    async def _schedule_partial(self, is_final: bool) -> None:
        if self._pending_partial_task and not self._pending_partial_task.done():
            if is_final:
                # Wait for the running task to complete so we can emit a final.
                await self._pending_partial_task
            else:
                return

        pcm_snapshot = bytes(self._buffer)
        loop = asyncio.get_running_loop()
        self._pending_partial_task = asyncio.create_task(
            self._emit_partial(loop, pcm_snapshot, is_final)
        )
        if is_final:
            await self._pending_partial_task

    async def _emit_partial(
        self, loop: asyncio.AbstractEventLoop, pcm_snapshot: bytes, is_final: bool
    ) -> None:
        partial = await loop.run_in_executor(
            self._executor, self._decode_snapshot, pcm_snapshot, is_final
        )
        if is_final:
            self._final = partial
        else:
            await self._partials.put(partial)

    def _decode_snapshot(self, pcm_snapshot: bytes, is_final: bool) -> STTPartial:
        audio = _pcm16_to_float32(pcm_snapshot)
        if audio.size == 0:
            return {"text": "", "is_final": is_final, "maybe_sentence_boundary": False}
        result = self._model.transcribe(
            audio,
            language=self._language,
            fp16=False,
            word_timestamps=True,
        )
        text = result.get("text", "").strip()
        tokens = [
            {
                "t": segment.get("text", "").strip(),
                "t0": float(segment.get("start", 0.0)),
                "t1": float(segment.get("end", 0.0)),
            }
            for segment in result.get("segments", [])
        ]
        maybe_boundary = bool(text) and text[-1] in {".", "?", "!", "¡", "¿", "…"}
        return {
            "text": text,
            "tokens": tokens,
            "is_final": is_final,
            "maybe_sentence_boundary": maybe_boundary,
        }

    def _reset_state(self) -> None:
        self._buffer.clear()
        self._final = None
        self._last_partial_ts = 0.0
        while True:
            try:
                self._partials.get_nowait()
            except QueueEmpty:
                break
            else:
                self._partials.task_done()
        self._pending_partial_task = None

