"""Voice session orchestrator coordinating STT, LLM and TTS services."""
from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from enum import Enum
from typing import AsyncIterator, Awaitable, Callable, Deque, Optional

from .interfaces import (
    LLMInterface,
    STTInterface,
    STTPartial,
    TTSInterface,
    VADInterface,
)
from .tracer import Tracer


class SessionState(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    INTERRUPTED = "INTERRUPTED"

class Orchestrator:
    """Coordinates the realtime full-duplex voice pipeline."""

    def __init__(
        self,
        stt: STTInterface,
        llm: LLMInterface,
        tts: TTSInterface,
        vad: VADInterface,
        sample_rate: int = 16000,
    ) -> None:
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.vad = vad
        self.sample_rate = sample_rate
        self.state: SessionState = SessionState.LISTENING
        self._speak_q: "asyncio.Queue[Optional[str]]" = asyncio.Queue()
        self._stop_speaking = asyncio.Event()
        self._llm_task: Optional[asyncio.Task] = None
        self._pending_prompts: Deque[str] = deque()
        self._session_done = asyncio.Event()
        self._stt_first_partial_emitted = False
        self._metrics_sent = asyncio.Event()

    async def handle_session(
        self,
        audio_chunks: AsyncIterator[bytes],
        ws_send: Callable[[object, bool], Awaitable[None]],
    ) -> None:
        """Drive a full conversational turn until the audio source completes."""

        tracer = Tracer()
        tracer.mark("turn_start")

        listen_task = asyncio.create_task(self._listen_loop(audio_chunks, ws_send, tracer))
        speak_task = asyncio.create_task(self._speak_loop(ws_send, tracer))

        await listen_task
        # Ensure all pending speech has been processed before stopping the speaker loop.
        await self._speak_q.join()
        await self._speak_q.put(None)
        await speak_task

        tracer.mark("turn_end")
        metrics = tracer.metrics()
        if metrics:
            await ws_send({"type": "metrics", **metrics}, False)
        tracer.dump()

    # ------------------------------------------------------------------
    async def _listen_loop(
        self,
        audio_chunks: AsyncIterator[bytes],
        ws_send: Callable[[object, bool], Awaitable[None]],
        tracer: Tracer,
    ) -> None:
        try:
            async for pcm in audio_chunks:
                user_is_speaking = self.vad.is_voice(pcm)

                if self.state == SessionState.SPEAKING and user_is_speaking:
                    await self._interrupt_speaking()
                    self.state = SessionState.INTERRUPTED

                if user_is_speaking:
                    self.state = SessionState.LISTENING
                    await self.stt.feed(pcm, self.sample_rate)
                    async for partial in self.stt.stream_partials():
                        await self._emit_partial(partial, ws_send, tracer)
                        if partial.get("maybe_sentence_boundary"):
                            await self._maybe_start_llm(partial["text"], ws_send, tracer)
                else:
                    # still feed STT to keep buffers aligned
                    await self.stt.feed(pcm, self.sample_rate)

                if self.vad.endpointed():
                    final = await self.stt.get_final()
                    if not tracer._mark_time("stt_final"):
                        tracer.mark("stt_final")
                    await ws_send(
                        {
                            "type": "stt_final",
                            "text": final.get("text", ""),
                            "is_final": True,
                        },
                        False,
                    )
                    await self._maybe_start_llm(final.get("text", ""), ws_send, tracer)
                    self.vad.reset()
        finally:
            # If listening loop exits ensure any ongoing LLM task completes or is cancelled.
            if self._llm_task and not self._llm_task.done():
                self._llm_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._llm_task

    async def _emit_partial(
        self,
        partial: STTPartial,
        ws_send: Callable[[object, bool], Awaitable[None]],
        tracer: Tracer,
    ) -> None:
        if not self._stt_first_partial_emitted:
            tracer.mark("stt_first_partial")
            self._stt_first_partial_emitted = True
        await ws_send(
            {
                "type": "stt_partial",
                "text": partial.get("text", ""),
                "is_final": bool(partial.get("is_final", False)),
            },
            False,
        )

    async def _maybe_start_llm(
        self,
        text: str,
        ws_send: Callable[[object, bool], Awaitable[None]],
        tracer: Tracer,
    ) -> None:
        if not text.strip():
            return
        if self.state in {SessionState.THINKING, SessionState.SPEAKING}:
            # Already in progress, queue for later.
            self._pending_prompts.append(text)
            return

        async def run(prompt: str) -> None:
            self.state = SessionState.THINKING
            first_chunk = True
            try:
                async for chunk in self.llm.generate_stream(prompt):
                    if self._stop_speaking.is_set():
                        break
                    if first_chunk:
                        tracer.mark("llm_first_token")
                        self.state = SessionState.SPEAKING
                        first_chunk = False
                    await ws_send({"type": "assistant_text", "text": chunk}, False)
                    await self._speak_q.put(chunk)
            finally:
                if self._stop_speaking.is_set():
                    self._stop_speaking.clear()
                if self._pending_prompts:
                    next_prompt = self._pending_prompts.popleft()
                    await self._maybe_start_llm(next_prompt, ws_send, tracer)
                else:
                    # Completed speaking, go back to listening.
                    self.state = SessionState.LISTENING

        self._llm_task = asyncio.create_task(run(text))

    async def _speak_loop(
        self,
        ws_send: Callable[[object, bool], Awaitable[None]],
        tracer: Tracer,
    ) -> None:
        first_audio_emitted = False
        while True:
            text = await self._speak_q.get()
            if text is None:
                self._speak_q.task_done()
                break
            try:
                async for audio_chunk in self.tts.stream(text):
                    if self._stop_speaking.is_set():
                        break
                    if not first_audio_emitted:
                        tracer.mark("tts_first_audio")
                        first_audio_emitted = True
                    await ws_send(audio_chunk, True)
            finally:
                self._speak_q.task_done()

    async def _interrupt_speaking(self) -> None:
        if not self._stop_speaking.is_set():
            self._stop_speaking.set()
        await self.tts.stop()
        while True:
            try:
                item = self._speak_q.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                if item is not None:
                    # Pending text entries are discarded to avoid stale playback.
                    pass
                self._speak_q.task_done()
