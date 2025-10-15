"""FastAPI entrypoint exposing the `/voice` websocket endpoint."""
from __future__ import annotations

import asyncio
import json
import os
from typing import AsyncIterator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from Services.sanctuary_core.interfaces import LLMInterface, STTInterface, TTSInterface, VADInterface
from Services.sanctuary_core.orchestrator import Orchestrator
from Services.sanctuary_core.vad import EnergyVAD
from Services.sanctuary_core.llm_transformers import StreamingLLM
from Services.sanctuary_stt.whisper_streaming import WhisperStreamingSTT
from Services.sanctuary_tts.xtts_tts import XTTSStreamingTTS

app = FastAPI()


def _build_services(sample_rate: int, frame_ms: int) -> tuple[STTInterface, LLMInterface, TTSInterface, VADInterface]:
    stt_model = os.getenv("SANCTUARY_STT_MODEL", "small")
    stt_language = os.getenv("SANCTUARY_STT_LANGUAGE", "es")
    partial_interval = int(os.getenv("SANCTUARY_STT_PARTIAL_MS", "150"))
    stt = WhisperStreamingSTT(
        model_size=stt_model,
        language=stt_language,
        sample_rate=sample_rate,
        partial_interval_ms=partial_interval,
    )

    llm_model = os.getenv("SANCTUARY_LLM_MODEL", "distilgpt2")
    llm_prefix = os.getenv("SANCTUARY_LLM_SYSTEM_PREFIX", "")
    max_tokens = int(os.getenv("SANCTUARY_LLM_MAX_TOKENS", "120"))
    stop_sequences = tuple(
        filter(None, os.getenv("SANCTUARY_LLM_STOP", "\n\n").split("|"))
    )
    llm = StreamingLLM(
        model_name=llm_model,
        system_prefix=llm_prefix,
        max_new_tokens=max_tokens,
        stop_sequences=list(stop_sequences) if stop_sequences else None,
    )

    tts_model = os.getenv("SANCTUARY_TTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
    tts_speaker = os.getenv("SANCTUARY_TTS_SPEAKER_WAV")
    tts_language = os.getenv("SANCTUARY_TTS_LANGUAGE", "es")
    jitter_ms = int(os.getenv("SANCTUARY_TTS_JITTER_MS", "150"))
    tts = XTTSStreamingTTS(
        model_name=tts_model,
        speaker_wav=tts_speaker,
        language=tts_language,
        sample_rate=sample_rate,
        jitter_ms=jitter_ms,
    )

    end_silence = int(os.getenv("SANCTUARY_VAD_END_SILENCE_MS", "300"))
    vad = EnergyVAD(sample_rate=sample_rate, frame_ms=frame_ms, silence_ms=end_silence)
    return stt, llm, tts, vad


async def _queue_iterator(queue: "asyncio.Queue[bytes | None]") -> AsyncIterator[bytes]:
    while True:
        chunk = await queue.get()
        queue.task_done()
        if chunk is None:
            break
        yield chunk


@app.websocket("/voice")
async def voice_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    audio_queue: "asyncio.Queue[bytes | None]" = asyncio.Queue()

    sample_rate = int(os.getenv("SANCTUARY_SR", "16000"))
    frame_ms = int(os.getenv("SANCTUARY_FRAME_MS", "20"))
    stt, llm, tts, vad = _build_services(sample_rate, frame_ms)
    orchestrator = Orchestrator(
        stt=stt,
        llm=llm,
        tts=tts,
        vad=vad,
        sample_rate=sample_rate,
    )

    async def ws_send(payload) -> None:
        if isinstance(payload, str):
            await ws.send_text(payload)
        else:
            await ws.send_text(json.dumps(payload))

    async def ws_send_bytes(pcm: bytes) -> None:
        await ws.send_bytes(pcm)

    await ws_send({"type": "tts_metadata", "sample_rate": tts.sample_rate})

    session_task = asyncio.create_task(
        orchestrator.handle_session(_queue_iterator(audio_queue), ws_send, ws_send_bytes)
    )

    try:
        while True:
            message = await ws.receive()
            msg_type = message.get("type")
            if msg_type == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                await audio_queue.put(message["bytes"])
            elif message.get("text") is not None:
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue
                if data.get("type") == "end_user_turn":
                    break
    except WebSocketDisconnect:
        pass
    finally:
        await audio_queue.put(None)
        await session_task
