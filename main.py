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
from Services.sanctuary_core.llm_transformers import TransformersStreamingLLM
from Services.sanctuary_stt.whisper_streaming import WhisperStreamingSTT
from Services.sanctuary_tts.coqui_streaming import CoquiStreamingTTS

app = FastAPI()


def _build_services() -> tuple[STTInterface, LLMInterface, TTSInterface, VADInterface]:
    stt_model = os.getenv("SANCTUARY_STT_MODEL", "small")
    stt_language = os.getenv("SANCTUARY_STT_LANGUAGE", "es")
    stt = WhisperStreamingSTT(model_size=stt_model, language=stt_language)

    llm_model = os.getenv("SANCTUARY_LLM_MODEL", "distilgpt2")
    llm_prefix = os.getenv("SANCTUARY_LLM_SYSTEM_PREFIX", "")
    llm = TransformersStreamingLLM(model_name=llm_model, system_prefix=llm_prefix)

    tts_model = os.getenv("SANCTUARY_TTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
    tts_speaker = os.getenv("SANCTUARY_TTS_SPEAKER_WAV")
    tts_language = os.getenv("SANCTUARY_TTS_LANGUAGE", "es")
    tts = CoquiStreamingTTS(model_name=tts_model, speaker_wav=tts_speaker, language=tts_language)

    vad = EnergyVAD()
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
    stt, llm, tts, vad = _build_services()
    orchestrator = Orchestrator(stt=stt, llm=llm, tts=tts, vad=vad)

    async def ws_send(payload, binary: bool = False) -> None:
        if binary:
            await ws.send_bytes(payload)
        else:
            if isinstance(payload, str):
                await ws.send_text(payload)
            else:
                await ws.send_text(json.dumps(payload))

    await ws_send({"type": "tts_metadata", "sample_rate": tts.sample_rate})

    session_task = asyncio.create_task(
        orchestrator.handle_session(_queue_iterator(audio_queue), ws_send)
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
