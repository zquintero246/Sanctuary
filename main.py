"""Application entry point exposing the `/voice` websocket endpoint."""
from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator, Callable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from Services.sanctuary_core.orchestrator import Orchestrator
from Services.sanctuary_core.stubs import ScriptedLLM, ScriptedSTT, ScriptedTTS
from Services.sanctuary_core.vad import EnergyVAD

app = FastAPI()


async def _queue_iterator(queue: "asyncio.Queue[bytes]") -> AsyncIterator[bytes]:
    while True:
        item = await queue.get()
        queue.task_done()
        if item is None:
            break
        yield item


@app.websocket("/voice")
async def voice_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    audio_queue: "asyncio.Queue[bytes | None]" = asyncio.Queue()

    # Placeholder scripted components; replace with concrete services.
    stt = ScriptedSTT(
        partials=[{"text": "hola", "is_final": False, "maybe_sentence_boundary": True}],
        final={"text": "hola", "is_final": True, "maybe_sentence_boundary": True},
    )
    llm = ScriptedLLM(["¡Hola!", " ¿En qué puedo ayudarte?"])
    tts = ScriptedTTS({
        "¡Hola!": [b"audio1"],
        " ¿En qué puedo ayudarte?": [b"audio2"],
    })
    vad = EnergyVAD()
    orchestrator = Orchestrator(stt=stt, llm=llm, tts=tts, vad=vad)

    async def ws_send(payload, binary: bool = False) -> None:
        if binary:
            await ws.send_bytes(payload)
        else:
            await ws.send_text(json.dumps(payload))

    session_task = asyncio.create_task(orchestrator.handle_session(_queue_iterator(audio_queue), ws_send))

    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                await audio_queue.put(message["bytes"])
            elif message.get("text") is not None:
                data = json.loads(message["text"])
                if data.get("type") == "end_user_turn":
                    break
    except WebSocketDisconnect:
        pass
    finally:
        await audio_queue.put(None)
        await session_task
