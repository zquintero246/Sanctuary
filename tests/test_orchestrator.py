import asyncio

import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Services.sanctuary_core.orchestrator import Orchestrator, SessionState
from Services.sanctuary_core.stubs import ScriptedLLM, ScriptedSTT, ScriptedTTS, ScriptedVAD


async def run_orchestrator(orchestrator, audio_chunks, ws_events):
    async def ws_send(payload, binary=False):
        ws_events.append(("binary" if binary else "text", payload))

    await orchestrator.handle_session(audio_chunks, ws_send)


async def audio_iter(chunks):
    for chunk in chunks:
        await asyncio.sleep(0)
        yield chunk


def test_stt_partials_before_final():
    async def runner():
        stt = ScriptedSTT(
            partials=[{"text": "hola", "is_final": False, "maybe_sentence_boundary": True}],
            final={"text": "hola", "is_final": True, "maybe_sentence_boundary": True},
        )
        llm = ScriptedLLM(["respuesta"])
        tts = ScriptedTTS({"respuesta": [b"audio"]})
        vad = ScriptedVAD([True, False], endpoint_after=2)
        orchestrator = Orchestrator(stt=stt, llm=llm, tts=tts, vad=vad)

        events = []
        await run_orchestrator(orchestrator, audio_iter([b"chunk1", b"chunk2"]), events)

        partial_index = next(
            i
            for i, (kind, payload) in enumerate(events)
            if kind == "text" and payload.get("type") == "stt_partial"
        )
        final_index = next(
            i
            for i, (kind, payload) in enumerate(events)
            if kind == "text" and payload.get("type") == "stt_final"
        )
        assert partial_index < final_index

    asyncio.run(runner())


class CoordinatedLLM(ScriptedLLM):
    def __init__(self):
        super().__init__(["uno", "dos", "tres"])
        self.first_chunk = asyncio.Event()
        self.continue_event = asyncio.Event()

    async def generate_stream(self, prompt: str):
        yield "uno"
        self.first_chunk.set()
        await self.continue_event.wait()
        yield "dos"
        yield "tres"


class InspectTTS(ScriptedTTS):
    def __init__(self):
        super().__init__({"uno": [b"a"], "dos": [b"b"], "tres": [b"c"]})
        self.calls = []
        self.first_call = asyncio.Event()

    async def stream(self, text: str):
        self.calls.append(text)
        if len(self.calls) == 1:
            self.first_call.set()
        async for chunk in super().stream(text):
            yield chunk


def test_tts_starts_on_first_llm_chunk():
    async def runner():
        stt = ScriptedSTT(
            partials=[{"text": "hola", "is_final": False, "maybe_sentence_boundary": True}],
            final={"text": "hola", "is_final": True, "maybe_sentence_boundary": True},
        )
        llm = CoordinatedLLM()
        tts = InspectTTS()
        vad = ScriptedVAD([True, False], endpoint_after=2)
        orchestrator = Orchestrator(stt=stt, llm=llm, tts=tts, vad=vad)

        events = []

        async def session():
            await run_orchestrator(orchestrator, audio_iter([b"chunk1", b"chunk2"]), events)

        task = asyncio.create_task(session())
        await asyncio.wait_for(llm.first_chunk.wait(), timeout=1.0)
        await asyncio.wait_for(tts.first_call.wait(), timeout=1.0)
        assert tts.calls[0] == "uno"
        llm.continue_event.set()
        await task

    asyncio.run(runner())


class BargeVAD(ScriptedVAD):
    def __init__(self):
        super().__init__([True, False, True, True, False], endpoint_after=5)
        self.calls = 0

    def is_voice(self, pcm_bytes: bytes) -> bool:
        self.calls += 1
        return super().is_voice(pcm_bytes)


class SlowLLM(ScriptedLLM):
    def __init__(self):
        super().__init__(["hola", " amigo", " ¿todo bien?"], delay=0.05)
        self.started = asyncio.Event()

    async def generate_stream(self, prompt: str):
        self.started.set()
        async for chunk in super().generate_stream(prompt):
            yield chunk


class ObservingTTS(ScriptedTTS):
    def __init__(self):
        super().__init__(
            {
                "hola": [b"a1", b"a2", b"a3"],
                " amigo": [b"b1", b"b2"],
                " ¿todo bien?": [b"c1", b"c2"],
            }
        )
        self.stop_event = asyncio.Event()
        self.started_event = asyncio.Event()

    async def stop(self) -> None:
        await super().stop()
        self.stop_event.set()

    async def stream(self, text: str):
        self.started_event.set()
        async for chunk in super().stream(text):
            await asyncio.sleep(0.05)
            yield chunk


def test_barge_in_triggers_stop():
    async def runner():
        stt = ScriptedSTT(
            partials=[{"text": "hola", "is_final": False, "maybe_sentence_boundary": True}],
            final={"text": "hola", "is_final": True, "maybe_sentence_boundary": True},
        )
        llm = SlowLLM()
        tts = ObservingTTS()
        vad = BargeVAD()
        orchestrator = Orchestrator(stt=stt, llm=llm, tts=tts, vad=vad)

        audio_queue: "asyncio.Queue[bytes | None]" = asyncio.Queue()

        async def queue_iter():
            while True:
                item = await audio_queue.get()
                audio_queue.task_done()
                if item is None:
                    break
                yield item

        events = []

        async def ws_send(payload, binary=False):
            events.append(("binary" if binary else "text", payload))

        task = asyncio.create_task(orchestrator.handle_session(queue_iter(), ws_send))
        await audio_queue.put(b"chunk1")  # initial speech
        await asyncio.sleep(0.01)
        await audio_queue.put(b"chunk2")  # silencio
        await asyncio.wait_for(llm.started.wait(), timeout=1.0)
        await asyncio.wait_for(tts.started_event.wait(), timeout=1.0)
        await audio_queue.put(b"chunk3")  # user barges in
        await audio_queue.put(b"chunk4")  # sustained speech
        for _ in range(20):
            if tts.stop_called:
                break
            await asyncio.sleep(0.05)
        assert tts.stop_called >= 1
        await audio_queue.put(None)
        await task
        assert orchestrator.state == SessionState.LISTENING

    asyncio.run(runner())


def test_metrics_emitted():
    async def runner():
        stt = ScriptedSTT(
            partials=[{"text": "hola", "is_final": False, "maybe_sentence_boundary": True}],
            final={"text": "hola", "is_final": True, "maybe_sentence_boundary": True},
        )
        llm = ScriptedLLM(["respuesta"])
        tts = ScriptedTTS({"respuesta": [b"audio"]})
        vad = ScriptedVAD([True, False], endpoint_after=2)
        orchestrator = Orchestrator(stt=stt, llm=llm, tts=tts, vad=vad)

        events = []
        await run_orchestrator(orchestrator, audio_iter([b"chunk1", b"chunk2"]), events)

        metrics = next(
            payload
            for kind, payload in events
            if kind == "text" and payload.get("type") == "metrics"
        )
        assert metrics["stt_first_partial_ms"] >= 0
        assert metrics["llm_first_token_ms"] >= 0
        assert metrics["tts_first_audio_ms"] >= 0
        assert metrics["turn_total_ms"] >= 0

    asyncio.run(runner())
