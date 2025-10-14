"""Command line client that captures microphone audio and streams it to `/voice`."""
from __future__ import annotations

import argparse
import asyncio
import json
import signal
from typing import Any, Optional
import sounddevice as sd
import websockets


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sanctuary realtime voice client")
    parser.add_argument("--uri", default="ws://localhost:8000/voice", help="WebSocket URI")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Capture sample rate")
    parser.add_argument(
        "--frame-ms",
        type=int,
        default=20,
        help="Capture frame size in milliseconds",
    )
    parser.add_argument(
        "--print-events",
        action="store_true",
        help="Print JSON events received from the server",
    )
    return parser


async def _playback_loop(queue: "asyncio.Queue[Any]") -> None:
    stream: Optional[sd.RawOutputStream] = None
    current_rate = 16000
    loop = asyncio.get_running_loop()

    while True:
        item = await queue.get()
        if item is None:
            break
        if isinstance(item, tuple) and item[0] == "rate":
            current_rate = int(item[1])
            if stream is not None:
                stream.stop()
                stream.close()
                stream = None
            continue
        audio_bytes = item
        if stream is None:
            stream = sd.RawOutputStream(
                samplerate=current_rate,
                channels=1,
                dtype="int16",
                blocksize=0,
            )
            stream.start()
        await loop.run_in_executor(None, stream.write, audio_bytes)

    if stream is not None:
        stream.stop()
        stream.close()


async def run_client(args: argparse.Namespace) -> None:
    sample_rate = args.sample_rate
    frame_samples = int(sample_rate * args.frame_ms / 1000)
    capture_queue: "asyncio.Queue[bytes]" = asyncio.Queue()
    playback_queue: "asyncio.Queue[Any]" = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _capture_callback(indata, frames, time_info, status):  # pragma: no cover - callback
        if status:
            print("[capture]", status)
        frame = bytes(indata)
        loop.call_soon_threadsafe(capture_queue.put_nowait, frame)

    input_stream = sd.RawInputStream(
        samplerate=sample_rate,
        blocksize=frame_samples,
        channels=1,
        dtype="int16",
        callback=_capture_callback,
    )
    input_stream.start()

    async with websockets.connect(args.uri, ping_interval=None) as ws:
        async def sender() -> None:
            try:
                while True:
                    chunk = await capture_queue.get()
                    if chunk is None:
                        break
                    await ws.send(chunk)
            finally:
                stop_event.set()

        async def receiver() -> None:
            try:
                async for message in ws:
                    if isinstance(message, bytes):
                        await playback_queue.put(message)
                        continue
                    event = json.loads(message)
                    if args.print_events:
                        print("[event]", event)
                    if event.get("type") == "tts_metadata":
                        await playback_queue.put(("rate", event.get("sample_rate", sample_rate)))
                    elif event.get("type") == "metrics":
                        print("[metrics]", event)
            finally:
                stop_event.set()

        async def playback() -> None:
            try:
                await _playback_loop(playback_queue)
            finally:
                stop_event.set()

        async def shutdown() -> None:
            input_stream.stop()
            input_stream.close()
            try:
                await ws.send(json.dumps({"type": "end_user_turn"}))
            except Exception:  # pragma: no cover - connection teardown
                pass
            await capture_queue.put(None)
            await playback_queue.put(None)

        stop_event = asyncio.Event()

        def _handle_sigint(*_):
            stop_event.set()

        try:
            loop.add_signal_handler(signal.SIGINT, _handle_sigint)
        except NotImplementedError:  # pragma: no cover - Windows fallback
            pass

        tasks = [
            asyncio.create_task(sender()),
            asyncio.create_task(receiver()),
            asyncio.create_task(playback()),
        ]

        try:
            await stop_event.wait()
        finally:
            await shutdown()
            await asyncio.gather(*tasks, return_exceptions=True)


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    asyncio.run(run_client(args))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
