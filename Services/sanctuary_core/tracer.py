"""Minimal tracer utilities for latency instrumentation."""
from __future__ import annotations

import contextlib
import json
import time
import uuid
from typing import Dict, Iterable, Optional


class Tracer:
    """Collects timestamped events and emits JSON telemetry."""

    def __init__(self, session_id: Optional[str] = None) -> None:
        self.sid = session_id or str(uuid.uuid4())
        self.events: list[dict] = []

    def mark(self, name: str, meta: Optional[Dict] = None) -> None:
        """Record a timestamped event."""

        self.events.append(
            {
                "t": time.perf_counter(),
                "type": "mark",
                "name": name,
                "meta": meta or {},
            }
        )

    @contextlib.contextmanager
    def span(self, name: str, meta: Optional[Dict] = None):
        """Context manager to record span start/end events."""

        t0 = time.perf_counter()
        self.events.append(
            {
                "t": t0,
                "type": "start",
                "name": name,
                "meta": meta or {},
            }
        )
        try:
            yield
        finally:
            t1 = time.perf_counter()
            self.events.append({"t": t1, "type": "end", "name": name})

    def dump(self) -> None:
        """Print the collected events as newline-delimited JSON."""

        base = self.events[0]["t"] if self.events else time.perf_counter()
        out = []
        for event in self.events:
            out.append(
                {
                    "session_id": self.sid,
                    "type": event["type"],
                    "name": event["name"],
                    "t_ms": int((event["t"] - base) * 1000),
                    "meta": event.get("meta", {}),
                }
            )
        print(json.dumps(out, ensure_ascii=False))

    # --- Metrics helpers -------------------------------------------------
    def _mark_time(self, name: str) -> Optional[float]:
        for event in self.events:
            if event["type"] == "mark" and event["name"] == name:
                return event["t"]
        return None

    def metrics(self) -> Dict[str, int]:
        """Compute latency metrics for the voice turn."""

        def diff(start: str, end: str) -> Optional[int]:
            t0 = self._mark_time(start)
            t1 = self._mark_time(end)
            if t0 is None or t1 is None:
                return None
            return int((t1 - t0) * 1000)

        start_name = "turn_start"
        end_name = "turn_end"
        metrics = {
            "stt_first_partial_ms": diff(start_name, "stt_first_partial"),
            "stt_final_ms": diff(start_name, "stt_final"),
            "llm_first_token_ms": diff(start_name, "llm_first_token"),
            "tts_first_audio_ms": diff(start_name, "tts_first_audio"),
            "turn_total_ms": diff(start_name, end_name),
        }
        return {k: v for k, v in metrics.items() if v is not None}
