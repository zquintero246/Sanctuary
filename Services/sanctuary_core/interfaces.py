"""Core interface definitions for the Sanctuary realtime voice pipeline."""
from __future__ import annotations

from typing import AsyncIterator, List, Optional, TypedDict


class TokenTiming(TypedDict, total=False):
    """Timing metadata for a recognized token."""

    t: str
    t0: float
    t1: float


class STTPartial(TypedDict, total=False):
    """Representation of a streaming STT update."""

    text: str
    tokens: List[TokenTiming]
    is_final: bool
    maybe_sentence_boundary: bool


class STTInterface:
    """Speech-to-text streaming interface."""

    async def feed(self, pcm_bytes: bytes, sample_rate: int) -> None:
        """Feed PCM audio into the recogniser."""

    async def stream_partials(self) -> AsyncIterator[STTPartial]:
        """Yield partial transcription updates as they become available."""

    async def get_final(self) -> STTPartial:
        """Return the final transcription segment once endpointed."""


class LLMInterface:
    """Large language model streaming interface."""

    async def generate_stream(self, prompt: str) -> AsyncIterator[str]:
        """Yield chunks of generated text for the supplied prompt."""


class TTSInterface:
    """Text-to-speech streaming interface."""

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        """Yield PCM audio chunks synthesised from *text*."""

    async def stop(self) -> None:
        """Immediately stop any in-flight synthesis (for barge-in)."""


class VADInterface:
    """Voice activity detector abstraction."""

    def is_voice(self, pcm_bytes: bytes) -> bool:
        """Return ``True`` if the chunk contains speech."""

    def endpointed(self) -> bool:
        """Return ``True`` once a segment endpoint has been detected."""

    def reset(self) -> None:
        """Reset any internal endpointing state so a new turn can start."""
