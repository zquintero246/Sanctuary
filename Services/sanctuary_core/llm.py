"""Generic streaming LLM adapter."""
from __future__ import annotations

from typing import AsyncIterator, Awaitable, Callable, Iterable, Optional

from .interfaces import LLMInterface


GeneratorFn = Callable[[str, Optional[int], Optional[list[str]]], AsyncIterator[str]]


class StreamingLLM(LLMInterface):
    """Adapter that exposes a streaming ``generate_stream`` API."""

    def __init__(
        self,
        generator: GeneratorFn,
        *,
        system_prefix: str = "",
        default_max_tokens: int = 256,
    ) -> None:
        self._generator = generator
        self._prefix = system_prefix
        self._default_max_tokens = default_max_tokens

    async def generate_stream(
        self,
        prompt: str,
        *,
        stop_sequences: Optional[list[str]] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        full_prompt = f"{self._prefix}{prompt}" if self._prefix else prompt
        max_toks = max_tokens or self._default_max_tokens
        async for chunk in self._generator(full_prompt, max_toks, stop_sequences or []):
            yield chunk
