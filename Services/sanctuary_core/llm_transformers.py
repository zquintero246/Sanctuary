"""Streaming LLM adapter built on top of HuggingFace Transformers."""
from __future__ import annotations

import asyncio
import threading
from typing import AsyncIterator, Dict, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    TextIteratorStreamer,
)

from .interfaces import LLMInterface


class TransformersStreamingLLM(LLMInterface):
    """Generate tokens incrementally using a local transformer model."""

    def __init__(
        self,
        model_name: str = "distilgpt2",
        *,
        device: Optional[str] = None,
        max_new_tokens: int = 200,
        stop_sequences: Optional[list[str]] = None,
        generation_kwargs: Optional[Dict] = None,
        system_prefix: str = "",
    ) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(model_name)
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._max_new_tokens = max_new_tokens
        self._stop_sequences = stop_sequences or []
        self._gen_kwargs = generation_kwargs or {
            "temperature": 0.7,
            "top_p": 0.95,
            "do_sample": True,
        }
        self._system_prefix = system_prefix

    async def generate_stream(self, prompt: str) -> AsyncIterator[str]:
        full_prompt = f"{self._system_prefix}{prompt}" if self._system_prefix else prompt
        tokenized = self._tokenizer(full_prompt, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in tokenized.items()}
        streamer = TextIteratorStreamer(
            self._tokenizer, skip_special_tokens=True, skip_prompt=True
        )

        generation_kwargs = dict(self._gen_kwargs)
        generation_kwargs.update(
            {
                "streamer": streamer,
                "max_new_tokens": self._max_new_tokens,
                "pad_token_id": self._tokenizer.eos_token_id,
            }
        )
        if self._stop_sequences:
            stop_ids = self._tokenizer(self._stop_sequences, add_special_tokens=False)[
                "input_ids"
            ]
            generation_kwargs["stopping_criteria"] = _StopSequencesCriteria(stop_ids)

        loop = asyncio.get_running_loop()
        thread = threading.Thread(
            target=self._model.generate,
            kwargs={**inputs, **generation_kwargs},
            daemon=True,
        )
        thread.start()

        try:
            while True:
                token = await loop.run_in_executor(None, streamer.text_queue.get)
                if token is None:
                    break
                yield token
        finally:
            thread.join(timeout=0.1)


class StreamingLLM(TransformersStreamingLLM):
    """Alias kept for compatibility with the public API."""


class _StopSequencesCriteria(StoppingCriteria):
    """Simple stopping criteria matching token sequences."""

    def __init__(self, stop_sequences: list[list[int]]) -> None:
        self.stop_sequences = [torch.tensor(seq, dtype=torch.long) for seq in stop_sequences]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        for seq in self.stop_sequences:
            if seq.numel() == 0:
                continue
            if torch.equal(input_ids[0, -seq.numel() :], seq.to(input_ids.device)):
                return True
        return False
