"""Backend protocol and a shared OpenAI-compatible chat client.

A `Backend` turns a batch of chat conversations into `Generation`s. Offline vLLM, online vLLM
(OpenAI-compatible server), and OpenRouter all satisfy the same protocol, so generation and
evaluation code is backend-agnostic.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from selfbias.textproc import split_thinking

# A conversation is a list of {"role": ..., "content": ...} messages.
Conversation = list[dict[str, str]]


@dataclass
class GenParams:
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int | None = None
    max_tokens: int = 32768

    @classmethod
    def from_sampling(cls, sampling: dict[str, Any]) -> "GenParams":
        return cls(
            temperature=sampling.get("temperature", 0.6),
            top_p=sampling.get("top_p", 0.95),
            top_k=sampling.get("top_k"),
            max_tokens=sampling.get("max_tokens", 32768),
        )


@dataclass
class Generation:
    text: str  # final answer text (thinking removed)
    thinking: str | None = None
    raw: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Backend(Protocol):
    def generate(self, conversations: list[Conversation], params: GenParams) -> list[Generation]: ...


class OpenAIChatBackend:
    """Shared client for any OpenAI-compatible chat endpoint (vLLM server or OpenRouter).

    Requests run concurrently with a thread pool (order preserved). Unsupported-parameter
    quirks of reasoning models are handled with a small fallback ladder.
    """

    def __init__(self, spec, base_url: str, api_key: str, *, extra_headers: dict | None = None,
                 max_workers: int = 8, max_retries: int = 4):
        from openai import OpenAI

        self.spec = spec
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.extra_headers = extra_headers or {}
        self.max_workers = max_workers
        self.max_retries = max_retries

    def _create(self, messages: Conversation, params: GenParams):
        # Fallback ladder: full params -> drop sampling -> use max_completion_tokens.
        attempts = [
            dict(temperature=params.temperature, top_p=params.top_p, max_tokens=params.max_tokens),
            dict(max_tokens=params.max_tokens),
            dict(max_completion_tokens=params.max_tokens),
        ]
        last_err: Exception | None = None
        for kwargs in attempts:
            try:
                return self.client.chat.completions.create(
                    model=self.spec.id, messages=messages,
                    extra_headers=self.extra_headers, **kwargs,
                )
            except Exception as e:  # noqa: BLE001 - provider error shapes vary
                last_err = e
        raise last_err  # type: ignore[misc]

    def _one(self, messages: Conversation, params: GenParams) -> Generation:
        for attempt in range(self.max_retries):
            try:
                resp = self._create(messages, params)
                msg = resp.choices[0].message
                text = msg.content or ""
                thinking = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
                t, answer = split_thinking(text, thinking)
                return Generation(text=answer, thinking=t, raw=text)
            except Exception as e:  # noqa: BLE001
                if attempt == self.max_retries - 1:
                    return Generation(text="", thinking=None, raw="", meta={"error": str(e)})
                time.sleep(2 ** attempt)
        return Generation(text="", thinking=None, raw="")  # unreachable

    def generate(self, conversations: list[Conversation], params: GenParams) -> list[Generation]:
        if self.max_workers <= 1:
            return [self._one(c, params) for c in conversations]
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            return list(ex.map(lambda c: self._one(c, params), conversations))
