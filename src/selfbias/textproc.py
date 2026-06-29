"""Helpers for separating reasoning ("thinking") from the final answer text.

Handles the three shapes seen in practice: an explicit `reasoning_content` field (some vLLM /
API backends), `<think>...</think>` inline tags (Qwen3/DeepSeek), and a dangling `</think>`
with no opening tag (OLMo, where vLLM strips the opening token).
"""

from __future__ import annotations

import re

_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def extract_thinking(text: str) -> str | None:
    m = _THINK_BLOCK.search(text)
    return m.group(1).strip() if m else None


def strip_thinking(text: str) -> str:
    return _THINK_BLOCK.sub("", text, count=0).strip()


def split_thinking(text: str, reasoning_content: str | None = None) -> tuple[str | None, str]:
    """Return (thinking, answer_text)."""
    if reasoning_content:
        return reasoning_content.strip(), text.strip()
    if "<think>" in text:
        return extract_thinking(text), strip_thinking(text)
    if "</think>" in text:
        head, _, tail = text.partition("</think>")
        return head.strip(), tail.strip()
    return None, text.strip()
