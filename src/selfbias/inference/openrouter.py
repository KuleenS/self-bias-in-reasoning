"""OpenRouter backend — unified access to open + frontier models via one API key."""

from __future__ import annotations

import os

from selfbias.inference.base import OpenAIChatBackend

_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterBackend(OpenAIChatBackend):
    def __init__(self, spec, max_workers: int = 8):
        from dotenv import load_dotenv

        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set (see .env.example).")
        super().__init__(
            spec,
            _BASE_URL,
            api_key,
            extra_headers={
                "HTTP-Referer": "https://github.com/KuleenS/self-bias-in-reasoning",
                "X-Title": "selfbias",
            },
            max_workers=max_workers,
        )
