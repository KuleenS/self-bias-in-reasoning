"""Online vLLM backend — talk to a running OpenAI-compatible vLLM server.

Start a server with e.g.
    vllm serve Qwen/Qwen3-32B --port 8000 --tensor-parallel-size 2
then point this backend at it via VLLM_BASE_URL (default http://localhost:8000/v1).
"""

from __future__ import annotations

import os

from selfbias.inference.base import OpenAIChatBackend


class VLLMOnlineBackend(OpenAIChatBackend):
    def __init__(self, spec, base_url: str | None = None, max_workers: int = 16):
        from dotenv import load_dotenv

        load_dotenv()
        base_url = base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        api_key = os.getenv("VLLM_API_KEY", "EMPTY")
        super().__init__(spec, base_url, api_key, max_workers=max_workers)
