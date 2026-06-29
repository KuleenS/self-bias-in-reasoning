"""Construct a Backend from a ModelSpec (optionally overriding the backend mode)."""

from __future__ import annotations

from selfbias.inference.base import Backend


def build_backend(spec, mode: str | None = None, **kwargs) -> Backend:
    """`mode` overrides spec.backend (e.g. force a vLLM model through OpenRouter for a smoke test)."""
    backend = mode or spec.backend
    if backend == "vllm_offline":
        from selfbias.inference.vllm_offline import VLLMOfflineBackend

        return VLLMOfflineBackend(spec)
    if backend == "vllm_online":
        from selfbias.inference.vllm_online import VLLMOnlineBackend

        return VLLMOnlineBackend(spec, **kwargs)
    if backend == "openrouter":
        from selfbias.inference.openrouter import OpenRouterBackend

        return OpenRouterBackend(spec, **kwargs)
    raise ValueError(f"unknown backend '{backend}'")
