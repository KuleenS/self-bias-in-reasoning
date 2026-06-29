"""Model registry: typed access to the model pool defined in configs/models.yaml.

This is also where per-family sampling defaults live (migrated from the old
`get_sampling_params`), so the recommended decoding settings are defined once and overridable
per model in YAML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from selfbias.config import models_config
from selfbias.utils import slugify

# Recommended decoding per model family (overridable per model in models.yaml).
_FAMILY_SAMPLING: dict[str, dict[str, Any]] = {
    "qwen": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 32768},
    "deepseek": {"temperature": 0.6, "top_p": 0.95, "max_tokens": 32768},
    "olmo": {"temperature": 0.6, "top_p": 0.95, "max_tokens": 32768},
    "phi": {"temperature": 0.8, "top_p": 0.95, "max_tokens": 32768},
    "gemma": {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "max_tokens": 32768},
}
_DEFAULT_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "max_tokens": 32768}

_FAMILY_KEYWORDS = ["qwen", "deepseek", "olmo", "gemma", "phi", "openai", "gpt",
                    "anthropic", "claude", "google", "gemini", "grok", "xai", "llama"]


def default_sampling(family: str) -> dict[str, Any]:
    return dict(_FAMILY_SAMPLING.get(family, _DEFAULT_SAMPLING))


def _infer_family(name: str) -> str:
    low = name.lower()
    for kw in _FAMILY_KEYWORDS:
        if kw in low:
            return {"gpt": "openai", "claude": "anthropic", "gemini": "google",
                    "grok": "xai"}.get(kw, kw)
    return "unknown"


@dataclass(frozen=True)
class ModelSpec:
    name: str  # canonical registry key
    id: str  # id passed to the backend (HF path or OpenRouter slug)
    backend: str  # vllm_offline | vllm_online | openrouter
    family: str
    short: str  # short label for filenames/plots, e.g. "qwen", "ds"
    reasoning: bool = True
    thinking_template: bool = False  # pass enable_thinking=True to the chat template (Qwen3/Gemma)
    tensor_parallel: int = 1
    max_model_len: int = 32768
    sampling: dict[str, Any] = field(default_factory=dict)


def load_registry() -> dict[str, ModelSpec]:
    raw = models_config()
    specs: dict[str, ModelSpec] = {}
    for name, d in raw.items():
        d = d or {}
        family = d.get("family") or _infer_family(name)
        sampling = {**default_sampling(family), **(d.get("sampling") or {})}
        specs[name] = ModelSpec(
            name=name,
            id=d.get("id", name),
            backend=d.get("backend", "openrouter"),
            family=family,
            short=d.get("short", slugify(name).lower()),
            reasoning=d.get("reasoning", True),
            thinking_template=d.get("thinking_template", False),
            tensor_parallel=d.get("tensor_parallel", 1),
            max_model_len=d.get("max_model_len", 32768),
            sampling=sampling,
        )
    return specs


def get_model(name: str) -> ModelSpec:
    reg = load_registry()
    if name in reg:
        return reg[name]
    # allow lookup by short label
    by_short = {s.short: s for s in reg.values()}
    if name in by_short:
        return by_short[name]
    raise KeyError(f"model '{name}' not in models.yaml; available: {sorted(reg)}")


def list_models(backend: str | None = None, reasoning_only: bool = False) -> list[ModelSpec]:
    specs = list(load_registry().values())
    if backend:
        specs = [s for s in specs if s.backend == backend]
    if reasoning_only:
        specs = [s for s in specs if s.reasoning]
    return specs


def resolve_models(names: list[str]) -> list[ModelSpec]:
    """Map a list of names or short labels to ModelSpecs, preserving order."""
    return [get_model(n) for n in names]
