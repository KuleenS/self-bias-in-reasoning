"""Unified reasoning example schema and dataset-loader registry.

Every dataset loader maps its native columns into `ReasoningExample` so the rest of the
pipeline (generation, answer-matching, evaluation) is dataset-agnostic. `answer_type` drives
both the generation instruction and the answer extractor in `selfbias.answers`.
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field
from typing import Any, Callable

from selfbias.config import get_dataset_config

ANSWER_TYPES = ("numeric", "mcq", "freeform")


@dataclass(frozen=True)
class ReasoningExample:
    id: str  # stable position-based id within the (dataset, split); aligns prompt_id across models
    dataset: str
    question: str  # full problem text shown to the model (includes options for mcq)
    gold_answer: str  # canonical answer: a letter for mcq, a number for numeric, else a string
    answer_type: str  # one of ANSWER_TYPES
    choices: dict[str, str] | None = None  # {"A": "...", ...} for mcq
    meta: dict[str, Any] = field(default_factory=dict)


# loader(cfg, limit, seed) -> list[ReasoningExample]
LoaderFn = Callable[[dict[str, Any], "int | None", int], list[ReasoningExample]]
_LOADERS: dict[str, LoaderFn] = {}


def register_loader(name: str) -> Callable[[LoaderFn], LoaderFn]:
    def deco(fn: LoaderFn) -> LoaderFn:
        _LOADERS[name] = fn
        return fn

    return deco


def get_loader(name: str) -> LoaderFn:
    if name not in _LOADERS:
        # Trigger registration of all loaders lazily (keeps `import selfbias` cheap).
        import selfbias.data.loaders  # noqa: F401
        import selfbias.data.folio  # noqa: F401
    if name not in _LOADERS:
        raise KeyError(f"no loader registered for dataset '{name}'; have {sorted(_LOADERS)}")
    return _LOADERS[name]


def list_datasets() -> list[str]:
    import selfbias.data.loaders  # noqa: F401
    import selfbias.data.folio  # noqa: F401

    return sorted(_LOADERS)


def load_examples(name: str, limit: int | None = None, seed: int = 0) -> list[ReasoningExample]:
    cfg = get_dataset_config(name)
    loader = get_loader(name)
    return loader(cfg, limit, seed)


# --- helpers shared by loaders -------------------------------------------------

def format_mcq(stem: str, options: list[str]) -> tuple[str, dict[str, str]]:
    """Return (question_with_options, {letter: option_text})."""
    letters = string.ascii_uppercase
    choices = {letters[i]: str(opt) for i, opt in enumerate(options)}
    rendered = "\n".join(f"{ltr}. {txt}" for ltr, txt in choices.items())
    return f"{stem.strip()}\n\nOptions:\n{rendered}", choices


def pick(row: dict[str, Any], *names: str, default: Any = None) -> Any:
    """First present (and non-None) value among candidate column names."""
    for n in names:
        if n in row and row[n] is not None:
            return row[n]
    return default
