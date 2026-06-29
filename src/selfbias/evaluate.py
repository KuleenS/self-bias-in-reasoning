"""Cross-model evaluation: an evaluator judges a generator's reasoning chains.

This is the core measurement. For each chain produced by `generator` on `dataset`, `evaluator`
outputs {"valid": bool}. The diagonal (evaluator == generator) gives the "self" cells. Ground
truth (`generator_correct`) is carried over from generation but never shown to the evaluator.

Output: data/judgments/{dataset}/{evaluator_slug}__on__{generator_slug}.jsonl (resumable).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from selfbias.config import chains_path, judgments_path
from selfbias.inference.base import Backend, GenParams
from selfbias.inference.factory import build_backend
from selfbias.models.registry import ModelSpec, get_model
from selfbias.prompts import evaluation_messages
from selfbias.utils import append_jsonl, load_done_indices, read_jsonl, slugify

_JSON_OBJ_RE = re.compile(r"\{.*?\}", re.DOTALL)
_VALID_RE = re.compile(r'"?valid"?\s*[:=]\s*(true|false)', re.IGNORECASE)


def parse_judgment(text: str) -> tuple[bool | None, bool]:
    """Return (judgment, parse_error). Tries JSON first, then a regex fallback."""
    if not text:
        return None, True
    m = _JSON_OBJ_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group())
            v = obj.get("valid")
            if isinstance(v, bool):
                return v, False
        except json.JSONDecodeError:
            pass
    m = _VALID_RE.search(text)
    if m:
        return m.group(1).lower() == "true", False
    return None, True


def _eval_params(evaluator: ModelSpec, eval_max_tokens: int) -> GenParams:
    if evaluator.reasoning:
        params = GenParams.from_sampling(evaluator.sampling)
        params.max_tokens = eval_max_tokens
        return params
    return GenParams(temperature=0.0, top_p=1.0, max_tokens=512)


def run_evaluation(
    evaluator: ModelSpec | str,
    generator: ModelSpec | str,
    dataset: str,
    mode: str | None = None,
    backend: Backend | None = None,
    chains_input: Path | None = None,
    out_path: Path | None = None,
    max_reasoning_chars: int = 20000,
    eval_max_tokens: int = 8192,
    resume: bool = True,
) -> Path:
    ev = get_model(evaluator) if isinstance(evaluator, str) else evaluator
    gen = get_model(generator) if isinstance(generator, str) else generator

    chains_input = chains_input or chains_path(dataset, slugify(gen.id))
    out_path = out_path or judgments_path(dataset, slugify(ev.id), slugify(gen.id))
    if not Path(chains_input).exists():
        raise FileNotFoundError(f"missing chains for {gen.short} x {dataset}: {chains_input} "
                                "(run generation first)")

    chains = list(read_jsonl(chains_input))
    done = load_done_indices(out_path) if resume else set()
    todo = [c for c in chains if c["index"] not in done]
    if not todo:
        print(f"[evaluate] {ev.short} on {gen.short} x {dataset}: nothing to do")
        return out_path

    backend = backend or build_backend(ev, mode)
    params = _eval_params(ev, eval_max_tokens)
    conversations = [
        evaluation_messages(c["question"], c["generated_text"], max_reasoning_chars) for c in todo
    ]
    outputs = backend.generate(conversations, params)

    is_self = ev.name == gen.name
    records, parse_errors = [], 0
    for chain, out in zip(todo, outputs):
        judgment, parse_error = parse_judgment(out.text or out.raw)
        parse_errors += int(parse_error)
        records.append({
            "dataset": dataset,
            "generator": gen.name,
            "evaluator": ev.name,
            "generator_short": gen.short,
            "evaluator_short": ev.short,
            "index": chain["index"],
            "is_self": int(is_self),
            "generator_correct": chain.get("is_correct"),
            "evaluator_judgment": judgment,
            "parse_error": parse_error,
            "evaluator_thinking": out.thinking,
            "raw_output_text": out.text,
        })

    append_jsonl(out_path, records)
    print(f"[evaluate] {ev.short} on {gen.short} x {dataset}: wrote {len(records)} "
          f"(parse errors {parse_errors}) -> {out_path}")
    return out_path
