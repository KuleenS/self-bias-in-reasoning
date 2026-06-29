"""Generate reasoning chains: one model over one dataset.

Output: data/chains/{dataset}/{model_slug}.jsonl, one record per prompt, including the answer
extracted from the chain and whether it matches gold (the answer-matching ground truth). Runs
are resumable — already-written indices are skipped.
"""

from __future__ import annotations

from pathlib import Path

from selfbias.answers import grade
from selfbias.config import chains_path
from selfbias.data.base import load_examples
from selfbias.inference.base import Backend, GenParams
from selfbias.inference.factory import build_backend
from selfbias.models.registry import ModelSpec, get_model
from selfbias.prompts import generation_messages
from selfbias.utils import append_jsonl, load_done_indices, slugify


def _chain_text(text: str, thinking: str | None) -> str:
    if thinking:
        return f"<think>\n{thinking}\n</think>\n{text}"
    return text


def run_generation(
    model: ModelSpec | str,
    dataset: str,
    n_prompts: int | None = None,
    seed: int = 0,
    mode: str | None = None,
    out_path: Path | None = None,
    backend: Backend | None = None,
    resume: bool = True,
) -> Path:
    spec = get_model(model) if isinstance(model, str) else model
    out_path = out_path or chains_path(dataset, slugify(spec.id))

    examples = load_examples(dataset, n_prompts, seed)
    done = load_done_indices(out_path) if resume else set()
    todo = [(i, ex) for i, ex in enumerate(examples) if i not in done]
    if not todo:
        print(f"[generate] {spec.short} x {dataset}: nothing to do ({len(done)} done)")
        return out_path

    backend = backend or build_backend(spec, mode)
    params = GenParams.from_sampling(spec.sampling)
    conversations = [generation_messages(ex) for _, ex in todo]
    generations = backend.generate(conversations, params)

    records = []
    n_correct = 0
    for (i, ex), gen in zip(todo, generations):
        graded_on = gen.text or gen.raw
        extracted, correct = grade(graded_on, ex.gold_answer, ex.answer_type)
        n_correct += int(correct)
        records.append({
            "dataset": dataset,
            "model": spec.name,
            "model_short": spec.short,
            "index": i,
            "example_id": ex.id,
            "question": ex.question,
            "answer_type": ex.answer_type,
            "gold_answer": ex.gold_answer,
            "generated_text": _chain_text(gen.text, gen.thinking),
            "thinking": gen.thinking,
            "extracted_answer": extracted,
            "is_correct": correct,
            "error": gen.meta.get("error"),
        })

    append_jsonl(out_path, records)
    print(f"[generate] {spec.short} x {dataset}: wrote {len(records)} "
          f"(acc {n_correct}/{len(records)}) -> {out_path}")
    return out_path
