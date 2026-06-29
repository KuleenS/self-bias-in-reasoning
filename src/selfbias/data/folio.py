"""FOLIO loader (local JSONL under data/FOLIO/).

FOLIO is a 3-way formal-logic entailment task. We present it as an MCQ over
{True, False, Uncertain} so the answer extractor and correctness check are uniform with the
other datasets.
"""

from __future__ import annotations

from typing import Any

from selfbias.config import data_dir
from selfbias.data.base import ReasoningExample, format_mcq, pick, register_loader

_LABELS = ["True", "False", "Uncertain"]
_LABEL_NORM = {
    "true": "True", "entailed": "True", "entailment": "True", "yes": "True",
    "false": "False", "contradicted": "False", "contradiction": "False", "no": "False",
    "uncertain": "Uncertain", "unknown": "Uncertain", "neutral": "Uncertain",
}


def _split_file(split: str) -> str:
    s = (split or "validation").lower()
    if s in ("val", "valid", "validation", "dev"):
        return "folio_validation.jsonl"
    if s == "test":
        return "folio_test.jsonl"
    return "folio_train.jsonl"


@register_loader("folio")
def load_folio(cfg: dict[str, Any], limit: int | None, seed: int) -> list[ReasoningExample]:
    import json

    path = data_dir() / "FOLIO" / _split_file(cfg.get("split", "validation"))
    examples: list[ReasoningExample] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            row = json.loads(line)
            premises = pick(row, "premises", default="")
            conclusion = pick(row, "conclusion", default="")
            label_raw = str(pick(row, "label", default="")).strip()
            gold_label = _LABEL_NORM.get(label_raw.lower(), label_raw or "Uncertain")
            stem = (
                f"Premises:\n{premises}\n\nConclusion:\n{conclusion}\n\n"
                "Based only on the premises, is the conclusion True, False, or Uncertain?"
            )
            question, choices = format_mcq(stem, _LABELS)
            # gold letter = index of the label in _LABELS
            gold_letter = "ABC"[_LABELS.index(gold_label)] if gold_label in _LABELS else "C"
            examples.append(
                ReasoningExample(
                    id=str(i),
                    dataset="folio",
                    question=question,
                    gold_answer=gold_letter,
                    answer_type="mcq",
                    choices=choices,
                    meta={"label": gold_label, "story_id": row.get("story_id")},
                )
            )
    return examples
