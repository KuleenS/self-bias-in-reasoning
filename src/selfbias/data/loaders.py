"""Loaders for the diverse top-10 reasoning datasets.

Each loader maps a dataset's native columns into `ReasoningExample`s. HF ids / subsets / splits
come from `configs/datasets.yaml` (the single edit point), so swapping a source only touches YAML.
`datasets` is imported lazily inside loaders to keep `import selfbias` cheap.
"""

from __future__ import annotations

import random
from typing import Any, Iterable

from selfbias.data.base import ReasoningExample, format_mcq, pick, register_loader


def _safe_load(hf_id: str, subset: str | None, split: str):
    from datasets import load_dataset

    args = (hf_id,) if subset is None else (hf_id, subset)
    try:
        return load_dataset(*args, split=split)
    except Exception as e:
        if "Dataset scripts are no longer supported" in str(e):
            # Recent `datasets` hard-blocks legacy loading-script repos; HF auto-converts them to
            # Parquet on this ref regardless.
            return load_dataset(*args, split=split, revision="refs/convert/parquet")
        # Older datasets ship a loading script.
        return load_dataset(*args, split=split, trust_remote_code=True)


def _iter_limited(ds: Iterable[dict[str, Any]], limit: int | None) -> Iterable[tuple[int, dict]]:
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        yield i, row


def _split(cfg: dict[str, Any], default: str) -> str:
    return cfg.get("split", default)


# --- math (numeric / freeform) -------------------------------------------------

@register_loader("gsm8k")
def load_gsm8k(cfg, limit, seed):
    ds = _safe_load(cfg["hf_id"], cfg.get("subset", "main"), _split(cfg, "test"))
    out = []
    for i, row in _iter_limited(ds, limit):
        gold = str(row["answer"]).split("####")[-1].strip().replace(",", "")
        out.append(ReasoningExample(str(i), "gsm8k", row["question"], gold, "numeric"))
    return out


@register_loader("math500")
def load_math500(cfg, limit, seed):
    ds = _safe_load(cfg["hf_id"], cfg.get("subset"), _split(cfg, "test"))
    out = []
    for i, row in _iter_limited(ds, limit):
        q = pick(row, "problem", "question")
        gold = str(pick(row, "answer", "solution"))
        out.append(ReasoningExample(str(i), "math500", q, gold, "freeform",
                                    meta={"subject": row.get("subject"), "level": row.get("level")}))
    return out


@register_loader("aime")
def load_aime(cfg, limit, seed):
    ds = _safe_load(cfg["hf_id"], cfg.get("subset"), _split(cfg, "train"))
    out = []
    for i, row in _iter_limited(ds, limit):
        q = pick(row, "Problem", "problem", "question")
        gold = str(pick(row, "Answer", "answer")).strip()
        out.append(ReasoningExample(str(i), "aime", q, gold, "numeric"))
    return out


# --- science / knowledge (mcq) -------------------------------------------------

@register_loader("gpqa_diamond")
def load_gpqa(cfg, limit, seed):
    ds = _safe_load(cfg["hf_id"], cfg.get("subset", "gpqa_diamond"), _split(cfg, "train"))
    out = []
    for i, row in _iter_limited(ds, limit):
        correct = str(pick(row, "Correct Answer")).strip()
        incorrect = [str(pick(row, f"Incorrect Answer {k}")).strip() for k in (1, 2, 3)]
        options = [correct, *incorrect]
        random.Random(seed * 100003 + i).shuffle(options)
        question, choices = format_mcq(str(pick(row, "Question")), options)
        gold = "ABCD"[options.index(correct)]
        out.append(ReasoningExample(str(i), "gpqa_diamond", question, gold, "mcq", choices,
                                    meta={"subdomain": row.get("Subdomain")}))
    return out


@register_loader("mmlu_pro")
def load_mmlu_pro(cfg, limit, seed):
    ds = _safe_load(cfg["hf_id"], cfg.get("subset"), _split(cfg, "test"))
    out = []
    for i, row in _iter_limited(ds, limit):
        options = [o for o in row["options"] if str(o).strip().upper() != "N/A"]
        question, choices = format_mcq(row["question"], options)
        gold = pick(row, "answer")
        if gold not in choices:  # fall back to index
            gold = "ABCDEFGHIJ"[int(row["answer_index"])]
        out.append(ReasoningExample(str(i), "mmlu_pro", question, gold, "mcq", choices,
                                    meta={"category": row.get("category")}))
    return out


@register_loader("arc_challenge")
def load_arc(cfg, limit, seed):
    ds = _safe_load(cfg["hf_id"], cfg.get("subset", "ARC-Challenge"), _split(cfg, "test"))
    out = []
    for i, row in _iter_limited(ds, limit):
        texts = row["choices"]["text"]
        labels = row["choices"]["label"]
        question, choices = format_mcq(row["question"], texts)
        gold = "ABCDEFG"[labels.index(row["answerKey"])]
        out.append(ReasoningExample(str(i), "arc_challenge", question, gold, "mcq", choices))
    return out


@register_loader("commonsense_qa")
def load_csqa(cfg, limit, seed):
    ds = _safe_load(cfg["hf_id"], cfg.get("subset"), _split(cfg, "validation"))
    out = []
    for i, row in _iter_limited(ds, limit):
        texts = row["choices"]["text"]
        labels = row["choices"]["label"]
        question, choices = format_mcq(row["question"], texts)
        gold = "ABCDE"[labels.index(row["answerKey"])]
        out.append(ReasoningExample(str(i), "commonsense_qa", question, gold, "mcq", choices))
    return out


# --- logical reasoning (mcq) ---------------------------------------------------

@register_loader("logiqa")
def load_logiqa(cfg, limit, seed):
    ds = _safe_load(cfg["hf_id"], cfg.get("subset"), _split(cfg, "test"))
    out = []
    for i, row in _iter_limited(ds, limit):
        context = pick(row, "context", default="")
        query = pick(row, "query", "question", default="")
        options = pick(row, "options", default=[])
        question, choices = format_mcq(f"{context}\n\n{query}", list(options))
        gold = "ABCD"[int(pick(row, "correct_option", "answer", default=0))]
        out.append(ReasoningExample(str(i), "logiqa", question, gold, "mcq", choices))
    return out


# --- diverse hard reasoning (freeform; aggregated over tasks) -------------------

@register_loader("bbh")
def load_bbh(cfg, limit, seed):
    tasks = cfg.get("tasks") or ["causal_judgement", "formal_fallacies",
                                 "logical_deduction_three_objects", "navigate",
                                 "date_understanding", "sports_understanding"]
    per_task = None
    if limit is not None:
        per_task = max(1, limit // len(tasks))
    out: list[ReasoningExample] = []
    for task in tasks:
        ds = _safe_load(cfg["hf_id"], task, _split(cfg, "test"))
        for j, row in _iter_limited(ds, per_task):
            q = pick(row, "input", "question")
            gold = str(pick(row, "target", "answer")).strip()
            out.append(ReasoningExample(f"{task}-{j}", "bbh", q, gold, "freeform",
                                        meta={"task": task}))
            if limit is not None and len(out) >= limit:
                return out
    return out
