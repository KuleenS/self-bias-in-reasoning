"""Answer extraction and correctness checking — the dataset-agnostic ground truth.

This replaces the FOLIO-only Lean pipeline: a chain is "correct" if its extracted final answer
matches the gold answer. `answer_type` (numeric / mcq / freeform) selects the matcher.
"""

from __future__ import annotations

import re

_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")
_ANSWER_CUE_RE = re.compile(r"(?:final answer|answer)\s*(?:is|:|=)?\s*", re.IGNORECASE)
_LETTER_RE = re.compile(r"\(?([A-J])\)?", re.IGNORECASE)


# --- low-level helpers ---------------------------------------------------------

def extract_boxed(text: str) -> str | None:
    """Return the content of the last \\boxed{...} with balanced braces."""
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    i = text.find("{", idx)
    if i == -1:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j].strip()
    return None


def _after_answer_cue(text: str) -> str | None:
    matches = list(_ANSWER_CUE_RE.finditer(text))
    if not matches:
        return None
    tail = text[matches[-1].end() :].strip()
    if not tail:
        return None
    return tail.splitlines()[0].strip()


def _to_float(s: str) -> float | None:
    s = s.strip().strip("$").replace(",", "").replace("%", "").rstrip(".")
    if "/" in s:  # simple fraction
        parts = s.split("/")
        if len(parts) == 2:
            try:
                return float(parts[0]) / float(parts[1])
            except (ValueError, ZeroDivisionError):
                return None
    try:
        return float(s)
    except ValueError:
        return None


def _last_number(text: str) -> str | None:
    nums = _NUMBER_RE.findall(text)
    return nums[-1].replace(",", "") if nums else None


def _normalize_freeform(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.replace("$", "").replace("\\,", "").replace(" ", "")
    s = s.strip().rstrip(".")
    return s


# --- public API ----------------------------------------------------------------

def extract_answer(text: str, answer_type: str) -> str | None:
    if not text:
        return None
    cue = _after_answer_cue(text)
    if answer_type == "mcq":
        source = cue or text[-80:]
        m = list(_LETTER_RE.finditer(source))
        return m[-1].group(1).upper() if m else None
    if answer_type == "numeric":
        boxed = extract_boxed(text)
        return _last_number(cue or boxed or text)
    # freeform
    boxed = extract_boxed(text)
    return (boxed or cue or text.strip().splitlines()[-1].strip()) or None


def is_correct(extracted: str | None, gold: str, answer_type: str) -> bool:
    if extracted is None:
        return False
    if answer_type == "mcq":
        return extracted.strip().upper() == gold.strip().upper()
    if answer_type == "numeric":
        a, b = _to_float(extracted), _to_float(gold)
        if a is not None and b is not None:
            return abs(a - b) < 1e-4
        return extracted.strip() == gold.strip()
    # freeform: normalized string match, with a numeric fallback
    if _normalize_freeform(extracted) == _normalize_freeform(gold):
        return True
    a, b = _to_float(extracted), _to_float(gold)
    return a is not None and b is not None and abs(a - b) < 1e-4


def grade(text: str, gold: str, answer_type: str) -> tuple[str | None, bool]:
    """Return (extracted_answer, is_correct)."""
    extracted = extract_answer(text, answer_type)
    return extracted, is_correct(extracted, gold, answer_type)
