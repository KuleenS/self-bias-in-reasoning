"""Flatten per-cell judgments into one long table for the mixed-effects model.

Columns map 1:1 to the statistical model:
    response, is_self, generator, evaluator, dataset, prompt_id, generator_correct

Reads the new layout (data/judgments/{dataset}/{evaluator}__on__{generator}.jsonl) and, with
`legacy=True`, the original FOLIO results (results/{evaluator}_on_{generator}_full.jsonl) so the
headline finding can be reproduced through the refactored pipeline.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterator

import pandas as pd

from selfbias.config import data_dir, results_dir
from selfbias.utils import ensure_dir, read_jsonl

LONG_COLUMNS = ["response", "is_self", "generator", "evaluator", "dataset",
                "prompt_id", "generator_correct"]


def _as_float(x) -> float:
    if x is None:
        return math.nan
    return float(x)


def _new_rows() -> Iterator[dict]:
    for path in sorted((data_dir() / "judgments").glob("*/*.jsonl")):
        for rec in read_jsonl(path):
            ev, gen = rec["evaluator_short"], rec["generator_short"]
            yield {
                "response": 1 if rec.get("evaluator_judgment") is True else 0,
                "is_self": int(rec.get("is_self", ev == gen)),
                "generator": gen,
                "evaluator": ev,
                "dataset": rec["dataset"],
                "prompt_id": f"{rec['dataset']}-{rec['index']}",
                "generator_correct": _as_float(rec.get("generator_correct")),
            }


def _legacy_rows() -> Iterator[dict]:
    for path in sorted(results_dir().glob("*_on_*_full.jsonl")):
        body = path.stem[:-5] if path.stem.endswith("_full") else path.stem
        ev, _, gen = body.partition("_on_")
        for rec in read_jsonl(path):
            yield {
                "response": 1 if rec.get("evaluator_judgment") is True else 0,
                "is_self": int(ev == gen),
                "generator": gen,
                "evaluator": ev,
                "dataset": "folio",
                "prompt_id": f"folio-{rec['index']}",
                "generator_correct": math.nan,
            }


def build_long_table(new: bool = True, legacy: bool = False) -> pd.DataFrame:
    rows: list[dict] = []
    if new:
        rows.extend(_new_rows())
    if legacy:
        rows.extend(_legacy_rows())
    df = pd.DataFrame(rows, columns=LONG_COLUMNS)
    return df


def save_long_table(df: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or results_dir() / "judgments_long.parquet"
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)
    return path
