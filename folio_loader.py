from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import pandas as pd


_LABEL_MAP = {
    "true": 1,
    "false": 0,
    "unknown": 2,
}


def _read_jsonl_df(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True, encoding="utf-8")


def _read_json_df(path: Path) -> pd.DataFrame:
    # Try common wrappers, else treat as single row or list of dicts.
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj is None:
        return pd.DataFrame()
    if isinstance(obj, list):
        return pd.DataFrame(obj)
    if isinstance(obj, dict):
        for key in ("data", "examples", "items", "rows"):
            if key in obj and isinstance(obj[key], list):
                return pd.DataFrame(obj[key])
        return pd.DataFrame([obj])
    raise TypeError("Unsupported dataset format; expected JSON/JSONL with dict rows.")


@dataclass(frozen=True)
class FolioExample:
    story_id: Union[int, str, None]
    example_id: Union[int, str, None]
    premises: str
    premises_fol: str
    conclusion: str
    conclusion_fol: str
    label: Any
    raw: Dict[str, Any]


class FolioDataset:
    """
    Minimal dataset loader for FOLIO-style examples.

    It searches under: data/FOLIO by default.
    """

    def __init__(
        self,
        root: Union[str, Path] = "data/FOLIO",
        split: Optional[str] = None,
        path: Optional[Union[str, Path]] = None,
        label_map: Optional[Dict[str, int]] = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.path = Path(path) if path is not None else None
        self.label_map = {**_LABEL_MAP, **(label_map or {})}

        self._df = self._load_df()
        self._normalize_df()
        self._examples: List[FolioExample] = [self._to_example(r) for r in self._df.to_dict("records")]

    def _load_df(self) -> pd.DataFrame:
        # If an explicit path is given, load it.
        if self.path is not None:
            p = self.path
            if not p.is_absolute():
                p = self.root / p
            if p.suffix.lower() == ".jsonl":
                return _read_jsonl_df(p)
            if p.suffix.lower() == ".json":
                return _read_json_df(p)
            raise ValueError(f"Unsupported file type: {p}")

        # Otherwise, infer from split inside root.
        if self.split:
            split_key = self.split.lower()
            if split_key in ("val", "valid", "validation", "dev"):
                fname = "folio_validation.jsonl"
            elif split_key == "test":
                fname = "folio_test.jsonl"
            elif split_key == "train":
                fname = "folio_train.jsonl"
            else:
                raise ValueError(f"Unsupported split: {self.split!r}")
            candidates = [self.root / fname]
        else:
            candidates = [
                self.root / "folio_train.jsonl",
                self.root / "folio_validation.jsonl",
                self.root / "folio_test.jsonl",
            ]

        for p in candidates:
            if p.exists():
                return _read_jsonl_df(p)

        raise FileNotFoundError(
            f"Could not find FOLIO data under {self.root} "
            f"(split={self.split!r}). Looked for: {', '.join(str(p) for p in candidates)}"
        )

    def _normalize_df(self) -> None:
        # Unify column names for FOL fields if present.
        rename: Dict[str, str] = {}
        if "premises-FOL" in self._df.columns:
            rename["premises-FOL"] = "premises_fol"
        if "premises_FOL" in self._df.columns:
            rename["premises_FOL"] = "premises_fol"
        if "conclusion-FOL" in self._df.columns:
            rename["conclusion-FOL"] = "conclusion_fol"
        if "conclusion_FOL" in self._df.columns:
            rename["conclusion_FOL"] = "conclusion_fol"
        if rename:
            self._df = self._df.rename(columns=rename)

        # Ensure expected columns exist.
        for col, default in (
            ("story_id", None),
            ("example_id", None),
            ("premises", ""),
            ("premises_fol", ""),
            ("conclusion", ""),
            ("conclusion_fol", ""),
            ("label", None),
        ):
            if col not in self._df.columns:
                self._df[col] = default

        # Map labels
        if "label" in self._df.columns:
            def _map(x: Any) -> Any:
                if isinstance(x, str):
                    k = x.strip().lower()
                    return self.label_map.get(k, x)
                return x

            self._df["label"] = self._df["label"].map(_map)

    def _to_example(self, row: Dict[str, Any]) -> FolioExample:
        return FolioExample(
            story_id=row.get("story_id", None),
            example_id=row.get("example_id", None),
            premises=row.get("premises", "") or "",
            premises_fol=row.get("premises_fol", "") or "",
            conclusion=row.get("conclusion", "") or "",
            conclusion_fol=row.get("conclusion_fol", "") or "",
            label=row.get("label", None),
            raw=row,
        )

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self._examples[idx]
        return {
            "story_id": ex.story_id,
            "example_id": ex.example_id,
            "premises": ex.premises,
            "premises_fol": ex.premises_fol,
            "conclusion": ex.conclusion,
            "conclusion_fol": ex.conclusion_fol,
            "label": ex.label,
        }

    def iter_examples(self) -> Iterator[FolioExample]:
        yield from self._examples

    def get_raw(self, idx: int) -> Dict[str, Any]:
        return self._examples[idx].raw

    def to_pandas(self) -> pd.DataFrame:
        return self._df.copy()


__all__ = ["FolioDataset", "FolioExample"]