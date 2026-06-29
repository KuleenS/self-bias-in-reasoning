"""Small shared helpers for paths and JSONL I/O."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def slugify(model_name: str) -> str:
    """Turn a HF/OpenRouter model id into a filesystem-safe short name.

    `Qwen/Qwen3-32B` -> `Qwen3-32B`; `openai/gpt-5` -> `gpt-5`.
    """
    return model_name.rstrip("/").split("/")[-1].replace(":", "-")


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_done_indices(path: str | Path, key: str = "index") -> set[int]:
    """Indices already present in a JSONL output, for resumable runs."""
    path = Path(path)
    if not path.exists():
        return set()
    done: set[int] = set()
    for rec in read_jsonl(path):
        try:
            done.add(rec[key])
        except KeyError:
            pass
    return done
