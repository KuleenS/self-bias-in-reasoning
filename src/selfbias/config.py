"""Resolve and load YAML configs and project paths.

Configs live in `configs/` at the repo root and are the single edit point for datasets,
models, and experiment defaults. Paths can be overridden with env vars (useful on clusters):
`SELFBIAS_CONFIG_DIR`, `SELFBIAS_ROOT`, `SELFBIAS_DATA_DIR`, `SELFBIAS_RESULTS_DIR`.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    env = os.getenv("SELFBIAS_ROOT")
    if env:
        return Path(env)
    # src/selfbias/config.py -> parents[2] == repo root
    return Path(__file__).resolve().parents[2]


def config_dir() -> Path:
    env = os.getenv("SELFBIAS_CONFIG_DIR")
    if env:
        return Path(env)
    candidate = project_root() / "configs"
    if candidate.exists():
        return candidate
    return Path.cwd() / "configs"


def data_dir() -> Path:
    return Path(os.getenv("SELFBIAS_DATA_DIR", str(project_root() / "data")))


def results_dir() -> Path:
    return Path(os.getenv("SELFBIAS_RESULTS_DIR", str(project_root() / "results")))


@lru_cache(maxsize=None)
def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_yaml(name: str) -> dict[str, Any]:
    return _load_yaml(str(config_dir() / name))


def datasets_config() -> dict[str, Any]:
    return load_yaml("datasets.yaml")


def models_config() -> dict[str, Any]:
    return load_yaml("models.yaml")


def experiment_config() -> dict[str, Any]:
    return load_yaml("experiment.yaml")


def get_dataset_config(name: str) -> dict[str, Any]:
    cfg = datasets_config()
    if name not in cfg:
        raise KeyError(f"dataset '{name}' not in datasets.yaml; available: {sorted(cfg)}")
    return {"name": name, **cfg[name]}


# --- canonical on-disk paths ---------------------------------------------------

def chains_path(dataset: str, model_slug: str) -> Path:
    return data_dir() / "chains" / dataset / f"{model_slug}.jsonl"


def judgments_path(dataset: str, evaluator_slug: str, generator_slug: str) -> Path:
    return data_dir() / "judgments" / dataset / f"{evaluator_slug}__on__{generator_slug}.jsonl"
