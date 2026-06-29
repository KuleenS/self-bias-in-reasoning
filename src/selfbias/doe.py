"""D-optimal design over the generator x evaluator square.

The full experiment is the G x G square of (generator, evaluator) cells, run on each dataset.
Evaluating every cell is the dominant inference cost. We instead pick a budget-sized subset of
cells that maximizes det(XᵀX) (D-optimality), so the self-bias coefficient and the model main
effects stay precisely estimable while we run far fewer cells.

Design matrix per cell (g, e): [intercept, is_self, effect-coded generator (G-1),
effect-coded evaluator (G-1)] -> 2G parameters. The G diagonal (self) cells are force-included
so `is_self` always has variation.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_RIDGE = 1e-6


@dataclass
class Design:
    models: list[str]
    cells: list[tuple[str, str]]  # (generator, evaluator)
    n_params: int
    n_full: int
    budget: int
    d_efficiency: float  # per-observation D-efficiency vs the full square (1.0 = as good)
    inference_saving: float  # fraction of cells skipped vs the full square


def _effect_code(level: int, n_levels: int) -> np.ndarray:
    v = np.zeros(n_levels - 1)
    if level < n_levels - 1:
        v[level] = 1.0
    else:
        v[:] = -1.0
    return v


def _row(gi: int, ei: int, g: int) -> np.ndarray:
    return np.concatenate(
        [[1.0, 1.0 if gi == ei else 0.0], _effect_code(gi, g), _effect_code(ei, g)]
    )


def _logdet(rows: np.ndarray) -> float:
    m = rows.T @ rows
    m = m + _RIDGE * np.eye(m.shape[0])
    sign, logdet = np.linalg.slogdet(m)
    return logdet if sign > 0 else -math.inf


def select_doptimal(models: list[str], budget: int, seed: int = 0, max_iter: int = 2000) -> Design:
    g = len(models)
    cells = [(gi, ei) for gi in range(g) for ei in range(g)]
    rows = np.array([_row(gi, ei, g) for gi, ei in cells])
    p = rows.shape[1]  # 2G
    diag = [k for k, (gi, ei) in enumerate(cells) if gi == ei]

    if budget < p:
        raise ValueError(f"budget {budget} < {p} parameters; design would be singular")
    if budget >= len(cells):
        selected = list(range(len(cells)))
    else:
        rng = random.Random(seed)
        selected = list(diag)
        # greedy forward selection
        while len(selected) < budget:
            best_k, best_ld = None, -math.inf
            order = list(range(len(cells)))
            rng.shuffle(order)  # break ties stochastically but reproducibly
            for k in order:
                if k in selected:
                    continue
                ld = _logdet(rows[selected + [k]])
                if ld > best_ld:
                    best_ld, best_k = ld, k
            selected.append(best_k)
        # Fedorov exchange refinement (forced diagonal cells stay)
        forced = set(diag)
        for _ in range(max_iter):
            cur_ld = _logdet(rows[selected])
            improved = False
            for ii, inp in enumerate(selected):
                if inp in forced:
                    continue
                for k in range(len(cells)):
                    if k in selected:
                        continue
                    trial = selected.copy()
                    trial[ii] = k
                    if _logdet(rows[trial]) > cur_ld + 1e-9:
                        selected = trial
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break

    n_full = len(cells)
    full_ld = _logdet(rows)
    sel_ld = _logdet(rows[selected])
    d_eff = math.exp((sel_ld - full_ld) / p) * (n_full / len(selected))
    saving = 1.0 - len(selected) / n_full
    cell_pairs = [(models[cells[k][0]], models[cells[k][1]]) for k in selected]
    return Design(models, cell_pairs, p, n_full, len(selected), d_eff, saving)


def design_to_manifest(design: Design, datasets: list[str]) -> dict:
    return {
        "models": design.models,
        "datasets": datasets,
        "n_params": design.n_params,
        "n_full_square": design.n_full,
        "budget": design.budget,
        "d_efficiency": round(design.d_efficiency, 4),
        "inference_saving": round(design.inference_saving, 4),
        "cells": [
            {"generator": g, "evaluator": e, "is_self": g == e} for g, e in design.cells
        ],
        "runs": [
            {"dataset": d, "generator": g, "evaluator": e}
            for d in datasets
            for g, e in design.cells
        ],
    }


def save_manifest(manifest: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    return path
