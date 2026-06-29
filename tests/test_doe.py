import random

import numpy as np

from selfbias.doe import _logdet, _row, select_doptimal


def _cell_logdet(models, cells):
    g = len(models)
    idx = {m: i for i, m in enumerate(models)}
    rows = np.array([_row(idx[gen], idx[ev], g) for gen, ev in cells])
    return _logdet(rows)


def test_diagonal_forced_and_budget_respected():
    models = ["a", "b", "c", "d"]
    design = select_doptimal(models, budget=12, seed=0)
    assert design.budget == 12
    assert len(design.cells) == 12
    # all self (diagonal) cells present
    for m in models:
        assert (m, m) in design.cells
    assert abs(design.inference_saving - (1 - 12 / 16)) < 1e-9


def test_full_square_when_budget_exceeds():
    models = ["a", "b", "c", "d"]
    design = select_doptimal(models, budget=99, seed=0)
    assert len(design.cells) == 16


def test_doptimal_beats_random_selection():
    models = ["a", "b", "c", "d"]
    budget = 11
    design = select_doptimal(models, budget=budget, seed=0)
    chosen_ld = _cell_logdet(models, design.cells)

    all_cells = [(g, e) for g in models for e in models]
    diag = [(m, m) for m in models]
    rng = random.Random(123)
    best_random = -1e18
    for _ in range(50):
        extra = [c for c in all_cells if c not in diag]
        rng.shuffle(extra)
        cells = diag + extra[: budget - len(diag)]
        best_random = max(best_random, _cell_logdet(models, cells))
    assert chosen_ld >= best_random - 1e-9
    assert design.d_efficiency > 0
