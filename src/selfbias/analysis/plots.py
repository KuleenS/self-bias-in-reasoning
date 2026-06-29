"""Endorsement heatmaps: P(evaluator says "valid") for each (evaluator, generator) cell.

The diagonal is self-endorsement; self-bias shows up as a brighter diagonal. Adapted from the
old plot_heatmaps.py but driven by the long judgments table.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def endorsement_matrix(df: pd.DataFrame, dataset: str | None = None) -> pd.DataFrame:
    """Rows = evaluator, cols = generator, value = mean response (endorsement rate)."""
    if dataset is not None:
        df = df[df["dataset"] == dataset]
    return df.pivot_table(index="evaluator", columns="generator", values="response", aggfunc="mean")


def plot_endorsement_heatmap(df: pd.DataFrame, out_path: Path, dataset: str | None = None,
                             title: str | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    mat = endorsement_matrix(df, dataset)
    evaluators = list(mat.index)
    generators = list(mat.columns)
    values = mat.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(max(4, len(generators)), max(3, len(evaluators))))
    im = ax.imshow(values, vmin=0.0, vmax=1.0, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(generators)), generators, rotation=45, ha="right")
    ax.set_yticks(range(len(evaluators)), evaluators)
    ax.set_xlabel("generator (whose chain)")
    ax.set_ylabel("evaluator (the judge)")
    ax.set_title(title or f"Endorsement rate{f' — {dataset}' if dataset else ''}")

    for i in range(len(evaluators)):
        for j in range(len(generators)):
            v = values[i, j]
            if not np.isnan(v):
                # outline the self (diagonal) cells
                if evaluators[i] == generators[j]:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                               edgecolor="blue", lw=2))
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="black" if v < 0.6 else "white", fontsize=9)

    fig.colorbar(im, ax=ax, label="P(valid)")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
