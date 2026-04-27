#!/usr/bin/env python3
"""
Generate heatmaps from a combined_metrics.json produced by batch_compute_metrics.py.

Each heatmap has judges on the Y-axis and judged models on the X-axis.
One PNG is written per metric.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


METRICS = ["accuracy", "precision", "recall", "f1"]
MODEL_ORDER = ["qwen", "olmo", "ds", "phi"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True,
                   help="Path to combined_metrics.json.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Directory for output PNGs (default: same dir as --input).")
    p.add_argument("--metrics", nargs="+", default=METRICS,
                   help=f"Metrics to plot (default: {METRICS}).")
    return p.parse_args()


def build_dataframe(combined: dict, metric: str) -> pd.DataFrame:
    judges = MODEL_ORDER
    judged_set = MODEL_ORDER

    data = {}
    for judge in judges:
        row = {}
        for judged in judged_set:
            entry = combined.get(judge, {}).get(judged)
            row[judged] = entry.get(metric) if entry else None
        data[judge] = row

    df = pd.DataFrame(data, index=judged_set).T  # rows=judges, cols=judged
    return df.reindex(index=judges, columns=judged_set)


def plot_heatmap(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    values = df.astype(float).to_numpy()
    judges = list(df.index)
    judged = list(df.columns)

    fig, ax = plt.subplots(figsize=(max(4, len(judged)), max(3, len(judges))))
    im = ax.imshow(values, vmin=0.0, vmax=1.0, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(judged)))
    ax.set_xticklabels(judged, rotation=45, ha="right")
    ax.set_yticks(range(len(judges)))
    ax.set_yticklabels(judges)

    for i in range(len(judges)):
        for j in range(len(judged)):
            v = values[i, j]
            if not np.isnan(v):
                text_color = "black" if v < 0.6 else "white"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=9, color=text_color)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric)
    ax.set_title(metric)
    ax.set_xlabel("judged model")
    ax.set_ylabel("judge model")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    args = parse_args()
    combined = json.loads(args.input.read_text())
    out_dir = args.output_dir or args.input.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric in args.metrics:
        df = build_dataframe(combined, metric)
        if df.empty or df.astype(float).isnull().all().all():
            print(f"skip {metric}: no data")
            continue
        plot_heatmap(df, metric, out_dir / f"{metric}.png")


if __name__ == "__main__":
    main()
