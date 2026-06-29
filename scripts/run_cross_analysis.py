#!/usr/bin/env python3
"""D-optimal cross-model analysis: build the design, run only the selected cells, fit the model.

Saves results/doe_manifest.json and results/judgments_long.parquet, then prints the self-bias
fixed effect and the per-evaluator Holm-corrected breakdown.

Example:
    python scripts/run_cross_analysis.py --pool open --budget 18 --method bayes
"""

from __future__ import annotations

import argparse

from selfbias.config import experiment_config
from selfbias.orchestrate import resolve_pool, run_cross_analysis


def main() -> None:
    exp = experiment_config()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", default=None)
    p.add_argument("--model", action="append")
    p.add_argument("--dataset", action="append")
    p.add_argument("--n-prompts", type=int, default=exp.get("n_prompts"))
    p.add_argument("--budget", type=int, default=exp.get("doe_budget"))
    p.add_argument("--backend", default=None)
    p.add_argument("--method", default="bayes", choices=["bayes", "lpm"])
    p.add_argument("--seed", type=int, default=exp.get("seed", 0))
    p.add_argument("--no-generate", action="store_true")
    a = p.parse_args()

    specs = resolve_pool(a.pool, a.model)
    datasets = a.dataset or exp["datasets"]
    out = run_cross_analysis(specs, datasets, a.n_prompts, a.budget, backend=a.backend,
                             seed=a.seed, method=a.method, do_generate=not a.no_generate)

    print("\n=== Self-bias fixed effect ===")
    for k, v in out["fit"].items():
        print(f"  {k:14s} {v}")
    print("\n=== Per-evaluator breakdown ===")
    print(out["breakdown"].to_string(index=False))


if __name__ == "__main__":
    main()
