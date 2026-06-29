#!/usr/bin/env python3
"""Run all models x datasets: generate chains, then evaluate (full square or D-optimal).

Examples:
    python scripts/run_all.py --pool open --design doe --budget 18
    python scripts/run_all.py --model qwen --model ds --dataset gsm8k --no-evaluate
"""

from __future__ import annotations

import argparse

from selfbias.config import experiment_config
from selfbias.orchestrate import resolve_pool, run_all


def main() -> None:
    exp = experiment_config()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", default=None, help="Named pool from experiment.yaml (open/api/mixed).")
    p.add_argument("--model", action="append", help="Model name/short (repeatable).")
    p.add_argument("--dataset", action="append", help="Dataset (repeatable); default = all.")
    p.add_argument("--n-prompts", type=int, default=exp.get("n_prompts"))
    p.add_argument("--design", default="full", choices=["full", "doe"])
    p.add_argument("--budget", type=int, default=exp.get("doe_budget"))
    p.add_argument("--backend", default=None, help="Override backend mode.")
    p.add_argument("--seed", type=int, default=exp.get("seed", 0))
    p.add_argument("--no-generate", action="store_true")
    p.add_argument("--no-evaluate", action="store_true")
    a = p.parse_args()

    specs = resolve_pool(a.pool, a.model)
    datasets = a.dataset or exp["datasets"]
    run_all(specs, datasets, a.n_prompts, mode_design=a.design, budget=a.budget,
            backend=a.backend, seed=a.seed, do_generate=not a.no_generate,
            do_evaluate=not a.no_evaluate)


if __name__ == "__main__":
    main()
