"""End-to-end orchestration: run all models x datasets, optionally under a D-optimal design.

Backends are built once per model and reused across datasets/cells (important for GPU loads).
For multi-model *offline* vLLM runs, prefer one model per SLURM job (see slurm/) — freeing a
vLLM engine mid-process is unreliable.
"""

from __future__ import annotations

import gc
from collections import defaultdict

from selfbias.config import experiment_config, results_dir
from selfbias.doe import design_to_manifest, save_manifest, select_doptimal
from selfbias.evaluate import run_evaluation
from selfbias.generate import run_generation
from selfbias.models.registry import ModelSpec, resolve_models


def resolve_pool(pool: str | None = None, models: list[str] | None = None) -> list[ModelSpec]:
    if models:
        return resolve_models(models)
    exp = experiment_config()
    pool = pool or exp.get("default_pool", "mixed")
    names = exp["pools"][pool]
    return resolve_models(names)


def _free(backend) -> None:
    try:
        del backend
        gc.collect()
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def generate_all(models: list[ModelSpec], datasets: list[str], n_prompts: int | None,
                 seed: int = 0, mode: str | None = None) -> None:
    from selfbias.inference.factory import build_backend

    for spec in models:
        backend = build_backend(spec, mode)
        for ds in datasets:
            try:
                run_generation(spec, ds, n_prompts, seed, backend=backend)
            except Exception as e:  # noqa: BLE001
                print(f"[generate] FAILED {spec.short} x {ds}: {e}")
        _free(backend)


def evaluate_cells(cells: list[tuple[str, str]], datasets: list[str],
                   mode: str | None = None) -> None:
    """cells: (generator_name, evaluator_name) pairs; each run on every dataset."""
    from selfbias.inference.factory import build_backend
    from selfbias.models.registry import get_model

    by_eval: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for gen, ev in cells:
        for ds in datasets:
            by_eval[ev].append((gen, ds))

    for ev_name, jobs in by_eval.items():
        ev = get_model(ev_name)
        backend = build_backend(ev, mode)
        for gen_name, ds in jobs:
            try:
                run_evaluation(ev, get_model(gen_name), ds, backend=backend)
            except Exception as e:  # noqa: BLE001
                print(f"[evaluate] FAILED {ev.short} on {gen_name} x {ds}: {e}")
        _free(backend)


def full_square(models: list[ModelSpec]) -> list[tuple[str, str]]:
    return [(g.name, e.name) for g in models for e in models]


def run_all(models: list[ModelSpec], datasets: list[str], n_prompts: int | None,
            mode_design: str = "full", budget: int | None = None, backend: str | None = None,
            seed: int = 0, do_generate: bool = True, do_evaluate: bool = True) -> list[tuple[str, str]]:
    if do_generate:
        generate_all(models, datasets, n_prompts, seed, backend)

    if mode_design == "doe":
        design = select_doptimal([m.name for m in models], budget or len(models) * 3, seed)
        save_manifest(design_to_manifest(design, datasets), results_dir() / "doe_manifest.json")
        cells = design.cells
        print(f"[doe] {design.budget}/{design.n_full} cells, "
              f"D-efficiency {design.d_efficiency:.3f}, saving {design.inference_saving:.1%}")
    else:
        cells = full_square(models)

    if do_evaluate:
        evaluate_cells(cells, datasets, backend)
    return cells


def run_cross_analysis(models: list[ModelSpec], datasets: list[str], n_prompts: int | None,
                       budget: int, backend: str | None = None, seed: int = 0,
                       method: str = "bayes", do_generate: bool = True) -> dict:
    from selfbias.aggregate import build_long_table, save_long_table
    from selfbias.analysis.mixed_effects import fit_self_bias, per_evaluator_breakdown

    run_all(models, datasets, n_prompts, mode_design="doe", budget=budget, backend=backend,
            seed=seed, do_generate=do_generate, do_evaluate=True)

    df = build_long_table(new=True, legacy=False)
    save_long_table(df)
    result = fit_self_bias(df, method=method)
    breakdown = per_evaluator_breakdown(df)
    return {"fit": result, "breakdown": breakdown, "n_obs": len(df)}
