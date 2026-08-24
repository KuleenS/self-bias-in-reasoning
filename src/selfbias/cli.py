"""`selfbias` command-line interface.

Subcommands: models, datasets, generate, evaluate, doe, run-all, aggregate, analyze, plot.
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="Measuring self-bias in LLM reasoning.")


@app.command("models")
def cmd_models() -> None:
    """List the model registry."""
    from selfbias.models.registry import load_registry

    for name, s in load_registry().items():
        flag = "reasoning" if s.reasoning else "baseline"
        typer.echo(f"{s.short:8s} {s.backend:13s} {flag:9s} {name}")


@app.command("datasets")
def cmd_datasets() -> None:
    """List the available datasets."""
    from selfbias.config import datasets_config

    for name, d in datasets_config().items():
        typer.echo(f"{name:16s} {d.get('answer_type',''):9s} {d.get('domain',''):11s} "
                   f"{d.get('description','')}")


@app.command("generate")
def cmd_generate(
    model: list[str] = typer.Option(..., "--model", "-m", help="Model name(s) or short label(s)."),
    dataset: list[str] = typer.Option(None, "--dataset", "-d", help="Dataset(s); default = all."),
    n_prompts: int = typer.Option(None, "--n-prompts", "-n"),
    backend: str = typer.Option(None, "--backend", help="Override backend mode."),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Generate reasoning chains for model(s) over dataset(s)."""
    from selfbias.config import experiment_config
    from selfbias.generate import run_generation
    from selfbias.inference.factory import build_backend
    from selfbias.models.registry import get_model

    datasets = dataset or experiment_config()["datasets"]
    for m in model:
        spec = get_model(m)
        be = build_backend(spec, backend)  # load the model once, reuse across datasets
        for ds in datasets:
            try:
                run_generation(spec, ds, n_prompts, seed, mode=backend, backend=be)
            except Exception as e:
                # One bad dataset (e.g. a transient/gone HF repo) shouldn't sink the rest of
                # this model's run, and definitely shouldn't leave the job hanging.
                typer.echo(f"[generate] {spec.short} x {ds}: FAILED ({e!r}); continuing", err=True)


@app.command("evaluate")
def cmd_evaluate(
    evaluator: str = typer.Option(..., "--evaluator", "-e"),
    generator: str = typer.Option(..., "--generator", "-g"),
    dataset: list[str] = typer.Option(None, "--dataset", "-d"),
    backend: str = typer.Option(None, "--backend"),
) -> None:
    """Have an evaluator judge a generator's chains on dataset(s)."""
    from selfbias.config import experiment_config
    from selfbias.evaluate import run_evaluation
    from selfbias.inference.factory import build_backend
    from selfbias.models.registry import get_model

    datasets = dataset or experiment_config()["datasets"]
    ev_spec = get_model(evaluator)
    be = build_backend(ev_spec, backend)  # load the model once, reuse across datasets
    for ds in datasets:
        try:
            run_evaluation(ev_spec, generator, ds, mode=backend, backend=be)
        except Exception as e:
            typer.echo(f"[evaluate] {ev_spec.short} on {generator} x {ds}: FAILED ({e!r}); "
                       "continuing", err=True)


@app.command("doe")
def cmd_doe(
    pool: str = typer.Option(None, "--pool", help="Named pool from experiment.yaml."),
    model: list[str] = typer.Option(None, "--model", "-m"),
    budget: int = typer.Option(None, "--budget", "-b", help="Cells to keep from the GxG square."),
    seed: int = typer.Option(0, "--seed"),
    output: Path = typer.Option(None, "--output", "-o"),
) -> None:
    """Build a D-optimal generator x evaluator design and write the run manifest."""
    from selfbias.config import experiment_config, results_dir
    from selfbias.doe import design_to_manifest, save_manifest, select_doptimal
    from selfbias.orchestrate import resolve_pool

    exp = experiment_config()
    specs = resolve_pool(pool, model)
    budget = budget or exp.get("doe_budget", len(specs) * 3)
    design = select_doptimal([s.name for s in specs], budget, seed)
    out = output or results_dir() / "doe_manifest.json"
    save_manifest(design_to_manifest(design, exp["datasets"]), out)
    typer.echo(f"D-optimal design: {design.budget}/{design.n_full} cells "
               f"({design.n_params} params)")
    typer.echo(f"  D-efficiency vs full square: {design.d_efficiency:.3f}")
    typer.echo(f"  inference saving:            {design.inference_saving:.1%}")
    typer.echo(f"  manifest -> {out}")


@app.command("run-all")
def cmd_run_all(
    pool: str = typer.Option(None, "--pool"),
    model: list[str] = typer.Option(None, "--model", "-m"),
    dataset: list[str] = typer.Option(None, "--dataset", "-d"),
    n_prompts: int = typer.Option(None, "--n-prompts", "-n"),
    design: str = typer.Option("full", "--design", help="full | doe"),
    budget: int = typer.Option(None, "--budget", "-b"),
    backend: str = typer.Option(None, "--backend"),
    seed: int = typer.Option(0, "--seed"),
    generate: bool = typer.Option(True, "--generate/--no-generate"),
    evaluate: bool = typer.Option(True, "--evaluate/--no-evaluate"),
) -> None:
    """Run the full pipeline: generate all chains, then evaluate (full square or D-optimal)."""
    from selfbias.config import experiment_config
    from selfbias.orchestrate import resolve_pool, run_all

    exp = experiment_config()
    specs = resolve_pool(pool, model)
    datasets = dataset or exp["datasets"]
    n = n_prompts if n_prompts is not None else exp.get("n_prompts")
    run_all(specs, datasets, n, mode_design=design, budget=budget, backend=backend, seed=seed,
            do_generate=generate, do_evaluate=evaluate)


@app.command("aggregate")
def cmd_aggregate(
    output: Path = typer.Option(None, "--output", "-o"),
    legacy: bool = typer.Option(False, "--legacy", help="Include the original FOLIO results."),
    new: bool = typer.Option(True, "--new/--no-new"),
) -> None:
    """Flatten judgments into results/judgments_long.parquet."""
    from selfbias.aggregate import build_long_table, save_long_table

    df = build_long_table(new=new, legacy=legacy)
    path = save_long_table(df, output)
    typer.echo(f"wrote {len(df)} rows -> {path}")


@app.command("analyze")
def cmd_analyze(
    input: Path = typer.Option(None, "--input", "-i", help="Long parquet; default results/judgments_long.parquet."),
    method: str = typer.Option("bayes", "--method", help="bayes (logistic GLMM) | lpm (linear)."),
    legacy: bool = typer.Option(False, "--legacy", help="Rebuild table from original FOLIO results."),
    rebuild: bool = typer.Option(False, "--rebuild"),
) -> None:
    """Fit the self-bias mixed-effects model and print the report."""
    import pandas as pd

    from selfbias.aggregate import build_long_table, save_long_table
    from selfbias.analysis.mixed_effects import fit_self_bias, per_evaluator_breakdown
    from selfbias.config import results_dir

    path = input or results_dir() / "judgments_long.parquet"
    if rebuild or legacy or not Path(path).exists():
        df = build_long_table(new=True, legacy=legacy)
        save_long_table(df, path)
    else:
        df = pd.read_parquet(path)

    typer.echo(f"\nobservations: {len(df)} | generators: {df['generator'].nunique()} | "
               f"evaluators: {df['evaluator'].nunique()} | datasets: {df['dataset'].nunique()}")
    fit = fit_self_bias(df, method=method)
    typer.echo("\n=== Self-bias fixed effect (is_self) ===")
    for k, v in fit.items():
        typer.echo(f"  {k:14s} {v}")

    typer.echo("\n=== Per-evaluator (Holm-corrected, one-sided H1: beta>0) ===")
    bd = per_evaluator_breakdown(df)
    typer.echo(bd.to_string(index=False) if not bd.empty else "  (insufficient self/cross cells)")


@app.command("plot")
def cmd_plot(
    input: Path = typer.Option(None, "--input", "-i"),
    dataset: str = typer.Option(None, "--dataset", "-d", help="Single dataset; default = pooled."),
    output: Path = typer.Option(None, "--output", "-o"),
) -> None:
    """Write an endorsement heatmap (evaluator x generator)."""
    import pandas as pd

    from selfbias.analysis.plots import plot_endorsement_heatmap
    from selfbias.config import results_dir

    path = input or results_dir() / "judgments_long.parquet"
    df = pd.read_parquet(path)
    out = output or results_dir() / "plots" / f"endorsement_{dataset or 'pooled'}.png"
    plot_endorsement_heatmap(df, out, dataset=dataset)
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()
