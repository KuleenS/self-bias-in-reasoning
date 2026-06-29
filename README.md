# Self-Bias in Reasoning

**Do reasoning models judge their *own* reasoning chains as correct more often than they judge
other models' chains?**

Each model in a pool both **generates** reasoning chains and **evaluates** chains (its own and
others'). The question is whether a model's endorsement of a chain goes up when *it* produced the
chain. We estimate the effect with a crossed mixed-effects model:

```
score_ijkl = β0 + β_self · I_self
             + u_generator_i + u_evaluator_j + u_dataset_k + u_prompt_l + ε
```

- `score` — does evaluator *j* judge generator *i*'s chain on prompt *l* of dataset *k* as correct?
- `I_self` — 1 when generator == evaluator. **`β_self` is the self-bias effect.**
- `u_*` — crossed random intercepts for generator, evaluator, dataset, and prompt.

Ground truth (whether a chain is *actually* correct) comes from **answer-matching**: we extract the
chain's final answer and compare it to the dataset's gold answer. It is recorded for analysis but
never shown to the evaluator.

## Install

Uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync                 # core: analysis + API inference (no GPU)
uv sync --extra vllm    # add local GPU inference (vLLM + torch)
cp .env.example .env    # add OPENROUTER_API_KEY / HF_TOKEN as needed
```

Everything is exposed through the `selfbias` CLI (and the `scripts/` orchestrators):

```bash
uv run selfbias --help
uv run selfbias models      # list the model pool
uv run selfbias datasets    # list the datasets
```

## Datasets (10)

`configs/datasets.yaml` is the single edit point (HF id / subset / split / answer type).

| dataset | domain | answer type |
|---|---|---|
| gsm8k, math500, aime | math | numeric / freeform |
| gpqa_diamond, mmlu_pro, arc_challenge | science / knowledge | mcq |
| folio, logiqa | logic | mcq |
| bbh | mixed hard | freeform |
| commonsense_qa | commonsense | mcq |

`answer_type` (`numeric` / `mcq` / `freeform`) drives both the generation instruction and the
answer extractor in [`answers.py`](src/selfbias/answers.py). GPQA is gated — set `HF_TOKEN`.

## Models

`configs/models.yaml` defines the pool. Each model both generates and evaluates. Open-weight
models run on local GPUs via vLLM; frontier models run through OpenRouter. **Verify OpenRouter
slugs before a paid run — they drift.**

- **Open (vLLM):** Qwen3-32B, DeepSeek-R1-Distill-32B, OLMo-3.1-32B-Think, Phi-4-reasoning, QwQ-32B, Gemma-3-27B (baseline).
- **Frontier (OpenRouter):** GPT-5-class, Claude (thinking), Gemini-2.5, DeepSeek-R1, Grok.

Three interchangeable inference backends (`src/selfbias/inference/`): `vllm_offline` (local batch),
`vllm_online` (an OpenAI-compatible vLLM server), `openrouter`.

## Pipeline

```bash
# 1. Generate reasoning chains (one model x one+ datasets)
uv run selfbias generate -m Qwen/Qwen3-32B -d gsm8k -d folio -n 200
#    -> data/chains/{dataset}/{model}.jsonl  (with extracted answer + is_correct)

# 2. Cross-model evaluation (evaluator judges a generator's chains)
uv run selfbias evaluate -e Qwen/Qwen3-32B -g microsoft/Phi-4-reasoning -d gsm8k
#    -> data/judgments/{dataset}/{evaluator}__on__{generator}.jsonl

# 3. Aggregate to the long table the model consumes
uv run selfbias aggregate                 # -> results/judgments_long.parquet

# 4. Fit the self-bias mixed-effects model
uv run selfbias analyze --method bayes    # logistic crossed-RE GLMM (statsmodels, pure Python)
uv run selfbias plot                      # endorsement heatmap (diagonal = self)
```

### Run everything

```bash
# all models x all datasets, full square
uv run python scripts/run_all.py --pool open

# ...or save inference with a D-optimal subset of the square, then fit
uv run python scripts/run_cross_analysis.py --pool open --budget 18
```

On a cluster, run one model per job: `sbatch slurm/generate.job <MODEL>` and
`sbatch slurm/evaluate.job <EVALUATOR> <GENERATOR>`.

## D-optimal design (saving inference)

The full experiment is the **G × G generator × evaluator square** run on every dataset; evaluating
every cell is the dominant cost. [`doe.py`](src/selfbias/doe.py) instead selects a budget-sized
subset of cells that maximizes `det(XᵀX)` (D-optimality) via Fedorov exchange, force-including all
G self (diagonal) cells so `β_self` stays estimable.

```bash
uv run selfbias doe --pool open --budget 18
#   D-optimal design: 18/36 cells (12 params)
#   D-efficiency vs full square: ~0.99
#   inference saving:            50.0%
#   manifest -> results/doe_manifest.json
```

## Statistics

[`analysis/mixed_effects.py`](src/selfbias/analysis/mixed_effects.py):

- `--method bayes` — logistic **crossed-random-effects GLMM** via statsmodels'
  `BinomialBayesMixedGLM` (no R/lme4 needed). Reports `β_self`, posterior SD, and P(β_self > 0).
- `--method lpm` — fast linear prob. model with a prompt random intercept (the original notebook
  model) for a frequentist p-value.
- Per-evaluator, Holm-corrected breakdown (one-sided H₁: β_self > 0).

Grouping factors with <2 levels (e.g. a single dataset) are dropped automatically.

### Reproduce the original FOLIO finding

The original 4×4 FOLIO results are preserved in `results/*_on_*_full.jsonl`:

```bash
uv run selfbias analyze --legacy --method lpm
# β_self ≈ 0.033, one-sided p < 1e-3  (matches the original notebook)
```

## Repository layout

```
configs/         datasets.yaml · models.yaml · experiment.yaml   (the edit points)
src/selfbias/
  data/          ReasoningExample + the 10 dataset loaders
  models/        model registry + per-family sampling
  inference/     vllm_offline · vllm_online · openrouter (one Backend protocol)
  answers.py     answer extraction + correctness (ground truth)
  prompts.py     generation + evaluation templates
  generate.py    evaluate.py   aggregate.py
  doe.py         D-optimal generator×evaluator design
  analysis/      mixed_effects.py · plots.py
  cli.py  orchestrate.py
scripts/         run_all.py · run_cross_analysis.py
slurm/           generate.job · evaluate.job
tests/
```
