# Self-Bias in Reasoning

Research project investigating self-bias in LLM reasoning chains. The core question: when LLMs evaluate reasoning chains, does the validation outcome differ based on the attributed perspective (she/he/they/LLM)? Uses the [FOLIO](https://github.com/Yale-LILY/FOLIO) formal logic dataset.

## Setup

```bash
conda env create -f environment.yaml -n self-bias
conda activate self-bias
```

`OPENAI_API_KEY` must be set for `convert_reasoning_to_lean.py`.

## Pipeline

### 1. Generate reasoning chains

```bash
python get_reasoning_chains.py \
    --model Qwen/Qwen3-32B \
    --output data/reasoning_chains_qwen/my_output.jsonl
```

Runs vLLM batch inference over the FOLIO dataset. Output fields: `prompt`, `generated_text`. The output filename is automatically suffixed with the model slug.

Supported model families: Qwen3, DeepSeek, OLMo, Gemma (model-specific sampling params are applied automatically).

### 2. Evaluate chains for perspective bias

```bash
python eval_reasoning_chains.py \
    --model Qwen/Qwen3-32B \
    --input data/reasoning_chains_qwen/reasoning_chains_outputs__Qwen3-32B.jsonl \
    --output validation_results.jsonl
```

Evaluates each reasoning chain from 4 perspectives: `she`, `he`, `they`, `LLM`. Output fields: `pov`, `validation_output`, `prompt_used`.

### 3. Convert reasoning to Lean code

```bash
python convert_reasoning_to_lean.py \
    --folio-input data/FOLIO/folio_train.jsonl \
    --reasoning-input data/reasoning_chains_qwen/reasoning_chains_outputs__Qwen3-32B.jsonl \
    --output data/lean_code/lean_code_outputs__Qwen3-32B.jsonl \
    --model gpt-4o \
    --temperature 0.0
```

Uses OpenAI Batch API to formalize NL reasoning into Lean 4 proofs. Supports `--resume` (default: true) to skip already-processed indices.

### 4. Verify Lean code

```bash
python verify_lean_code.py \
    --input data/lean_code/lean_code_outputs__Qwen3-32B.jsonl \
    --output data/code_verification/lean_verified__Qwen3-32B.jsonl \
    --lean-cmd auto \
    --timeout-seconds 30
```

Runs the Lean compiler on each generated proof. The `lean_verification` output field contains `is_valid`, `error_type`, `error_message`, and `exit_code`.

### 5. Evaluate error detection

```bash
python eval_error_detection.py \
    --evaluator-model allenai/OLMo-3.1-32B-Think \
    --evaluated-model Qwen/Qwen3-32B \
    --enable-thinking \
    --output results/olmo_on_qwen.jsonl
```

Uses one model (the evaluator) to judge whether reasoning chains produced by another model (the evaluated model) are logically correct. Ground truth comes from Lean verification. Writes results incrementally in batches and supports resuming interrupted runs.

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--evaluator-model` | `Qwen/Qwen3-32B` | Model doing the judging |
| `--evaluated-model` | `Qwen/Qwen3-32B` | Model whose traces are being judged |
| `--enable-thinking` | off | Enable chain-of-thought tokens (Qwen3/Gemma) |
| `--batch-size` | 50 | Prompts per vLLM call; results flushed after each batch |
| `--num-problems` | all | Limit number of traces evaluated |
| `--max-reasoning-chars` | 20000 | Truncate long reasoning chains before inserting into prompt |
| `--reasoning-input` | from map | Override reasoning chains path |
| `--verification-input` | from map | Override Lean verification path |

Output fields per record: `index`, `evaluator_model`, `evaluated_model`, `premises`, `hypothesis`, `generated_text`, `raw_output_text`, `evaluator_thinking`, `evaluator_judgment` (bool), `parse_error` (bool).

The `--evaluated-model` / `--reasoning-input` / `--verification-input` defaults are defined in `MODEL_DATA_MAP` at the top of the script.

### 6. Compute evaluation metrics

```bash
python compute_eval_metrics.py \
    --eval-input results/olmo_on_qwen.jsonl \
    --verification-input data/code_verification/lean_verified__Qwen3-32B.jsonl \
    --output results/olmo_on_qwen_metrics.json
```

Computes accuracy, precision, recall, and F1 against Lean-verified ground truth. Positive class is "reasoning trace is correct" (`evaluator_judgment == true`).

Also reports:
- `acc_on_correct_traces`: accuracy on traces Lean verified as correct
- `acc_on_incorrect_traces`: accuracy on traces Lean verified as incorrect (i.e., error detection rate)

`--output` is optional; omitting it prints results to stdout only.

## Data

- `data/FOLIO/` — Source dataset (train: 1001, validation: 203, test: 226 examples)
- `data/reasoning_chains_{model}/` — Per-model chain outputs
- `data/lean_code/` — Formalized Lean proofs
- `data/code_verification/` — Lean verification results
- `results/` — `eval_error_detection` outputs and metric summaries
