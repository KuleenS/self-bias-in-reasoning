#!/bin/bash
#
# Submit the whole self-bias experiment (DOE-budget cells over the `open` pool) to Slurm with
# one command, wiring up dependencies so nothing runs before its inputs exist:
#
#   1. Generation, offline models (vllm_offline): ONE job array (slurm/generate_array.job),
#      one task per model, all running in parallel (independent 2-GPU tasks). Cancel the whole
#      phase with `scancel <phase1-jobid>`.
#   2. Generation + self/cross-evaluation, online models (vllm_online): one
#      slurm/serve_online.job per model (not an array — each needs a differently-shaped
#      --generators list). Only one online model can be served at a time (shared
#      VLLM_BASE_URL/.env), so these are chained sequentially via --dependency, and the first one
#      waits on all offline generation so it can evaluate against offline generators' chains.
#      Cross online-model dependencies (an online evaluator needing another online model's
#      chains) are resolved by ordering those jobs accordingly; a genuine cycle aborts the plan
#      (see the Python block below) rather than silently mis-sequencing runs.
#   3. Evaluation, offline evaluators on any generator: ONE job array
#      (slurm/evaluate_array.job), one task per (evaluator, generator) cell, submitted after ALL
#      generation (offline + online) completes. Cancel the whole phase with
#      `scancel <phase3-jobid>`.
#
# Each phase prints a single job id — `scancel <id>` cancels every task in that phase at once;
# `scancel <id>_<n>` cancels just task n. Phase 2's job ids are individual (one per online model).
#
# Usage:
#   scripts/run_cluster.sh            # dry run: print the plan and sbatch commands, submit nothing
#   scripts/run_cluster.sh --submit   # actually submit to Slurm
set -euo pipefail
cd "$(dirname "$0")/.."

SUBMIT=0
if [[ "${1:-}" == "--submit" ]]; then
  SUBMIT=1
elif [[ -n "${1:-}" ]]; then
  echo "usage: $0 [--submit]" >&2
  exit 1
fi

MANIFEST="results/doe_manifest.json"
if [[ ! -f "$MANIFEST" ]]; then
  echo "No DOE manifest at $MANIFEST; building one (selfbias doe --pool open)..."
  uv run selfbias doe --pool open
fi

PLAN=$(python3 - "$MANIFEST" <<'PY'
import json, sys

manifest = json.load(open(sys.argv[1]))
import yaml
models_cfg = yaml.safe_load(open("configs/models.yaml"))
backend = {m: (models_cfg.get(m) or {}).get("backend", "openrouter") for m in manifest["models"]}

generators = sorted({c["generator"] for c in manifest["cells"]})
offline_gens = [g for g in generators if backend.get(g) == "vllm_offline"]
online_gens = [g for g in generators if backend.get(g) == "vllm_online"]

# For each online evaluator: the other generators (any backend) it needs to judge, beyond itself.
online_extra: dict[str, set[str]] = {}
for c in manifest["cells"]:
    e, g = c["evaluator"], c["generator"]
    if backend.get(e) == "vllm_online" and g != e:
        online_extra.setdefault(e, set()).add(g)

# Topologically order online models so an online generator always runs before an online
# evaluator that needs its chains (Kahn's algorithm; abort on a cycle).
edges = {m: set() for m in online_gens}  # m -> set of online models that must run before m
for e, gens in online_extra.items():
    for g in gens:
        if g in edges:
            edges[e].add(g)

ordered, remaining = [], set(online_gens)
while remaining:
    ready = sorted(m for m in remaining if edges[m] <= set(ordered))
    if not ready:
        cyclic = ", ".join(sorted(remaining))
        print(f"ERROR: cyclic online-model dependency among [{cyclic}]; "
              "rerun `selfbias doe` with a different --seed or lower --budget", file=sys.stderr)
        sys.exit(1)
    ordered.extend(ready)
    remaining -= set(ready)

offline_eval_cells = sorted({(c["evaluator"], c["generator"]) for c in manifest["cells"]
                              if backend.get(c["evaluator"]) == "vllm_offline"})

tp = {m: (models_cfg.get(m) or {}).get("tensor_parallel", 1) for m in manifest["models"]}
offline_gens_1gpu = [g for g in offline_gens if tp.get(g, 1) == 1]
offline_gens_2gpu = [g for g in offline_gens if tp.get(g, 1) != 1]
# evaluate.job only loads the evaluator model, so the evaluator's tensor_parallel is what matters.
offline_eval_1gpu = [(e, g) for e, g in offline_eval_cells if tp.get(e, 1) == 1]
offline_eval_2gpu = [(e, g) for e, g in offline_eval_cells if tp.get(e, 1) != 1]

print("OFFLINE_GENS_1GPU\t" + "\t".join(offline_gens_1gpu))
print("OFFLINE_GENS_2GPU\t" + "\t".join(offline_gens_2gpu))
for m in ordered:
    extra = sorted(online_extra.get(m, []))
    print("ONLINE_GEN\t" + m + "\t" + "\t".join(extra))
for e, g in offline_eval_1gpu:
    print("OFFLINE_EVAL_1GPU\t" + e + "\t" + g)
for e, g in offline_eval_2gpu:
    print("OFFLINE_EVAL_2GPU\t" + e + "\t" + g)
PY
)

# Submits (or, in dry-run, just prints) a job and prints its job id (or a placeholder) on stdout.
submit_and_capture() {
  local out jid
  if [[ "$SUBMIT" == "1" ]]; then
    out=$("$@")               # sbatch --parsable prints just "<jobid>" (or "<jobid>;<cluster>")
    jid="${out%%;*}"
    jid="$(echo "$jid" | tr -d '[:space:]')"
  else
    printf '[dry-run] ' >&2
    printf '%q ' "$@" >&2
    printf '\n' >&2
    jid="job-id-placeholder-$RANDOM"
  fi
  echo "$jid"
}

# Array concurrency caps: at most this many tasks of a given array run at once, so we don't
# flood the group's Slurm queue (and don't ask for more concurrent GPU-slots than the shared
# "nvl" nodes, 4 H100s each and usually already partly used by others, actually have — this is
# what broke job 1984997: 8 concurrent 2-GPU tasks landed on one 4-GPU node and half of them got
# invalid/overlapping device bindings). Override via env, e.g. CONCURRENCY_1GPU=5 scripts/run_cluster.sh --submit
CONCURRENCY_1GPU="${CONCURRENCY_1GPU:-3}"
CONCURRENCY_2GPU="${CONCURRENCY_2GPU:-3}"

RUN_DIR="results/cluster_run/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

OFFLINE_GENS_1GPU=()
OFFLINE_GENS_2GPU=()
ONLINE_GENS=()
declare -A ONLINE_EXTRA
OFFLINE_EVAL_1GPU=()
OFFLINE_EVAL_2GPU=()

while IFS=$'\t' read -r -a fields; do
  kind="${fields[0]}"
  rest=("${fields[@]:1}")
  case "$kind" in
    OFFLINE_GENS_1GPU)
      OFFLINE_GENS_1GPU=("${rest[@]}")
      ;;
    OFFLINE_GENS_2GPU)
      OFFLINE_GENS_2GPU=("${rest[@]}")
      ;;
    ONLINE_GEN)
      m="${rest[0]}"
      ONLINE_GENS+=("$m")
      ONLINE_EXTRA["$m"]="${rest[*]:1}"
      ;;
    OFFLINE_EVAL_1GPU)
      OFFLINE_EVAL_1GPU+=("${rest[0]}|${rest[1]}")
      ;;
    OFFLINE_EVAL_2GPU)
      OFFLINE_EVAL_2GPU+=("${rest[0]}|${rest[1]}")
      ;;
  esac
done <<< "$PLAN"

# Submits one generate array (given a model list, GPU count and concurrency cap) and prints its
# job id, or "" if the list is empty. `sbatch --gres=...` on the command line overrides the
# script's own #SBATCH --gres, so one generate_array.job file serves both GPU tiers.
submit_generate_array() {
  local plan_path=$1 gpus=$2 concurrency=$3; shift 3
  local models=("$@")
  [[ ${#models[@]} -eq 0 ]] && return 0
  printf '%s\n' "${models[@]}" > "$plan_path"
  local n=${#models[@]}
  local jid
  jid=$(submit_and_capture sbatch --parsable --gres="gpu:$gpus" \
        --array="0-$((n - 1))%${concurrency}" slurm/generate_array.job "$plan_path")
  echo "  [$gpus GPU] ${models[*]}" >&2
  echo "  -> array job $jid (0-$((n - 1))%${concurrency} tasks; plan: $plan_path)" >&2
  echo "  cancel whole phase: scancel $jid" >&2
  echo "$jid"
}

echo "=== Phase 1: offline generation (${#OFFLINE_GENS_1GPU[@]} x 1-GPU, ${#OFFLINE_GENS_2GPU[@]} x 2-GPU, job arrays) ==="
jid_1gpu=$(submit_generate_array "$RUN_DIR/offline_generate_1gpu.txt" 1 "$CONCURRENCY_1GPU" "${OFFLINE_GENS_1GPU[@]}")
jid_2gpu=$(submit_generate_array "$RUN_DIR/offline_generate_2gpu.txt" 2 "$CONCURRENCY_2GPU" "${OFFLINE_GENS_2GPU[@]}")
OFFLINE_DEP=""
dep_ids=()
[[ -n "$jid_1gpu" ]] && dep_ids+=("$jid_1gpu")
[[ -n "$jid_2gpu" ]] && dep_ids+=("$jid_2gpu")
if [[ ${#dep_ids[@]} -gt 0 ]]; then
  OFFLINE_DEP=$(IFS=:; echo "${dep_ids[*]}")
fi

echo "=== Phase 2: online models — generate + self/cross-evaluate (${#ONLINE_GENS[@]} models, sequential) ==="
PREV_ONLINE_JOBID=""
LAST_ONLINE_JOBID=""
for m in "${ONLINE_GENS[@]}"; do
  dep_parts=()
  [[ -n "$OFFLINE_DEP" ]] && dep_parts+=("afterok:$OFFLINE_DEP")
  [[ -n "$PREV_ONLINE_JOBID" ]] && dep_parts+=("afterok:$PREV_ONLINE_JOBID")
  dep_arg=""
  if [[ ${#dep_parts[@]} -gt 0 ]]; then
    dep_arg="--dependency=$(IFS=,; echo "${dep_parts[*]}")"
  fi
  extra="${ONLINE_EXTRA[$m]:-}"
  if [[ -n "$dep_arg" ]]; then
    jid=$(submit_and_capture sbatch --parsable "$dep_arg" slurm/serve_online.job "$m" --generators $extra)
  else
    jid=$(submit_and_capture sbatch --parsable slurm/serve_online.job "$m" --generators $extra)
  fi
  echo "  $m (extra generators: ${extra:-none}) -> job $jid"
  PREV_ONLINE_JOBID="$jid"
  LAST_ONLINE_JOBID="$jid"
done

eval_dep_parts=()
[[ -n "$OFFLINE_DEP" ]] && eval_dep_parts+=("afterok:$OFFLINE_DEP")
[[ -n "$LAST_ONLINE_JOBID" ]] && eval_dep_parts+=("afterok:$LAST_ONLINE_JOBID")
eval_dep_arg=()
if [[ ${#eval_dep_parts[@]} -gt 0 ]]; then
  eval_dep_arg=(--dependency="$(IFS=,; echo "${eval_dep_parts[*]}")")
fi

# Submits one evaluate array (given cells, GPU count and concurrency cap) and prints its job id,
# or "" if the list is empty. See submit_generate_array above for why this is split by GPU tier.
submit_evaluate_array() {
  local plan_path=$1 gpus=$2 concurrency=$3; shift 3
  local cells=("$@")
  [[ ${#cells[@]} -eq 0 ]] && return 0
  : > "$plan_path"
  local cell evaluator generator
  for cell in "${cells[@]}"; do
    IFS='|' read -r evaluator generator <<< "$cell"
    printf '%s\t%s\n' "$evaluator" "$generator" >> "$plan_path"
  done
  local n=${#cells[@]}
  local jid
  jid=$(submit_and_capture sbatch --parsable "${eval_dep_arg[@]}" --gres="gpu:$gpus" \
        --array="0-$((n - 1))%${concurrency}" slurm/evaluate_array.job "$plan_path")
  echo "  [$gpus GPU] ${cells[*]}" >&2
  echo "  -> array job $jid (0-$((n - 1))%${concurrency} tasks; plan: $plan_path)" >&2
  echo "  cancel whole phase: scancel $jid" >&2
  echo "$jid"
}

echo "=== Phase 3: offline evaluation (${#OFFLINE_EVAL_1GPU[@]} x 1-GPU, ${#OFFLINE_EVAL_2GPU[@]} x 2-GPU, job arrays) ==="
submit_evaluate_array "$RUN_DIR/offline_evaluate_1gpu.tsv" 1 "$CONCURRENCY_1GPU" "${OFFLINE_EVAL_1GPU[@]}" > /dev/null
submit_evaluate_array "$RUN_DIR/offline_evaluate_2gpu.tsv" 2 "$CONCURRENCY_2GPU" "${OFFLINE_EVAL_2GPU[@]}" > /dev/null

if [[ "$SUBMIT" == "0" ]]; then
  echo
  echo "Dry run only — nothing was submitted. Re-run with --submit to actually sbatch these jobs."
fi
