#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

from compute_eval_metrics import load_eval, load_ground_truth, compute_stats

MODEL_DATA_MAP: dict[str, tuple[Path, Path]] = {
    "Qwen/Qwen3-32B": (
        Path("data/reasoning_chains_qwen/reasoning_chains_outputs__Qwen3-32B.jsonl"),
        Path("data/code_verification/lean_verified__Qwen3-32B.jsonl"),
    ),
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": (
        Path("data/reasoning_chains_deepseek/reasoning_chain__DeepSeek-R1-Distill-Qwen-32B.jsonl"),
        Path("data/code_verification/lean_verified__Deepseek-R1.jsonl"),
    ),
    "allenai/Olmo-3.1-32B-Think": (
        Path("data/reasoning_chains_olmo_think/reasoning_chain__Olmo-3.1-32B-Think.jsonl"),
        Path("data/code_verification/lean_verified__Olmo-Think.jsonl"),
    ),
    "microsoft/Phi-4-reasoning": (
        Path("data/reasoning_chains_phi/reasoning_chain__phi4-reasoning.jsonl"),
        Path("data/code_verification/lean_verified__Phi4.jsonl"),
    ),
}


def process(eval_path: Path) -> dict:
    records = load_eval(eval_path)
    first = next(iter(records.values()))
    evaluated_model = first["evaluated_model"]
    if evaluated_model not in MODEL_DATA_MAP:
        raise ValueError(f"unknown evaluated_model '{evaluated_model}'; add it to MODEL_DATA_MAP")
    _, verification_path = MODEL_DATA_MAP[evaluated_model]
    ground_truth = load_ground_truth(verification_path)

    joined = []
    no_gt = 0
    for idx, rec in records.items():
        if idx not in ground_truth:
            no_gt += 1
            continue
        joined.append({**rec, "ground_truth_is_correct": ground_truth[idx]})

    stats = compute_stats(joined)
    stats["no_ground_truth"] = no_gt
    return stats


def main():
    p = argparse.ArgumentParser(
        description="Batch compute metrics for eval_error_detection outputs."
    )
    p.add_argument("--input-list", type=Path, required=True,
                   help="Text file with one eval JSONL path per line.")
    args = p.parse_args()

    paths = [
        Path(line.strip())
        for line in args.input_list.read_text().splitlines()
        if line.strip()
    ]

    combined: dict[str, dict] = {}

    for path in paths:
        out = path.parent / (path.stem + "_metrics.json")
        try:
            stats = process(path)
            out.write_text(json.dumps(stats, indent=2))
            print(f"ok: {path} -> {out}")

            parts = path.stem.split("_on_", 1)
            if len(parts) == 2:
                judge, judged = parts
                judged = judged.removesuffix("_full")
                combined.setdefault(judge, {})[judged] = stats
            else:
                print(f"warning: can't parse judge/judged from '{path.stem}', skipping from combined", file=sys.stderr)
        except Exception as e:
            print(f"error: {path}: {e}", file=sys.stderr)

    combined_out = args.input_list.parent / "combined_metrics.json"
    combined_out.write_text(json.dumps(combined, indent=2))
    print(f"combined: {combined_out}")


if __name__ == "__main__":
    main()
