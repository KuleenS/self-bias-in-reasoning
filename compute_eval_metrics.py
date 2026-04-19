import json
import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute accuracy/precision/recall/F1 for eval_error_detection outputs vs Lean ground truth."
    )
    p.add_argument("--eval-input", type=Path, required=True,
                   help="JSONL output from eval_error_detection.py.")
    p.add_argument("--verification-input", type=Path, required=True,
                   help="Lean verification JSONL (data/code_verification/lean_verified__*.jsonl).")
    p.add_argument("--output", type=Path, default=None,
                   help="Optional path to write JSON summary.")
    return p.parse_args()


def load_eval(path: Path) -> dict[int, dict]:
    result = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            result[rec["index"]] = rec
    return result


def load_ground_truth(path: Path) -> dict[int, bool]:
    result = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            lv = rec.get("lean_verification", {})
            is_valid = lv.get("is_valid")
            if is_valid is not None:
                result[rec["index"]] = is_valid
    return result


def compute_stats(records: list[dict]) -> dict:
    valid = [r for r in records if not r["parse_error"]]
    if not valid:
        return {}

    correct = sum(r["evaluator_judgment"] == r["ground_truth_is_correct"] for r in valid)
    accuracy = correct / len(valid)

    # Positive class = "trace is correct"
    tp = sum(r["evaluator_judgment"] and r["ground_truth_is_correct"] for r in valid)
    fp = sum(r["evaluator_judgment"] and not r["ground_truth_is_correct"] for r in valid)
    fn = sum(not r["evaluator_judgment"] and r["ground_truth_is_correct"] for r in valid)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    gt_correct = [r for r in valid if r["ground_truth_is_correct"]]
    gt_incorrect = [r for r in valid if not r["ground_truth_is_correct"]]
    acc_on_correct = (
        sum(r["evaluator_judgment"] for r in gt_correct) / len(gt_correct)
        if gt_correct else None
    )
    acc_on_incorrect = (
        sum(not r["evaluator_judgment"] for r in gt_incorrect) / len(gt_incorrect)
        if gt_incorrect else None
    )

    n_gt_correct = len(gt_correct)
    n_gt_incorrect = len(gt_incorrect)
    n_acc_on_correct = sum(r["evaluator_judgment"] for r in gt_correct)
    n_acc_on_incorrect = sum(not r["evaluator_judgment"] for r in gt_incorrect)

    return {
        "total": len(records),
        "parse_errors": len(records) - len(valid),
        "no_ground_truth": 0,  # filled by caller
        "evaluated": len(valid),
        "accuracy": accuracy,
        "accuracy_n": (correct, len(valid)),
        "precision": precision,
        "precision_n": (tp, tp + fp),
        "recall": recall,
        "recall_n": (tp, tp + fn),
        "f1": f1,
        "acc_on_correct_traces": acc_on_correct,
        "acc_on_correct_traces_n": (n_acc_on_correct, n_gt_correct) if gt_correct else None,
        "acc_on_incorrect_traces": acc_on_incorrect,
        "acc_on_incorrect_traces_n": (n_acc_on_incorrect, n_gt_incorrect) if gt_incorrect else None,
    }


def main():
    args = parse_args()

    eval_records = load_eval(args.eval_input)
    ground_truth = load_ground_truth(args.verification_input)

    joined = []
    no_gt = 0
    for idx, rec in eval_records.items():
        if idx not in ground_truth:
            no_gt += 1
            continue
        joined.append({**rec, "ground_truth_is_correct": ground_truth[idx]})

    stats = compute_stats(joined)
    stats["no_ground_truth"] = no_gt

    def fmt(key, stats):
        v = stats.get(key, 0)
        n = stats.get(f"{key}_n")
        frac = f" ({n[0]}/{n[1]})" if n else ""
        return f"{v:.3f}{frac}"

    print(f"\n=== Results ===")
    print(f"total:             {stats.get('total')}")
    print(f"no ground truth:   {stats.get('no_ground_truth')}")
    print(f"parse errors:      {stats.get('parse_errors')}")
    print(f"evaluated:         {stats.get('evaluated')}")
    print(f"accuracy:          {fmt('accuracy', stats)}")
    print(f"precision:         {fmt('precision', stats)}")
    print(f"recall:            {fmt('recall', stats)}")
    print(f"f1:                {stats.get('f1', 0):.3f}")
    if stats.get("acc_on_correct_traces") is not None:
        print(f"acc on correct traces:   {fmt('acc_on_correct_traces', stats)}")
    if stats.get("acc_on_incorrect_traces") is not None:
        print(f"acc on incorrect traces: {fmt('acc_on_incorrect_traces', stats)}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"\nsummary written to {args.output}")


if __name__ == "__main__":
    main()
