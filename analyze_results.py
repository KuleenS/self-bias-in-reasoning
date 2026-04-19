import json
import pandas as pd
from pathlib import Path
import argparse

def parse_args():
    p = argparse.ArgumentParser(description="Analyze results .")
    p.add_argument(
        "--input",
        type=Path,
        default=Path("data/llm_verification/Qwen_Qwen3-32B.jsonl"),
        help="Path for JSONL input.",
    )
    return p.parse_args()

def analyze_logic_results(file_path):
    data = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            
            entry = json.loads(line)
            pov = entry['pov']
            raw_output = entry['validation_output'].lower()
            
            is_valid = None
            if 'true' in raw_output:
                is_valid = True
            elif 'false' in raw_output:
                is_valid = False
            
            reasoning_content = entry['prompt_used'].split('Student Reasoning:')[1]
            
            data.append({
                'pov': pov,
                'is_valid': is_valid,
                'problem_id': hash(reasoning_content) 
            })

    df = pd.DataFrame(data)

    # 1. Metrics for each pronoun
    print("### Accuracy by Pronoun/POV ###")
    summary = df.groupby('pov')['is_valid'].value_counts(normalize=True).unstack().fillna(0)
    print(summary)
    print("\n")

    # 2. Overall average
    print("### Overall Distribution ###")
    overall = df['is_valid'].value_counts(normalize=True)
    print(overall)
    print("\n")

    # 3. Pull out examples where decision changed based on pronoun
    print("### Decision Mismatches (Bias Detection) ###")
    # Group by the specific reasoning block and see if there is more than 1 unique answer
    mismatches = df.groupby('problem_id')['is_valid'].nunique()
    mismatched_ids = mismatches[mismatches > 1].index

    if len(mismatched_ids) == 0:
        print("No mismatches found. The model was consistent across all pronouns.")
    else:
        for p_id in mismatched_ids:
            subset = df[df['problem_id'] == p_id]
            print(f"Problem ID: {p_id}")
            for _, row in subset.iterrows():
                print(f"  - POV: {row['pov']} | Result: {row['is_valid']}")
            print("-" * 30)

if __name__ == "__main__":
    args = parse_args()
    analyze_logic_results(args.input)