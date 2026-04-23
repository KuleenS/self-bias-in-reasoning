import json
import os
import pandas as pd
from pathlib import Path

LEAN_FILE = "data/code_verification/lean_verified__Olmo-Think.jsonl"
VAL_FILE = "data/llm_verification/Olmo-3.1-32B-Think_eval_reasoning_output.jsonl"
    
def get_matched_lean_data(lean_lookup, llm_result):
    idx = llm_result.get("index")
    if idx is not None and idx in lean_lookup:
        return lean_lookup[idx]
    return None

def analyze_agreement(lean_file, validation_file):
    # 1. Load in LEAN data
    lean_lookup = {}
    with open(lean_file, 'r') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            idx = data.get("index")
            if idx is not None:
                lean_lookup[idx] = data

    # 2. Load and Compare LLM Validation Data
    comparison_data = []
    
    with open(validation_file, 'r') as f:
        for line in f:
            if not line.strip(): continue
            llm_result = json.loads(line)
            
            pov = llm_result.get("POV")
            llm_judgment = llm_result.get("evaluator_judgment", "")
            
            matched_lean_obj = get_matched_lean_data(lean_lookup, llm_result)

            if matched_lean_obj is not None:
                lean_truth = matched_lean_obj.get("lean_verification", {}).get("is_valid")
                comparison_data.append({
                    "pov": pov,
                    "llm_judgment": llm_judgment,
                    "lean_truth": lean_truth,
                    "agrees": llm_judgment == lean_truth
                })

    df = pd.DataFrame(comparison_data)

    if df.empty:
        print("No matches found between Lean results and LLM results.")
        return

    # 3. Generate Metrics
    print("--- Agreement Analysis (LLM vs Lean Formal Proof) ---")
    
    accuracy_by_pov = df.groupby('pov')['agrees'].mean() * 100
    print("\nAgreement Rate by POV (% matching Lean):")
    print(accuracy_by_pov)

    print("\nRaw Count of 'Valid' labels by POV:")
    print(df.groupby('pov')['llm_judgment'].value_counts().unstack().fillna(0))

    print("\nOverall Agreement:")
    print(df['agrees'].value_counts())
    print("\nOverall Agreement %:")
    print(df['agrees'].mean() * 100)

    print("\nOverall Confusion Matrix:")
    print(pd.crosstab(df['lean_truth'], df['llm_judgment'], rownames=['Lean Truth'], colnames=['LLM Judgment']))

if __name__ == "__main__":
    analyze_agreement(LEAN_FILE, VAL_FILE)