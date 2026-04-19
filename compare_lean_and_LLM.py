import json
import pandas as pd
from pathlib import Path

def get_matched_lean_data(lean_results_list, llm_result):
    prompt_used = llm_result.get("prompt_used", "")
    
    # lean_results_list is now a list of the actual JSON objects
    for result in lean_results_list:
        lean_premises = result.get("premises", "").strip()
        lean_hyp = result.get("hypothesis", "").strip()
        
        if lean_premises and lean_premises in prompt_used and lean_hyp in prompt_used:
            # Return the boolean 'is_valid' from the lean_verification dict
            return result.get("lean_verification", {}).get("is_valid", False)
            
    return None

def analyze_agreement(lean_file, validation_file):
    # 1. Load Lean Data as a LIST of dictionaries
    lean_results_list = []
    with open(lean_file, 'r') as f:
        for line in f:
            if not line.strip(): continue
            lean_results_list.append(json.loads(line))

    # 2. Load and Compare LLM Validation Data
    comparison_data = []
    
    with open(validation_file, 'r') as f:
        for line in f:
            if not line.strip(): continue
            llm_result = json.loads(line)
            
            pov = llm_result.get("pov")
            raw_output = llm_result.get("validation_output", "").lower()
            
            # Extract LLM judgment
            llm_judgment = "true" in raw_output
            
            # matched_lean_val will now correctly get the boolean result
            matched_lean_val = get_matched_lean_data(lean_results_list, llm_result)

            if matched_lean_val is not None:
                comparison_data.append({
                    "pov": pov,
                    "llm_judgment": llm_judgment,
                    "lean_truth": matched_lean_val,
                    "agrees": llm_judgment == matched_lean_val
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
    LEAN_FILE = "data/code_verification/lean_verified__Qwen3-32B.jsonl"
    VAL_FILE = "data/llm_verification/Qwen_Qwen3-32B.jsonl"
    
    analyze_agreement(LEAN_FILE, VAL_FILE)