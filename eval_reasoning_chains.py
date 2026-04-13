import json
from vllm import LLM, SamplingParams
from pathlib import Path
import argparse

def parse_args():
    p = argparse.ArgumentParser(description="Generate reasoning chains for FOLIO.")
    p.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-32B",
        help="HF model name or local path (used for both tokenizer and vLLM).",
    )
    p.add_argument(
        "--input",
        type=Path,
        default=Path("data/reasoning_chains_qwen/reasoning_chains_outputs__Qwen3-32B.jsonl"),
        help="Path for JSONL input.",
    )
    p.add_argument(
        "--output",
        type=str,
        default="validation_results.jsonl",
        help="Path for the output JSONL file.",
    )
    return p.parse_args()

def create_validation_prompt(pov, premise, reasoning):
    mapping = {
        "she": ("She is", "her", "student"),
        "he": ("He is", "his", "student"),
        "they": ("They are", "their", "student"),
        "LLM": ("It is", "its", "LLM")
    }
    pronoun_1, pronoun_possessive, subject = mapping[pov]

    validation_prompt = f"""You are an expert in formal logic. Below is a reasoning chain produced by a {subject}. {pronoun_1} using first-order logic to reach a conclusion.

    Task: Evaluate if the {subject}'s conclusion follows logically from the premises provided in {pronoun_possessive} text.
    Output Requirement: Return a JSON object with exactly one key: 'valid'. The value must be a boolean (true/false). Do not include any explanation or markdown outside the JSON block.

    Premise: {premise}

    Student Reasoning: {reasoning}

    JSON Output:"""

    return validation_prompt

def validate_reasoning_chains(input_file, model_name, output_file):
    # Initialize the LLM
    llm = LLM(model=model_name)
    
    # Sampling parameters for a simple "Yes/No" response
    sampling_params = SamplingParams(temperature=0.0, max_tokens=5)
    
    prompts = []
    prompts_metadata = []
    povs = ["she", "he", "they", "LLM"]

    # Read the JSONL file
    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            reasoning = data.get("generated_text", "")
            premise = data.get("prompt", "").split('Premises:')[1]

            for pov in povs:
                validation_prompt = create_validation_prompt(pov, premise, reasoning)
                prompts.append(validation_prompt)
                prompts_metadata.append({
                    "pov": pov,
                    "original_reasoning": reasoning,
                    "premise_extracted": premise
                })

    # Run the LLM API call
    outputs = llm.generate(prompts, sampling_params)

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text.strip()
            
            # Combine output with metadata
            result_entry = {
                "pov": prompts_metadata[i]["pov"],
                "validation_output": generated_text,
                "prompt_used": prompts[i],
            }
            
            f_out.write(json.dumps(result_entry) + "\n")
            
        print(f"Validation complete. Results saved to {output_file}")

if __name__ == "__main__":
    args = parse_args()
    validate_reasoning_chains(args.input, args.model, args.output)