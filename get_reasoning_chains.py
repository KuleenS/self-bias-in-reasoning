from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from folio_loader import FolioDataset
import json
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
        "--output",
        type=Path,
        default=Path("reasoning_chains_outputs.jsonl"),
        help="Path to write JSONL outputs.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Configurae the sampling parameters (for thinking mode)
    sampling_params = SamplingParams(
        temperature=0.6, top_p=0.95, top_k=20, max_tokens=32768
    )

    # Initialize the vLLM engine
    llm = LLM(model=args.model)

    folio_dataset = FolioDataset()
    total_prompts = []

    prompt_template = """You are an expert evaluator for the FOLIO logical reasoning benchmark.

Task:
You will be given a FOLIO example consisting of:
- A set of premises/facts (may include rules, quantifiers, and relations)
- A hypothesis/claim to evaluate

Instructions:
1) Use ONLY the information in the premises. Do not assume any additional facts or real-world knowledge.
2) Determine the correct label for the hypothesis:
    - ENTAILED: the premises logically imply the hypothesis.
    - CONTRADICTED: the premises logically imply the hypothesis is false.
    - UNKNOWN: the hypothesis cannot be proven true or false from the premises.
3) Provide a clear chain-of-reasoning that references specific premises and intermediate conclusions.
4) If there is ambiguity or missing information, choose UNKNOWN and explain why.
5) Output format (exactly):
Label: <ENTAILED|CONTRADICTED|UNKNOWN>
Reasoning:
- <step 1>
- <step 2>
...
Conclusion: <one sentence restating the label>.

Input:
Premises:
{premises}

Hypothesis:
{hypothesis}
"""

    for example in folio_dataset:
        prompt = prompt_template.format(
            premises=example["premises"], hypothesis=example["conclusion"]
        )

        messages = [{"role": "user", "content": prompt}]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,  # Set to False to strictly disable thinking
        )

        total_prompts.append(text)

    # Generate outputs
    outputs = llm.generate(total_prompts, sampling_params)

    # Save the outputs.
    # Make output path dependent on the output argument and model name.
    model_slug = args.model.rstrip("/").split("/")[-1]
    out_path = args.output
    out_path = out_path.with_name(f"{out_path.stem}__{model_slug}{out_path.suffix}")
    
    with out_path.open("w", encoding="utf-8") as f:
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text

            record = {
                "prompt": prompt,
                "generated_text": generated_text,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
