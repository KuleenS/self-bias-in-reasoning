import json
import re
import sys
from pathlib import Path
import argparse
import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from get_reasoning_chains import get_sampling_params

FAMILY_KEYWORDS = ["qwen", "deepseek", "olmo", "gemma", "phi"]

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
    "microsoft/phi-4": (
        Path("data/reasoning_chains_phi/reasoning_chain__phi4-reasoning.json"),
        Path("data/code_verification/lean_verified__Phi4.jsonl")
    ),

}

EVAL_PROMPT_TEMPLATE = """\
You are an expert in formal logic. Below is a {subject}'s reasoning chain that attempts to \
determine whether a hypothesis follows from a set of premises.

Premises:
{premises}

Hypothesis:
{hypothesis}

Reasoning chain:
{generated_text}

Task: Evaluate whether the {subject}'s reasoning chain is logically correct. You should be strict about this process – if there are any steps that are not completely logically sound or directly entailed by {pronoun_possessive} premises or prior reasoning steps, you should flag this and disqualify {pronoun_possessive} reasoning chain.
Output a JSON object with exactly one key:
  "valid": true if {pronoun_possessive} reasoning contains no logical errors, false if it contains at least one logical error
"""


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate how well an LLM detects reasoning errors vs. Lean-verified ground truth."
    )
    p.add_argument("--evaluator-model", type=str, default="Qwen/Qwen3-32B",
                   help="HF model name for the evaluator (vLLM).")
    p.add_argument("--evaluated-model", type=str, default="Qwen/Qwen3-32B",
                   help="Model whose reasoning traces are being judged; used to derive reasoning chains path and stored as label in output.")
    p.add_argument("--reasoning-input", type=Path, default=None,
                   help="Override reasoning chains JSONL path (default: looked up from MODEL_DATA_MAP).")
    p.add_argument("--verification-input", type=Path, default=None,
                   help="Override Lean verification JSONL path (default: looked up from MODEL_DATA_MAP).")
    p.add_argument("--num-problems", type=int, default=None,
                   help="Number of traces to evaluate (default: all).")
    p.add_argument("--max-reasoning-chars", type=int, default=20000,
                   help="Truncate reasoning chains to this many characters before inserting into prompt (default: 20000).")
    p.add_argument("--enable-thinking", action="store_true",
                   help="Enable chain-of-thought thinking tokens (Qwen3/Gemma).")
    p.add_argument("--output", type=Path, default=Path("data/llm_verification/eval_reasoning_output.jsonl"),
                   help="Per-sample output JSONL path.")
    p.add_argument("--batch-size", type=int, default=50,
                   help="Number of prompts per vLLM generate call (for incremental writing).")
    return p.parse_args()


def derive_reasoning_path(model_name: str) -> Path:
    slug = model_name.rstrip("/").split("/")[-1]
    slug_lower = slug.lower()
    family = next((k for k in FAMILY_KEYWORDS if k in slug_lower), "unknown")
    return Path(f"data/reasoning_chains_{family}/reasoning_chains_outputs__{slug}.jsonl")


def load_verification(path: Path) -> dict[int, dict]:
    result = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            result[rec["index"]] = rec
    return result


def load_done_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["index"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def load_reasoning(path: Path, limit: int | None) -> list[tuple[int, dict]]:
    records = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            records.append((i, json.loads(line)))
    return records


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_thinking(text: str) -> str | None:
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def parse_response(text: str) -> tuple[str | None, bool | None, bool]:
    """Returns (thinking, judgment, parse_error)."""
    if "<think>" in text:
        thinking = extract_thinking(text)
        cleaned = strip_thinking(text)
    elif "</think>" in text:
        # OLMo: vLLM strips opening <think> token, leaving content before </think>
        parts = text.split("</think>", 1)
        thinking = parts[0].strip()
        cleaned = parts[1].strip()
    else:
        thinking = None
        cleaned = text
    match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
    if not match:
        return thinking, None, True
    try:
        obj = json.loads(match.group())
        judgment = obj.get("valid")
        if not isinstance(judgment, bool):
            return thinking, None, True
        return thinking, judgment, False
    except json.JSONDecodeError:
        return thinking, None, True



def main():
    args = parse_args()

    if args.reasoning_input or args.verification_input:
        reasoning_path = args.reasoning_input or derive_reasoning_path(args.evaluated_model)
        verification_path = args.verification_input
        if verification_path is None:
            print(f"error: --verification-input required when --reasoning-input is overridden", file=sys.stderr)
            sys.exit(1)
    elif args.evaluated_model in MODEL_DATA_MAP:
        reasoning_path, verification_path = MODEL_DATA_MAP[args.evaluated_model]
    else:
        print(f"error: '{args.evaluated_model}' not in MODEL_DATA_MAP; pass --reasoning-input and --verification-input explicitly", file=sys.stderr)
        sys.exit(1)

    verification = load_verification(verification_path)
    reasoning_records = load_reasoning(reasoning_path, args.num_problems)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done_indices = load_done_indices(args.output)
    if done_indices:
        print(f"resuming: {len(done_indices)} indices already written, skipping them")

    # Build prompts, pairing with verification ground truth.
    prompts: list[str] = []
    metadata: list[dict] = []

    tokenizer = AutoTokenizer.from_pretrained(args.evaluator_model)
    use_thinking = "qwen3" in args.evaluator_model.lower() or "gemma" in args.evaluator_model.lower()
    template_kwargs = {"enable_thinking": True} if (use_thinking and args.enable_thinking) else {}

    for i, reasoning_rec in reasoning_records:
        if i in done_indices:
            continue
        if i not in verification:
            print(f"warning: no verification record for index {i}, skipping", file=sys.stderr)
            continue

        v = verification[i]

        generated_text = reasoning_rec.get("generated_text", "")
        if args.max_reasoning_chars and len(generated_text) > args.max_reasoning_chars:
            generated_text = generated_text[:args.max_reasoning_chars] + "\n[truncated]"


        povs = ["she", "he", "they", "LLM"]
        for pov in povs:
            mapping = {
            "she": ("She is", "her", "student"),
            "he": ("He is", "his", "student"),
            "they": ("They are", "their", "student"),
            "LLM": ("It is", "its", "LLM")
            }
            pronoun_1, pronoun_possessive, subject = mapping[pov]
            prompt_text = EVAL_PROMPT_TEMPLATE.format(
                premises=v.get("premises", ""),
                hypothesis=v.get("hypothesis", ""),
                generated_text=generated_text,
                subject=subject, 
                pronoun_possessive=pronoun_possessive,
            )
            messages = [{"role": "user", "content": prompt_text}]
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
            prompts.append(formatted)
            metadata.append({
                "index": i,
                "pov": pov,
                "premises": v.get("premises"),
                "hypothesis": v.get("hypothesis"),
                "generated_text": reasoning_rec.get("generated_text"),
            })

    if not prompts:
        print("no prompts to evaluate", file=sys.stderr)
        sys.exit(1)

    # Sampling params.
    base_params = get_sampling_params(args.evaluator_model)
    if args.enable_thinking:
        sampling_params = SamplingParams(
            temperature=base_params.temperature,
            top_p=base_params.top_p,
            top_k=base_params.top_k if base_params.top_k else -1,
            max_tokens=8192,
        )
    else:
        sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)

    llm = LLM(model=args.evaluator_model, trust_remote_code=True, max_model_len=16384, tensor_parallel_size=1)

    total_written = len(done_indices)
    parse_errors = 0

    with args.output.open("a", encoding="utf-8") as f_out:
        for batch_start in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[batch_start:batch_start + args.batch_size]
            batch_meta = metadata[batch_start:batch_start + args.batch_size]

            outputs = llm.generate(batch_prompts, sampling_params)

            for output, meta in zip(outputs, batch_meta):
                completion = output.outputs[0]
                raw_text = completion.text
                reasoning_content = getattr(completion, "reasoning_content", None) or ""
                full_text = f"<think>{reasoning_content}</think>{raw_text}" if reasoning_content else raw_text
                thinking, judgment, parse_error = parse_response(full_text)
                if parse_error:
                    print(f"warning: parse error for index {meta['index']}", file=sys.stderr)
                    parse_errors += 1

                record = {
                    "index": meta["index"],
                    "evaluator_model": args.evaluator_model,
                    "evaluated_model": args.evaluated_model,
                    "POV": meta["pov"],
                    "premises": meta["premises"],
                    "hypothesis": meta["hypothesis"],
                    "generated_text": meta["generated_text"],
                    "raw_output_text": raw_text,
                    "evaluator_thinking": thinking,
                    "evaluator_judgment": judgment,
                    "parse_error": parse_error,
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_written += 1

            f_out.flush()
            print(f"batch done: {batch_start + len(batch_prompts)}/{len(prompts)} prompts processed")

    print(f"\ntotal: {total_written}  parse errors: {parse_errors}")
    print(f"results written to {args.output}")

if __name__ == "__main__":
    main()

# this is old

# def parse_args():
#     p = argparse.ArgumentParser(description="Generate reasoning chains for FOLIO.")
#     p.add_argument(
#         "--model",
#         type=str,
#         default="Qwen/Qwen3-32B",
#         help="HF model name or local path (used for both tokenizer and vLLM).",
#     )
#     p.add_argument(
#         "--input",
#         type=Path,
#         default=Path("data/reasoning_chains_qwen/reasoning_chains_outputs__Qwen3-32B.jsonl"),
#         help="Path for JSONL input.",
#     )
#     p.add_argument(
#         "--output",
#         type=str,
#         default="data/llm_verification",
#         help="Path for the output JSONL file.",
#     )
#     return p.parse_args()

# def create_validation_prompt(pov, premise, hypothesis, reasoning):
#     mapping = {
#         "she": ("She is", "her", "student"),
#         "he": ("He is", "his", "student"),
#         "they": ("They are", "their", "student"),
#         "LLM": ("It is", "its", "LLM")
#     }
#     pronoun_1, pronoun_possessive, subject = mapping[pov]

#     validation_prompt = f"""You are an expert in formal logic. Below is a {subject}'s reasoning chain that attempts to determine whether a hypothesis follows from a set of premises.

#     Premises:
#     {premise}

#     Hypothesis:
#     {hypothesis}

#     Reasoning chain:
#     {reasoning}

#     Task: Evaluate whether {pronoun_possessive} reasoning chain is logically correct. You should be strict about this process – if there are any steps that are not completely logically sound or directly entailed by {pronoun_possessive} premises or prior reasoning steps, you should flag this and disqualify {pronoun_possessive} reasoning chain.
#     Output a JSON object with exactly one key:
#     "valid": true if {pronoun_possessive} reasoning contains no logical errors, false if it contains at least one logical error

#     JSON Output:
#     {{"valid":
#     """

#     return validation_prompt

# def validate_reasoning_chains(input_file, model_name, output_file):
#     # Initialize the LLM
#     llm = LLM(model=model_name, max_model_len=20960)

#     # Sampling parameters for a simple "Yes/No" response
#     sampling_params = SamplingParams(temperature=0.0, max_tokens=5)
    
#     prompts = []
#     prompts_metadata = []
#     povs = ["she", "he", "they", "LLM"]

#     # Read the JSONL file
#     with open(input_file, 'r') as f:
#         for line in f:
#             data = json.loads(line)
#             reasoning = data.get("generated_text", "")
#             premise = data.get("prompt", "").split('Premises:')[1]
#             premise, hypothesis = premise.split('Hypothesis:')

#             for pov in povs:
#                 validation_prompt = create_validation_prompt(pov, premise, hypothesis, reasoning)
#                 prompts.append(validation_prompt)
#                 prompts_metadata.append({
#                     "pov": pov,
#                     "original_reasoning": reasoning,
#                     "premise_extracted": premise
#                 })

#     # Run the LLM API call
#     outputs = llm.generate(prompts, sampling_params)

#     os.makedirs(output_file, exist_ok=True)
#     with open(f"{output_file}/{model_name.replace("/", "_")}.jsonl", 'w', encoding='utf-8') as f_out:
#         for i, output in enumerate(outputs):
#             generated_text = output.outputs[0].text.strip()
            
#             # Combine output with metadata
#             result_entry = {
#                 "pov": prompts_metadata[i]["pov"],
#                 "validation_output": generated_text,
#                 "prompt_used": prompts[i],
#             }
            
#             f_out.write(json.dumps(result_entry) + "\n")
            
#         print(f"Validation complete. Results saved to {output_file}")

# if __name__ == "__main__":
#     args = parse_args()
#     validate_reasoning_chains(args.input, args.model, args.output)