#!/usr/bin/env python3
"""Batch-convert natural-language reasoning chains into Lean code using GPT-5.4.

Usage:
	python convert_reasoning_to_lean.py \
		--input reasoning_chains.jsonl \
		--output reasoning_chains_lean.jsonl

Environment:
	OPENAI_API_KEY must be set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable

try:
	from openai import OpenAI
except ImportError as exc:  # pragma: no cover
	raise SystemExit(
		"Missing dependency: openai. Install with `pip install openai`."
	) from exc


DEFAULT_SYSTEM_PROMPT = """Convert the provided reasoning chain into Lean 4 proof-assistant code.

Requirements:
1) Output Lean 4 code only (no markdown fences, no prose).
2) Preserve the logical structure from the chain.
3) Use clear declarations/theorems and include assumptions explicitly.
"""


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Batch prompt GPT-5.4 to convert reasoning chains into Lean code."
	)
	parser.add_argument(
		"--input",
		type=Path,
		default=Path("data/reasoning_chains.jsonl"),
		help="Input JSONL file containing a 'chain' field.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("data/reasoning_chains_lean.jsonl"),
		help="Output JSONL file for generated Lean code.",
	)
	parser.add_argument(
		"--model",
		default="gpt-5.4",
		help="Model to use for conversion (default: gpt-5.4).",
	)
	parser.add_argument(
		"--temperature",
		type=float,
		default=0.0,
		help="Sampling temperature (default: 0.0).",
	)
	parser.add_argument(
		"--max-retries",
		type=int,
		default=5,
		help="Maximum retries per item on API errors (default: 5).",
	)
	parser.add_argument(
		"--retry-backoff-seconds",
		type=float,
		default=2.0,
		help="Base seconds for exponential backoff between retries (default: 2.0).",
	)
	parser.add_argument(
		"--resume",
		action="store_true",
		default=True,
		help="Resume from existing output file by skipping already processed indices.",
	)
	parser.add_argument(
		"--no-resume",
		action="store_false",
		dest="resume",
		help="Disable resume behavior and process all rows.",
	)
	parser.add_argument(
		"--limit",
		type=int,
		default=None,
		help="Optional max number of records to process.",
	)
	return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict]]:
	with path.open("r", encoding="utf-8") as f:
		for idx, line in enumerate(f):
			line = line.strip()
			if not line:
				continue
			try:
				obj = json.loads(line)
			except json.JSONDecodeError as exc:
				raise ValueError(f"Invalid JSON on line {idx + 1}: {exc}") from exc
			yield idx, obj


def load_processed_indices(output_path: Path) -> set[int]:
	if not output_path.exists():
		return set()

	processed: set[int] = set()
	with output_path.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			try:
				row = json.loads(line)
			except json.JSONDecodeError:
				continue
			idx = row.get("index")
			if isinstance(idx, int):
				processed.add(idx)
	return processed


def extract_response_text(response) -> str:
	text = getattr(response, "output_text", None)
	if text:
		return text.strip()

	out_chunks: list[str] = []
	for item in getattr(response, "output", []) or []:
		for content in getattr(item, "content", []) or []:
			if getattr(content, "type", None) == "output_text":
				out_chunks.append(getattr(content, "text", ""))
	return "\n".join(chunk for chunk in out_chunks if chunk).strip()


def convert_chain_to_lean(
	client: OpenAI,
	chain: str,
	model: str,
	temperature: float,
	max_retries: int,
	retry_backoff_seconds: float,
) -> str:
	user_prompt = (
		"Convert this reasoning chain into Lean 4 code:\n\n"
		f"{chain}\n"
	)

	for attempt in range(1, max_retries + 1):
		try:
			response = client.responses.create(
				model=model,
				temperature=temperature,
				input=[
					{"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
					{"role": "user", "content": user_prompt},
				],
			)
			lean_code = extract_response_text(response)
			if not lean_code:
				raise RuntimeError("Empty model response")
			return lean_code
		except Exception as exc:
			if attempt >= max_retries:
				raise RuntimeError(
					f"Failed after {max_retries} attempts: {exc}"
				) from exc
			sleep_seconds = retry_backoff_seconds * (2 ** (attempt - 1))
			print(
				f"Request failed (attempt {attempt}/{max_retries}): {exc}. "
				f"Retrying in {sleep_seconds:.1f}s...",
				file=sys.stderr,
			)
			time.sleep(sleep_seconds)

	raise RuntimeError("Unreachable retry state")


def main() -> None:
	args = parse_args()

	api_key = os.environ.get("OPENAI_API_KEY")
	if not api_key:
		raise SystemExit("OPENAI_API_KEY is not set.")

	if not args.input.exists():
		raise SystemExit(f"Input file not found: {args.input}")

	args.output.parent.mkdir(parents=True, exist_ok=True)
	processed_indices = load_processed_indices(args.output) if args.resume else set()

	client = OpenAI(api_key=api_key)

	mode = "a" if args.resume else "w"
	processed_count = 0
	skipped_count = 0

	with args.output.open(mode, encoding="utf-8") as out_f:
		for idx, obj in iter_jsonl(args.input):
			if args.limit is not None and processed_count >= args.limit:
				break

			if idx in processed_indices:
				skipped_count += 1
				continue

			chain = obj.get("chain")
			if not isinstance(chain, str) or not chain.strip():
				print(f"Skipping index {idx}: missing/empty 'chain' field", file=sys.stderr)
				continue

			print(f"Processing index {idx}...")
			try:
				lean_code = convert_chain_to_lean(
					client=client,
					chain=chain,
					model=args.model,
					temperature=args.temperature,
					max_retries=args.max_retries,
					retry_backoff_seconds=args.retry_backoff_seconds,
				)
				result = {
					"index": idx,
					"model": args.model,
					"chain": chain,
					"lean_code": lean_code,
				}
			except Exception as exc:
				result = {
					"index": idx,
					"model": args.model,
					"chain": chain,
					"error": str(exc),
				}

			out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
			out_f.flush()
			processed_count += 1

	print(
		f"Done. Processed={processed_count}, "
		f"Skipped(existing)={skipped_count}, Output={args.output}"
	)


if __name__ == "__main__":
	main()
