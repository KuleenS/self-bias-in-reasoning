#!/usr/bin/env python3
"""Batch-convert natural-language reasoning chains into Lean code using GPT-5.4.

Usage:
	python convert_reasoning_to_lean.py \
		--folio-input data/FOLIO/folio_train.jsonl \
		--reasoning-input data/reasoning_chains_qwen/reasoning_chains_outputs__Qwen3-32B.jsonl \
		--output reasoning_chains_lean.jsonl

Environment:
	OPENAI_API_KEY must be set.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import sys
import time
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable

try:
	from openai import OpenAI
except ImportError as exc:  # pragma: no cover
	raise SystemExit(
		"Missing dependency: openai. Install with `pip install openai`."
	) from exc


PROMPT = """A model was provided these premises and asked to determine whether the hypothesis is true, false, or indeterminate. I want to evaluate it's reasoning by writing Lean 5 proof-assistant code. Write the program to include these premises and hypothesis. Then, write the reasoning steps of the model as lean code that attempts to get from the premises to the conclusion. If it has logical errors, do not attempt to fix them.

Requirements:
1) Output Lean 4 code only (no markdown fences, no prose).
2) Do not attempt to correct any logical errors in the reasoning chain

Premises: {premises} 

Hypothesis: {hypothesis}

Reasoning Chain: {chain}
"""


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Batch prompt GPT-5.4 to convert reasoning chains into Lean code."
	)
	parser.add_argument(
		"--folio-input",
		type=Path,
		default=Path("data/FOLIO/folio_train.jsonl"),
		help="Input FOLIO JSONL file containing 'premises' and 'conclusion' fields.",
	)
	parser.add_argument(
		"--reasoning-input",
		type=Path,
		default=Path("data/reasoning_chains_qwen/reasoning_chains_outputs__Qwen3-32B.jsonl"),
		help="Input JSONL file containing reasoning chains in the 'generated_text' field.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("data/lean_code/lean_code_outputs__Qwen3-32B.jsonl"),
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
		"--batch-completion-window",
		default="24h",
		help="Batch completion window for OpenAI Batch API (default: 24h).",
	)
	parser.add_argument(
		"--batch-poll-seconds",
		type=float,
		default=10.0,
		help="Seconds between batch status polls (default: 10.0).",
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


def extract_response_text_from_body(body: dict[str, Any]) -> str:
	text = body.get("output_text")
	if isinstance(text, str) and text.strip():
		return text.strip()

	out_chunks: list[str] = []
	for item in body.get("output", []) or []:
		for content in item.get("content", []) or []:
			if content.get("type") == "output_text":
				chunk = content.get("text", "")
				if isinstance(chunk, str) and chunk:
					out_chunks.append(chunk)
	return "\n".join(out_chunks).strip()


def read_api_file_text(client: OpenAI, file_id: str) -> str:
	content = client.files.content(file_id)
	text = getattr(content, "text", None)
	if isinstance(text, str):
		return text

	read_fn = getattr(content, "read", None)
	if callable(read_fn):
		data = read_fn()
		if isinstance(data, bytes):
			return data.decode("utf-8")
		if isinstance(data, str):
			return data

	if isinstance(content, bytes):
		return content.decode("utf-8")

	return str(content)


def main() -> None:
	args = parse_args()

	api_key = os.environ.get("OPENAI_API_KEY")
	if not api_key:
		raise SystemExit("OPENAI_API_KEY is not set.")

	if not args.folio_input.exists():
		raise SystemExit(f"FOLIO input file not found: {args.folio_input}")

	if not args.reasoning_input.exists():
		raise SystemExit(f"Reasoning input file not found: {args.reasoning_input}")

	args.output.parent.mkdir(parents=True, exist_ok=True)
	processed_indices = load_processed_indices(args.output) if args.resume else set()

	client = OpenAI(api_key=api_key)

	mode = "a" if args.resume else "w"
	processed_count = 0
	skipped_count = 0
	pending_rows: list[dict[str, Any]] = []

	for pair_idx, pair in enumerate(
		zip_longest(iter_jsonl(args.folio_input), iter_jsonl(args.reasoning_input))
	):
		folio_item, reasoning_item = pair
		if folio_item is None or reasoning_item is None:
			raise ValueError(
				"Input files have different numbers of non-empty JSONL rows. "
				f"Mismatch at paired row {pair_idx}."
			)

		idx, folio_obj = folio_item
		reasoning_idx, reasoning_obj = reasoning_item

		if idx != reasoning_idx:
			print(
				f"Warning: row indices differ (folio={idx}, reasoning={reasoning_idx}). "
				f"Using folio index {idx} for output.",
				file=sys.stderr,
			)

		if args.limit is not None and len(pending_rows) >= args.limit:
			break

		if idx in processed_indices:
			skipped_count += 1
			continue

		premises = folio_obj.get("premises")
		hypothesis = folio_obj.get("conclusion")
		chain = reasoning_obj.get("generated_text")

		if not isinstance(premises, str) or not premises.strip():
			print(
				f"Skipping index {idx}: missing/empty 'premises' field in folio input",
				file=sys.stderr,
			)
			continue

		if not isinstance(hypothesis, str) or not hypothesis.strip():
			print(
				f"Skipping index {idx}: missing/empty 'conclusion' field in folio input",
				file=sys.stderr,
			)
			continue

		if not isinstance(chain, str) or not chain.strip():
			print(
				f"Skipping index {idx}: missing/empty 'generated_text' field in reasoning input",
				file=sys.stderr,
			)
			continue

		pending_rows.append(
			{
				"index": idx,
				"premises": premises,
				"hypothesis": hypothesis,
				"chain": chain,
			}
		)

	if not pending_rows:
		print(
			f"Done. Processed=0, Skipped(existing)={skipped_count}, Output={args.output}"
		)
		return

	batch_request_lines: list[dict[str, Any]] = []
	for row in pending_rows:
		user_prompt = PROMPT.format(
			premises=row["premises"],
			hypothesis=row["hypothesis"],
			chain=row["chain"],
		)
		batch_request_lines.append(
			{
				"custom_id": str(row["index"]),
				"method": "POST",
				"url": "/v1/responses",
				"body": {
					"model": args.model,
					"temperature": args.temperature,
					"input": [
						{"role": "system", "content": "You are a precise code generation assistant."},
						{"role": "user", "content": user_prompt},
					],
				},
			}
		)

	with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".jsonl", delete=False) as tmp_f:
		tmp_path = Path(tmp_f.name)
		for req in batch_request_lines:
			tmp_f.write(json.dumps(req, ensure_ascii=False) + "\n")

	print(f"Submitting batch with {len(batch_request_lines)} requests...")
	with tmp_path.open("rb") as upload_f:
		input_file = client.files.create(file=upload_f, purpose="batch")

	batch = client.batches.create(
		input_file_id=input_file.id,
		endpoint="/v1/responses",
		completion_window=args.batch_completion_window,
	)
	print(f"Batch submitted: {batch.id}")

	final_statuses = {"completed", "failed", "expired", "cancelled"}
	while True:
		batch = client.batches.retrieve(batch.id)
		status = getattr(batch, "status", "unknown")
		if status in final_statuses:
			print(f"Batch finished with status: {status}")
			break
		print(f"Batch status: {status}. Polling again in {args.batch_poll_seconds:.1f}s...")
		time.sleep(args.batch_poll_seconds)

	output_map: dict[int, dict[str, Any]] = {}
	if getattr(batch, "output_file_id", None):
		output_text = read_api_file_text(client, batch.output_file_id)
		for line in output_text.splitlines():
			line = line.strip()
			if not line:
				continue
			try:
				row = json.loads(line)
			except json.JSONDecodeError:
				continue

			custom_id = row.get("custom_id")
			try:
				idx = int(custom_id)
			except (TypeError, ValueError):
				continue

			resp = row.get("response") or {}
			status_code = resp.get("status_code")
			body = resp.get("body") or {}
			error_obj = row.get("error")

			if error_obj:
				output_map[idx] = {"error": str(error_obj)}
				continue

			if status_code != 200:
				output_map[idx] = {"error": f"Non-200 response from batch item: {status_code}"}
				continue

			lean_code = extract_response_text_from_body(body)
			if lean_code:
				output_map[idx] = {"lean_code": lean_code}
			else:
				output_map[idx] = {"error": "Empty model response"}

	if getattr(batch, "error_file_id", None):
		error_text = read_api_file_text(client, batch.error_file_id)
		for line in error_text.splitlines():
			line = line.strip()
			if not line:
				continue
			try:
				row = json.loads(line)
			except json.JSONDecodeError:
				continue
			custom_id = row.get("custom_id")
			try:
				idx = int(custom_id)
			except (TypeError, ValueError):
				continue
			error_obj = row.get("error")
			if error_obj and idx not in output_map:
				output_map[idx] = {"error": str(error_obj)}

	with args.output.open(mode, encoding="utf-8") as out_f:
		for row in pending_rows:
			idx = row["index"]
			batch_result = output_map.get(idx)
			if batch_result is None:
				result = {
					"index": idx,
					"model": args.model,
					"premises": row["premises"],
					"hypothesis": row["hypothesis"],
					"chain": row["chain"],
					"error": "No result returned for this batch item.",
				}
			elif "lean_code" in batch_result:
				result = {
					"index": idx,
					"model": args.model,
					"premises": row["premises"],
					"hypothesis": row["hypothesis"],
					"lean_code": batch_result["lean_code"],
				}
			else:
				result = {
					"index": idx,
					"model": args.model,
					"premises": row["premises"],
					"hypothesis": row["hypothesis"],
					"chain": row["chain"],
					"error": batch_result.get("error", "Unknown batch error"),
				}

			out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
			processed_count += 1

		out_f.flush()

	try:
		tmp_path.unlink()
	except OSError:
		pass

	print(
		f"Done. Processed={processed_count}, "
		f"Skipped(existing)={skipped_count}, Output={args.output}"
	)


if __name__ == "__main__":
	main()
