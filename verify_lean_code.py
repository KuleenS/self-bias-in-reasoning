#!/usr/bin/env python3
"""Verify Lean code in a JSONL dataset and classify error types.

Usage:
	python verify_lean_code.py \
		--input reasoning_chains_lean.jsonl \
		--output reasoning_chains_lean_verified.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable


SYNTAX_PATTERNS = [
	re.compile(r"\bparse error\b", re.IGNORECASE),
	re.compile(r"\bunexpected token\b", re.IGNORECASE),
	re.compile(r"\binvalid syntax\b", re.IGNORECASE),
	re.compile(r"\bexpected (?:token|command|term|tactic|identifier)\b", re.IGNORECASE),
	re.compile(r"\bunexpected end of input\b", re.IGNORECASE),
	re.compile(r"\bunterminated\b", re.IGNORECASE),
	re.compile(r"\bunknown parser declaration\b", re.IGNORECASE),
]

REASONING_PATTERNS = [
	re.compile(r"\btype mismatch\b", re.IGNORECASE),
	re.compile(r"\bapplication type mismatch\b", re.IGNORECASE),
	re.compile(r"\bunsolved goals?\b", re.IGNORECASE),
	re.compile(r"\bfailed to prove\b", re.IGNORECASE),
	re.compile(r"\btactic\b", re.IGNORECASE),
	re.compile(r"\bdon'?t know how to synthesize\b", re.IGNORECASE),
	re.compile(r"\bfailed to synthesize\b", re.IGNORECASE),
	re.compile(r"\bcannot prove\b", re.IGNORECASE),
	re.compile(r"\bunknown identifier\b", re.IGNORECASE),
	re.compile(r"\bunknown constant\b", re.IGNORECASE),
	re.compile(r"\bfailed to unify\b", re.IGNORECASE),
	re.compile(r"\bfunction expected\b", re.IGNORECASE),
	re.compile(r"\binvalid field\b", re.IGNORECASE),
]

LEAN_DIAGNOSTIC_RE = re.compile(
	r"^(?:.+?:\d+:\d+:\s*)?(error|warning|information):\s*(.+)$",
	re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Verify Lean code in JSONL rows and classify errors."
	)
	parser.add_argument(
		"--input",
		type=Path,
		default=Path("reasoning_chains_lean.jsonl"),
		help="Input JSONL file containing a 'lean_code' field.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("reasoning_chains_lean_verified.jsonl"),
		help="Output JSONL file with verification results.",
	)
	parser.add_argument(
		"--lean-cmd",
		type=str,
		default="auto",
		help=(
			"Lean command to run. Use 'auto' to detect or pass a full command string, "
			"e.g. 'lean' or 'lake env lean'."
		),
	)
	parser.add_argument(
		"--timeout-seconds",
		type=int,
		default=30,
		help="Per-row timeout for Lean verification (default: 30).",
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


def resolve_lean_command(lean_cmd: str) -> list[str]:
	if lean_cmd != "auto":
		return shlex.split(lean_cmd)

	if shutil.which("lean"):
		return ["lean"]

	if shutil.which("lake"):
		return ["lake", "env", "lean"]

	raise SystemExit(
		"Could not find Lean executable. Install Lean or pass --lean-cmd explicitly."
	)


def classify_error(error_text: str) -> str:
	error_messages = extract_lean_error_messages(error_text)
	combined = "\n".join(error_messages) if error_messages else error_text

	for pattern in SYNTAX_PATTERNS:
		if pattern.search(combined):
			return "syntax"

	for pattern in REASONING_PATTERNS:
		if pattern.search(combined):
			return "reasoning"

	return "other"


def extract_lean_error_messages(error_text: str) -> list[str]:
	messages: list[str] = []
	for raw_line in error_text.splitlines():
		line = raw_line.strip()
		if not line:
			continue
		match = LEAN_DIAGNOSTIC_RE.match(line)
		if not match:
			continue
		severity = match.group(1).lower()
		message = match.group(2).strip()
		if severity == "error" and message:
			messages.append(message)

	if messages:
		return messages

	fallback = error_text.strip()
	return [fallback] if fallback else []


def verify_lean_code(lean_code: str, lean_command: list[str], timeout_seconds: int) -> dict:
	with tempfile.TemporaryDirectory(prefix="lean_verify_") as tmpdir:
		lean_file = Path(tmpdir) / "Main.lean"
		lean_file.write_text(lean_code, encoding="utf-8")

		cmd = [*lean_command, str(lean_file)]
		try:
			result = subprocess.run(
				cmd,
				capture_output=True,
				text=True,
				timeout=timeout_seconds,
				check=False,
			)
		except subprocess.TimeoutExpired as exc:
			error_message = f"Lean verification timed out after {timeout_seconds}s"
			return {
				"is_valid": False,
				"error_type": "other",
				"error_message": error_message,
				"stdout": (exc.stdout or "").strip(),
				"stderr": (exc.stderr or "").strip(),
				"exit_code": None,
			}

		stdout = (result.stdout or "").strip()
		stderr = (result.stderr or "").strip()
		has_error = result.returncode != 0
		error_message = stderr or stdout

		return {
			"is_valid": not has_error,
			"error_type": None if not has_error else classify_error(error_message),
			"error_message": None if not has_error else error_message,
			"stdout": stdout,
			"stderr": stderr,
			"exit_code": result.returncode,
		}


def main() -> None:
	args = parse_args()

	if not args.input.exists():
		raise SystemExit(f"Input file not found: {args.input}")

	lean_command = resolve_lean_command(args.lean_cmd)
	args.output.parent.mkdir(parents=True, exist_ok=True)

	valid_count = 0
	error_count = 0

	with args.output.open("w", encoding="utf-8") as out_f:
		for idx, row in iter_jsonl(args.input):
			lean_code = row.get("lean_code")
			if not isinstance(lean_code, str) or not lean_code.strip():
				verification = {
					"is_valid": False,
					"error_type": "other",
					"error_message": "Missing or empty 'lean_code' field",
					"stdout": "",
					"stderr": "",
					"exit_code": None,
				}
				error_count += 1
			else:
				verification = verify_lean_code(
					lean_code=lean_code,
					lean_command=lean_command,
					timeout_seconds=args.timeout_seconds,
				)
				if verification["is_valid"]:
					valid_count += 1
				else:
					error_count += 1

			result = {
				**row,
				"index": row.get("index", idx),
				"lean_verification": verification,
			}
			out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

	print(
		f"Done. Total={valid_count + error_count}, Valid={valid_count}, "
		f"Errors={error_count}, Output={args.output}"
	)


if __name__ == "__main__":
	main()