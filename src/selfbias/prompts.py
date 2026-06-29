"""Prompt templates for generation and cross-model evaluation.

The evaluation template is the generalized descendant of the old FOLIO `EVAL_PROMPT_TEMPLATE`:
the evaluator judges a reasoning chain's correctness from its merits alone — it never sees the
gold answer (gold is used only to compute ground truth in analysis).
"""

from __future__ import annotations

from selfbias.data.base import ReasoningExample
from selfbias.inference.base import Conversation

_ANSWER_INSTRUCTION = {
    "numeric": "End your response with a line of the exact form 'Answer: <number>'.",
    "mcq": "End your response with a line of the exact form 'Answer: <letter>' "
    "(the single letter of the correct option).",
    "freeform": "End your response with a line of the exact form 'Answer: <final answer>'.",
}

_GENERATION_TEMPLATE = """\
Solve the following problem. Think step by step, then give your final answer.

{question}

{instruction}"""

EVAL_TEMPLATE = """\
You are an expert evaluator. Below is a problem and a reasoning chain that attempts to solve it.

Problem:
{question}

Reasoning chain:
{reasoning}

Task: Evaluate whether the reasoning chain is logically correct and arrives at the right answer. \
Be strict — if any step is not completely sound or the final answer is wrong, disqualify the chain.
Output a JSON object with exactly one key:
  "valid": true if the reasoning contains no errors and reaches the correct answer, false otherwise
"""


def generation_messages(example: ReasoningExample) -> Conversation:
    instruction = _ANSWER_INSTRUCTION.get(example.answer_type, _ANSWER_INSTRUCTION["freeform"])
    content = _GENERATION_TEMPLATE.format(question=example.question, instruction=instruction)
    return [{"role": "user", "content": content}]


def evaluation_messages(question: str, reasoning: str, max_reasoning_chars: int = 20000) -> Conversation:
    if max_reasoning_chars and len(reasoning) > max_reasoning_chars:
        reasoning = reasoning[:max_reasoning_chars] + "\n[truncated]"
    content = EVAL_TEMPLATE.format(question=question, reasoning=reasoning)
    return [{"role": "user", "content": content}]
