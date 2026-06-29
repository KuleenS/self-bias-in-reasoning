"""Offline vLLM backend — local GPU batch inference (the original generation path)."""

from __future__ import annotations

from selfbias.inference.base import Conversation, GenParams, Generation
from selfbias.textproc import split_thinking


class VLLMOfflineBackend:
    def __init__(self, spec):
        from transformers import AutoTokenizer
        from vllm import LLM

        self.spec = spec
        self.tokenizer = AutoTokenizer.from_pretrained(spec.id)
        self.llm = LLM(
            model=spec.id,
            trust_remote_code=True,
            max_model_len=spec.max_model_len,
            tensor_parallel_size=spec.tensor_parallel,
        )

    def _sampling(self, params: GenParams):
        from vllm import SamplingParams

        return SamplingParams(
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k if params.top_k else -1,
            max_tokens=params.max_tokens,
        )

    def generate(self, conversations: list[Conversation], params: GenParams) -> list[Generation]:
        kwargs = {"enable_thinking": True} if self.spec.thinking_template else {}
        texts = [
            self.tokenizer.apply_chat_template(
                c, tokenize=False, add_generation_prompt=True, **kwargs
            )
            for c in conversations
        ]
        outputs = self.llm.generate(texts, self._sampling(params))
        results: list[Generation] = []
        for out in outputs:
            comp = out.outputs[0]
            reasoning = getattr(comp, "reasoning_content", None)
            thinking, answer = split_thinking(comp.text, reasoning)
            raw = f"<think>{reasoning}</think>{comp.text}" if reasoning else comp.text
            results.append(Generation(text=answer, thinking=thinking, raw=raw))
        return results
