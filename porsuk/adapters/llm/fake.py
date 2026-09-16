"""In-process LLM double so the pipeline runs without a GPU or network."""

from __future__ import annotations

from porsuk.core.registry import register


@register("llm", "fake")
class FakeLLM:
    def __init__(self, model: str = "fake-model", responses: list[str] | None = None) -> None:
        self.model = model
        self._responses = list(responses) if responses else ["fake response"]
        self._index = 0
        self.prompts: list[str] = []

    def complete(self, prompt: str, *, max_tokens: int = 512, enable_thinking: bool = True) -> str:
        self.prompts.append(prompt)
        response = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        return response
