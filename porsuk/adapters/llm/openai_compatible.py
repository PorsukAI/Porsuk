"""LLM over any OpenAI-compatible /v1/chat/completions.

This is the provider behind the agent LLM and the LLM half of
document profiling (doküman profili). Porsuk always runs
separately from the model - the model on a rented GPU, Porsuk on a laptop
(infra/README.md) - so this adapter talks HTTP: it POSTs a chat completion
request to a vLLM (or any OpenAI-compatible) endpoint and returns the
assistant message text.
"""

from __future__ import annotations

import time

import httpx

from porsuk.core.registry import register

# The agent and the profiler make far fewer calls than indexing does, but a
# hosted endpoint (or a tunnel in front of one) still drops some to rate
# limits or transient 5xx. Retry with backoff rather than fail on one blip;
# a transient failure recovers within a few tries, more just delays the
# error the caller has to surface when the endpoint is truly down.
_MAX_RETRIES = 5
_BACKOFF_BASE_S = 1.0
_BACKOFF_CAP_S = 15.0


@register("llm", "openai_compatible")
class OpenAICompatibleLLM:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._temperature = temperature
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def _request(self, url: str, payload: dict) -> dict:
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._client.post(url, headers=self._headers, json=payload)
                resp.raise_for_status()
                return resp.json()
            except (httpx.HTTPError, ValueError) as exc:
                last_exc = exc
                # Only a connection-level failure needs a fresh client; a
                # 5xx leaves the pool usable, so reuse it.
                if isinstance(exc, httpx.TransportError):
                    self._client.close()
                    self._client = httpx.Client(timeout=self._timeout)
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(min(_BACKOFF_BASE_S * 2**attempt, _BACKOFF_CAP_S))
        raise RuntimeError(f"request to {url} failed after {_MAX_RETRIES} tries: {last_exc}")

    def complete(self, prompt: str, *, max_tokens: int = 512, enable_thinking: bool = True) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": self._temperature,
        }
        if not enable_thinking:
            # A reasoning model (e.g. Qwen3's --reasoning-parser) can spend the
            # whole max_tokens budget on the hidden reasoning trace, finishing
            # with finish_reason=length and an empty content field before it
            # ever emits the answer. Callers that don't need the reasoning
            # (profiling) turn it off so content is never empty on a long
            # prompt. Two different fields because there's no one signal
            # every OpenAI-compatible endpoint understands: vLLM/mlx_lm.server
            # read chat_template_kwargs.enable_thinking, OpenRouter reads
            # reasoning.enabled - each ignores the field it doesn't know.
            payload["chat_template_kwargs"] = {"enable_thinking": False}
            payload["reasoning"] = {"enabled": False}
        data = self._request(self._url, payload)
        return data["choices"][0]["message"]["content"]
