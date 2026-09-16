"""LangChain chat model adapters (real, wrapping `langchain_openai.ChatOpenAI`,
and a scripted fake for network-free tests) that give the agent a tool-calling
chat model without `porsuk/agent/` importing `langchain_openai` directly.
"""

from __future__ import annotations

from porsuk.core.registry import register


@register("llm", "langgraph_chat")
class LangGraphChatModel:
    """Wraps `langchain_openai.ChatOpenAI` so the agent reaches an
    OpenAI-compatible endpoint through an adapter, not directly."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout: float = 120.0,
    ) -> None:
        from langchain_openai import ChatOpenAI

        self.model = model
        self.chat = ChatOpenAI(
            model=model,
            base_url=base_url.rstrip("/"),
            api_key=api_key or "not-needed",
            temperature=temperature,
            timeout=timeout,
        )


@register("llm", "fake_chat")
class FakeChatModel:
    """The network-free test double. A `GenericFakeChatModel` subclass (the stock
    LangChain fakes do not implement `bind_tools`, which `create_agent`
    requires) with a no-op `bind_tools`. Each `script` entry is either a
    plain string (a final answer, surfaced as `.content`) or a dict
    `{"tool": name, "args": {...}}` (a tool call, surfaced as `.tool_calls`),
    turned into an `AIMessage` iterator that sequential `.invoke()` calls
    consume one at a time. The default script is one final answer, so
    `build("llm", {"provider": "fake_chat"})` yields a usable model with no
    config."""

    def __init__(
        self, *, model: str = "fake-model", script: list | None = None, **_: object
    ) -> None:
        # `**_`: swallow unknown kwargs so `build_chat_model` stays
        # provider-agnostic. It dumps `AgentConfig` and hands the adapter
        # whatever keys survive its exclude-set: for `provider=fake_chat`
        # that includes `temperature`, which only the real model uses.
        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
        from langchain_core.messages import AIMessage

        self.model = model
        messages = []
        for i, step in enumerate(script or ["fake answer\nKULLANILAN: "]):
            if isinstance(step, str):
                messages.append(AIMessage(content=step))
            else:
                messages.append(
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": step["tool"],
                                "args": step["args"],
                                "id": step.get("id", f"call_{i}"),
                                "type": "tool_call",
                            }
                        ],
                    )
                )

        class _Fake(GenericFakeChatModel):
            def bind_tools(self, tools, **kwargs):  # noqa: ARG002
                return self

        self.chat = _Fake(messages=iter(messages))
