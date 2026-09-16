"""The single-node ReAct loop (`build_graph`, a thin wrapper over
`langchain.agents.create_agent`) and its step limit: `recursion_limit`
converts the config's `max_tool_calls` into LangGraph's super-step budget
(`2 * n + 1`). Optional `context_trim_tokens` wires `ContextEditingMiddleware`
to replace tool results older than the most recent 3 calls with a placeholder
once the running token count crosses the threshold, so stale payloads stop
burning context; `None` disables it.
"""

from __future__ import annotations

from porsuk.agent.prompts import SYSTEM_PROMPT

_CLEAR_TOOL_USES_PLACEHOLDER = (
    "[bu aracın eski sonucu bağlam alanı için kısaltıldı — gerekiyorsa tekrar çağır]"
)


def recursion_limit(max_tool_calls: int) -> int:
    return 2 * max_tool_calls + 1


def build_graph(chat_model, tools, *, context_trim_tokens: int | None = None):
    from langchain.agents import create_agent
    from langchain.agents.middleware import ClearToolUsesEdit, ContextEditingMiddleware

    middleware = []
    if context_trim_tokens is not None:
        middleware.append(
            ContextEditingMiddleware(
                edits=[
                    ClearToolUsesEdit(
                        trigger=context_trim_tokens,
                        keep=3,
                        placeholder=_CLEAR_TOOL_USES_PLACEHOLDER,
                    )
                ]
            )
        )

    return create_agent(chat_model, tools, system_prompt=SYSTEM_PROMPT, middleware=middleware)
