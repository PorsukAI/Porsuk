"""The ReAct loop calls the scripted tool, then answers."""

from __future__ import annotations

from porsuk.adapters.llm.langgraph_chat import FakeChatModel
from porsuk.agent.graph import build_graph, recursion_limit


def test_recursion_limit_is_two_x_plus_one():
    assert recursion_limit(8) == 17
    assert recursion_limit(1) == 3


def test_clear_tool_uses_edit_replaces_old_tool_results_past_the_token_trigger():
    """Exercises the exact strategy build_graph wires in: proves the
    placeholder text and the "keep the last N" rule work as configured,
    without needing to drive the whole graph through an LLM."""
    from langchain_core.messages import AIMessage, ToolMessage

    from porsuk.agent.graph import _CLEAR_TOOL_USES_PLACEHOLDER
    from langchain.agents.middleware import ClearToolUsesEdit

    def make_pair(i: int, text: str) -> list:
        call_id = f"call_{i}"
        ai = AIMessage(
            content="",
            tool_calls=[{"name": "semantic_search", "args": {}, "id": call_id, "type": "tool_call"}],
        )
        tool = ToolMessage(content=text, name="semantic_search", tool_call_id=call_id)
        return [ai, tool]

    # 5 tool round-trips, each with a long-ish result so the running token
    # count crosses a low trigger.
    messages = []
    for i in range(5):
        messages.extend(make_pair(i, "sonuç metni " * 50))

    edit = ClearToolUsesEdit(trigger=50, keep=2, placeholder=_CLEAR_TOOL_USES_PLACEHOLDER)
    edit.apply(messages, count_tokens=lambda msgs: sum(len(m.content) for m in msgs) // 4)

    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    # the 2 most recent are untouched; the 3 older ones are cleared.
    assert tool_messages[-1].content.startswith("sonuç metni")
    assert tool_messages[-2].content.startswith("sonuç metni")
    assert tool_messages[0].content == _CLEAR_TOOL_USES_PLACEHOLDER
    assert tool_messages[1].content == _CLEAR_TOOL_USES_PLACEHOLDER
    assert tool_messages[2].content == _CLEAR_TOOL_USES_PLACEHOLDER


def test_build_graph_with_trim_enabled_clears_old_tool_results_from_the_llm_call(stub_tools):
    """End-to-end through the compiled graph: once several tool round-trips
    push the running message size past a (deliberately tiny) trigger, the
    chat model's later invocations no longer see the old tool payloads,
    just the placeholder, while still seeing the most recent ones."""
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    from porsuk.agent.graph import _CLEAR_TOOL_USES_PLACEHOLDER

    seen_invocations: list[list] = []

    # 6 tool calls (each followed by the loop feeding the tool's real,
    # short, stub result back in), then a final answer with no more calls.
    script_messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "semantic_search",
                    "args": {"query": f"soru {i}"},
                    "id": f"call_{i}",
                    "type": "tool_call",
                }
            ],
        )
        for i in range(6)
    ] + [AIMessage(content="Cevap hazır.")]

    class _SpyChat(GenericFakeChatModel):
        def invoke(self, input, *args, **kwargs):  # noqa: A002
            seen_invocations.append(list(input) if not isinstance(input, dict) else list(input["messages"]))
            return super().invoke(input, *args, **kwargs)

        def bind_tools(self, tools, **kwargs):  # noqa: ARG002
            return self

    chat = _SpyChat(messages=iter(script_messages))
    # A tiny trigger: the stub tool's own short result text is enough to
    # cross it after a few round-trips, well before context genuinely fills.
    graph = build_graph(chat, stub_tools, context_trim_tokens=50)
    graph.invoke(
        {"messages": [("user", "birden çok arama yap")]},
        config={"recursion_limit": recursion_limit(10)},
    )

    # By the last invocation, at least one earlier tool result must have
    # been cleared: the model no longer pays for every old payload.
    last_seen = seen_invocations[-1]
    placeholder_count = sum(
        1 for m in last_seen if getattr(m, "content", None) == _CLEAR_TOOL_USES_PLACEHOLDER
    )
    assert placeholder_count > 0


def test_loop_calls_the_scripted_tool_then_answers(stub_tools):
    chat = FakeChatModel(
        script=[
            {"tool": "semantic_search", "args": {"query": "ödeme"}},
            "Ödeme 30 gün içinde.\nKULLANILAN: c1",
        ]
    ).chat
    graph = build_graph(chat, stub_tools)
    out = graph.invoke({"messages": [("user", "ödeme koşulu ne?")]})

    kinds = [m.type for m in out["messages"]]
    assert kinds == ["human", "ai", "tool", "ai"]
    assert out["messages"][-1].content.startswith("Ödeme 30 gün")
