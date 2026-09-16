"""The langgraph_chat adapter registers a real ChatOpenAI wrapper and a
scripted fake, both exposing `.model` and `.chat`."""

from __future__ import annotations


def test_registered_real_and_fake():
    from porsuk.core.container import _load_adapters
    from porsuk.core.registry import build

    _load_adapters()
    real = build(
        "llm",
        {"provider": "langgraph_chat", "model": "qwen3-4b", "base_url": "http://x/v1"},
    )
    assert real.model == "qwen3-4b"
    from langchain_openai import ChatOpenAI

    assert isinstance(real.chat, ChatOpenAI)

    fake = build("llm", {"provider": "fake_chat"})
    assert fake.model == "fake-model"


def test_fake_chat_replays_script():
    from langchain_core.messages import HumanMessage

    from porsuk.adapters.llm.langgraph_chat import FakeChatModel

    m = FakeChatModel(
        script=[
            {"tool": "semantic_search", "args": {"query": "ödeme"}},
            "işte cevap\nKULLANILAN: c1",
        ]
    )
    first = m.chat.invoke([HumanMessage(content="soru")])
    assert first.tool_calls[0]["name"] == "semantic_search"
    assert first.tool_calls[0]["args"] == {"query": "ödeme"}
    second = m.chat.invoke([HumanMessage(content="devam")])
    assert "KULLANILAN: c1" in second.content


def test_fake_chat_drives_create_agent():
    """The reason FakeChatModel subclasses GenericFakeChatModel with a bind_tools
    no-op: create_agent calls model.bind_tools(), which the stock fakes raise on."""
    from langchain.agents import create_agent
    from langchain_core.tools import tool

    from porsuk.adapters.llm.langgraph_chat import FakeChatModel

    @tool
    def semantic_search(query: str) -> str:
        """search"""
        return "found"

    m = FakeChatModel(
        script=[
            {"tool": "semantic_search", "args": {"query": "x"}},
            "cevap\nKULLANILAN: c0",
        ]
    )
    agent = create_agent(m.chat, [semantic_search])  # must not raise
    out = agent.invoke({"messages": [("user", "soru")]})
    assert [type(x).__name__ for x in out["messages"]] == [
        "HumanMessage",
        "AIMessage",
        "ToolMessage",
        "AIMessage",
    ]
    assert out["messages"][-1].content.startswith("cevap")


def test_fake_chat_default_script():
    from langchain_core.messages import HumanMessage

    from porsuk.adapters.llm.langgraph_chat import FakeChatModel

    m = FakeChatModel()
    out = m.chat.invoke([HumanMessage(content="soru")])
    assert "KULLANILAN" in out.content
