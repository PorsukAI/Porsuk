"""Tests for the SSE agent-event adapter (`agent_events`)."""

from __future__ import annotations

import threading

from porsuk.api._events import agent_events


def _stub_run_with_events(
    question, *, lang=None, verbose=False, on_event=None, stop=None, scope=None, history=None
):
    from porsuk.agent.models import AgentAnswer, Source

    if on_event:
        on_event({"type": "tool_call", "name": "semantic_search", "args": {"query": question}})
        on_event({"type": "tool_result", "name": "semantic_search", "summary": "1 hit"})
    return AgentAnswer(text="cevap", sources=[Source("f.pdf", 1, None, "q", "c1")], tool_calls=1)


def test_agent_events_yields_steps_then_answer():
    events = list(agent_events(_stub_run_with_events, "soru"))
    assert [e["type"] for e in events] == ["tool_call", "tool_result", "answer"]
    call = events[0]
    assert call["name"] == "semantic_search"
    assert call["args"] == {"query": "soru"}
    answer = events[-1]["answer"]
    assert answer["text"] == "cevap"
    assert answer["sources"][0]["file"] == "f.pdf"
    assert answer["sources"][0]["chunk_id"] == "c1"


def test_agent_events_passes_lang_through():
    seen = {}

    def run(question, *, lang=None, verbose=False, on_event=None, stop=None, scope=None, history=None):
        from porsuk.agent.models import AgentAnswer

        seen["lang"] = lang
        return AgentAnswer(text="x")

    list(agent_events(run, "soru", lang="tr"))
    assert seen["lang"] == "tr"


def test_agent_events_threads_scope():
    seen = {}

    def run(question, *, lang=None, verbose=False, on_event=None, stop=None, scope=None, history=None):
        from porsuk.agent.models import AgentAnswer

        seen["scope"] = scope
        return AgentAnswer(text="x")

    list(agent_events(run, "soru", scope="job-9"))
    assert seen["scope"] == "job-9"


def test_agent_events_yields_error_on_exception():
    def boom(question, *, lang=None, verbose=False, on_event=None, stop=None, scope=None, history=None):
        raise RuntimeError("kaboom")

    events = list(agent_events(boom, "soru"))
    assert events == [{"type": "error", "detail": "kaboom"}]


def test_agent_events_sets_stop_on_close():
    """Closing the generator early (client disconnect) must set the cancel
    Event the agent loop watches."""
    seen: dict = {}
    release = threading.Event()

    def run(question, *, lang=None, verbose=False, on_event=None, stop=None, scope=None, history=None):
        from porsuk.agent.models import AgentAnswer

        seen["stop"] = stop
        on_event({"type": "tool_call", "name": "search", "args": {}})
        release.wait(timeout=5)  # block until the generator is closed
        return AgentAnswer(text="x")

    gen = agent_events(run, "soru")
    assert next(gen)["type"] == "tool_call"  # consume only the first event
    gen.close()
    release.set()

    assert seen["stop"] is not None
    assert seen["stop"].is_set()


def test_agent_events_forwards_thought_and_chunk_then_answer():
    def run(question, *, lang=None, verbose=False, on_event=None, stop=None, scope=None, history=None):
        from porsuk.agent.models import AgentAnswer

        on_event({"type": "thought", "text": "düşünüyorum"})
        on_event({"type": "tool_call", "name": "s", "args": {}})
        on_event({"type": "tool_result", "name": "s", "summary": "1 hit"})
        on_event({"type": "chunk", "text": "Cevap "})
        on_event({"type": "chunk", "text": "burada."})
        return AgentAnswer(text="Cevap burada.")

    events = list(agent_events(run, "soru"))
    kinds = [e["type"] for e in events]
    assert kinds == ["thought", "tool_call", "tool_result", "chunk", "chunk", "answer"]
    assert events[0]["text"] == "düşünüyorum"
    assert "".join(e["text"] for e in events if e["type"] == "chunk") == "Cevap burada."
