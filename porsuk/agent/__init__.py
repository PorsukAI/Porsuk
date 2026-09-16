"""The LangGraph agent package.

`models` holds the answer domain types, provider-free, importable anywhere.
`run_agent` builds and runs the loop and so pulls in `langchain`; it is
exposed lazily through `__getattr__` so importing `porsuk.agent` for
`AgentAnswer` / `Source` costs nothing.
"""

from __future__ import annotations

from porsuk.agent.models import AgentAnswer, Source

__all__ = ["AgentAnswer", "Source", "run_agent"]


def __getattr__(name: str):  # lazy: run_agent pulls in langchain
    if name == "run_agent":
        from porsuk.agent.run import run_agent

        return run_agent
    raise AttributeError(name)
