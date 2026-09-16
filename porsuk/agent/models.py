"""The answer domain types the agent produces.

`Source` is one cited passage: file, page, section, the quoted text.
`AgentAnswer` is the whole result: the answer prose, the sources it cites,
every document it looked at, how many tool calls it spent, whether the
loop hit its step limit before finishing, and whether the run
was cancelled (a caller's `stop` Event fired, the answer is partial but is
*not* a step-limit overflow). Both are frozen so a returned answer cannot be
mutated by a caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Source:
    file: str
    page: int | None
    section: str | None
    quote: str
    chunk_id: str | None = None
    doc_id: str | None = None


@dataclass(frozen=True)
class AgentAnswer:
    text: str
    sources: list[Source] = field(default_factory=list)
    documents_seen: list[Source] = field(default_factory=list)
    tool_calls: int = 0
    truncated: bool = False
    cancelled: bool = False
