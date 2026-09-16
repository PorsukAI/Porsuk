"""`run_agent` drives the `langchain.agents.create_agent` loop, enforces
`max_tool_calls` via LangGraph's `recursion_limit`, and turns the final
message list into an `AgentAnswer`, resolving sources from inline
`⟦chunk_id⟧` marks the model places in its answer rather than parsing prose.
The loop always runs through `.stream(...)` rather than `.invoke`, because
`.invoke` does not hand back the partial transcript on a step-limit
`GraphRecursionError`; a caller may also pass a `threading.Event` as `stop`
to cancel a run (e.g. on SSE client disconnect) without it being reported as
step-limit truncation.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Callable, Iterator
from threading import Event

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.errors import GraphRecursionError

from porsuk.agent.graph import build_graph, recursion_limit
from porsuk.agent.models import AgentAnswer, Source
from porsuk.agent.prompts import SYSTEM_PROMPT
from porsuk.agent.tools import build_tools
from porsuk.core import container
from porsuk.core.models import Filters

_KULLANILAN = re.compile(r"(?:^|\s)KULLANILAN:[ \t]*(.*)$", re.MULTILINE)
# A real chunk_id is `<uuid5>:<5 digits>` (splitter.py): the char class must
# admit the colon and the uuid's hyphens (and a dot, defensively).
_MARK = re.compile(r"⟦([A-Za-z0-9_:.\-]+)⟧")
_STEP_LIMIT_PROMPT = "Adım limitine ulaşıldı. Eldeki bilgiyle, yeni araç çağırmadan cevap ver."
_TRUNCATED_MARK = "[adım limiti aşıldı] "


def run_agent(
    question: str,
    *,
    cfg,
    lang: str | None = None,
    verbose: bool = False,
    on_event: Callable[[dict], None] | None = None,
    stop: Event | None = None,
    app=None,
    scope: str | None = None,
    history: list[dict[str, str]] | None = None,
) -> AgentAnswer:
    app = app or container.build_app(cfg)
    retriever = container.build_retriever(cfg, app=app)
    store = app.store
    chat = container.build_chat_model(cfg).chat

    tools = build_tools(retriever, store, cfg=cfg.agent, lang=lang, scope=scope)
    graph = build_graph(chat, tools, context_trim_tokens=cfg.agent.context_trim_tokens)

    # unscoped on purpose: a display-only doc_id→filename map, consulted via
    # .get(did, did) only for ids already in the (scoped) chunk pool, so no
    # out-of-scope filename can reach the output.
    doc_id_to_file = {
        r.document_id: r.filename for r in store.list_documents(Filters(), limit=10_000)
    }
    limit = recursion_limit(cfg.agent.max_tool_calls)

    sink = on_event or (_print_event if verbose else None)
    stream_mode = ["updates", "messages"] if sink is not None else "updates"

    messages: list = []
    truncated = False
    cancelled = False
    at_turn_start = True  # see the "messages" branch below
    seed = [*_history_messages(history), ("user", question)]
    try:
        for item in graph.stream(
            {"messages": seed},
            stream_mode=stream_mode,
            config={"recursion_limit": limit},
        ):
            # A single-string stream_mode yields bare payloads; a list yields
            # (mode, payload) tuples. Normalize so the loop body is uniform.
            mode, payload = ("updates", item) if stream_mode == "updates" else item

            if stop is not None and stop.is_set():
                # Cancelled: partial answer, so truncated, but not a step-limit
                # overflow, so `_build_answer` withholds the step-limit marker.
                truncated = True
                cancelled = True
                break

            if mode == "messages":
                # (AIMessageChunk, metadata): a chunk carries prose in
                # `.content` OR a piece of a tool call's JSON in
                # `.tool_call_chunks`, never both (verified against
                # OpenRouter's stream directly: a tool-calling turn's
                # `content` is empty/None on every chunk). So prose can be
                # pushed to the sink the instant it arrives, no need to wait
                # for the turn to finish to know whether it "was" a tool
                # call. `_build_answer` gets the final text from `messages`
                # (the "updates" tuples) independently, not from these
                # streamed chunks. Ignored entirely when there is no sink,
                # keeping the `on_event=None` path unchanged.
                message_chunk, meta = payload
                if sink is not None and meta.get("langgraph_node") == "model":
                    text = _chunk_text(message_chunk)
                    # A turn's opening chunk(s) are sometimes leading
                    # whitespace ("\n\n", or "\n\nReal text" in one chunk) -
                    # template padding before the real content or a tool
                    # call starts, not prose (observed directly on
                    # OpenRouter). Forwarding it makes the UI's message box
                    # jump/scroll before anything readable has arrived, so
                    # it's stripped - but only while still at the start of
                    # the turn, and only the leading edge (lstrip, not
                    # strip): a mid-answer chunk that happens to be
                    # whitespace (e.g. the space between two words, which
                    # some models emit as its own chunk) is real content and
                    # must reach the UI unchanged, or words run together.
                    if at_turn_start:
                        text = text.lstrip()
                    if text:
                        sink({"type": "chunk", "text": text})
                        at_turn_start = False
                continue

            # mode == "updates": {node: {"messages": [...]}}
            for node, node_payload in payload.items():
                new = node_payload["messages"]
                messages.extend(new)
                if sink is not None:
                    if node == "model":
                        at_turn_start = True
                    for ev in _events_from(node, new, doc_id_to_file):
                        sink(ev)
                    if node == "tools":
                        # Grow the UI's resolvable-source set as soon as each
                        # tool result lands, not only once the final answer
                        # arrives, otherwise a ⟦chunk_id⟧ mark streamed
                        # inside the answer text has nothing to resolve
                        # against yet and the UI silently drops the badge
                        # until the whole run finishes.
                        pool, _docs_only, _n = _build_pool(messages)
                        sink(
                            {
                                "type": "sources_update",
                                "sources": [
                                    dataclasses.asdict(_source(e, doc_id_to_file))
                                    for e in pool.values()
                                ],
                            }
                        )
    except GraphRecursionError:
        # The turn that tripped the limit is replaced by the recovery answer
        # below (not token-streamed, see the module docstring).
        # `stream(stream_mode="updates")` never emits the seed messages, and
        # `create_agent` injects the system prompt at model-call time, not
        # into graph state, so `messages` here is only this turn's AI/tool
        # messages. Rebuild the full context for the one tool-free recovery
        # turn: system prompt (carries the citation-mark instruction) + any
        # prior-turn history + the question + the transcript so far + the
        # "answer now" nudge. Only the response is appended to `messages`;
        # `_build_answer` needs nothing else.
        recovery_input = [
            SystemMessage(SYSTEM_PROMPT),
            *_history_messages(history),
            HumanMessage(question),
            *messages,
            HumanMessage(_STEP_LIMIT_PROMPT),
        ]
        # The recovery turn already carries the full multi-turn transcript
        # (every tool result from the run that hit the limit) - on a
        # reasoning model that is a lot of context to re-think through
        # before answering "with what you have, no more tools". Turning
        # reasoning off here keeps the recovery turn from turning into a
        # second, even-longer stall on top of the one that triggered it.
        # Both field names covered - see openai_compatible.py's `complete`.
        response = chat.bind(
            extra_body={"chat_template_kwargs": {"enable_thinking": False}, "reasoning": {"enabled": False}}
        ).invoke(recovery_input)
        messages.append(response)
        truncated = True

    return _build_answer(messages, doc_id_to_file, truncated=truncated, cancelled=cancelled)


def _history_messages(history: list[dict[str, str]] | None) -> list[tuple[str, str]]:
    """Prior turns as `(role, content)` pairs the graph's message seed and the
    recovery `HumanMessage`/`AIMessage` list both accept.

    Stateless by design (the "one App" model carries no per-caller
    state): the caller, the API request, the CLI's `--history`, supplies
    every prior turn on every call. A turn's `⟦chunk_id⟧` marks are not
    re-resolved into `sources` for this call (only the pool built from THIS
    turn's tool results is); they stay in the text as-is, which is fine: the
    model reads them as prose context, and the UI already rendered them when
    that turn was current. Only "user" and "assistant" roles are accepted;
    anything else is dropped rather than raising, so a client sending stray
    metadata cannot break the call.
    """
    if not history:
        return []
    return [
        (turn["role"], turn["content"])
        for turn in history
        if turn.get("role") in ("user", "assistant") and turn.get("content")
    ]


def _chunk_text(message_chunk) -> str:
    """The text content of one streamed `AIMessageChunk`. `.content` is a str
    for the models Porsuk uses; guard the list form (content blocks) just in
    case a provider returns it."""
    content = getattr(message_chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return ""


def _events_from(node: str, new_messages: list, doc_id_to_file: dict) -> Iterator[dict]:
    if node == "model":
        for m in new_messages:
            for call in getattr(m, "tool_calls", None) or []:
                yield {"type": "tool_call", "name": call["name"], "args": call.get("args", {})}
    elif node == "tools":
        for m in new_messages:
            if isinstance(m, ToolMessage):
                yield {
                    "type": "tool_result",
                    "name": m.name,
                    "summary": _tool_summary(m.content, doc_id_to_file),
                }


def _tool_summary(content, doc_id_to_file: dict) -> str:
    """A human-readable one-liner for a tool result, for the `tool_result`
    event, what the UI's agent-trace panel shows for "what did
    this step find". The raw content is a JSON `list[dict]` of chunk/document
    rows or an error; a raw-JSON prefix (the old behaviour) reads as noise,
    not an answer, reported from the live trace panel. Chunk-level rows
    (keyword_search/semantic_search) carry `doc_id`, not a filename:
    `doc_id_to_file` (already built once in `run_agent`) resolves
    it the same way `_source` does, so the summary reads "mevzuat_2547.pdf",
    not a uuid.
    """
    try:
        rows = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return " ".join(str(content).split())[:200]
    if not isinstance(rows, list):
        return " ".join(str(content).split())[:200]
    if not rows:
        return "0 results"
    if isinstance(rows[0], dict) and "error" in rows[0]:
        return str(rows[0]["error"])[:200]

    def _label(row: dict) -> str:
        did = row.get("doc_id")
        file = row.get("file") or row.get("filename") or doc_id_to_file.get(did) or did or "?"
        section = row.get("section")
        page = row.get("page")
        bits = [str(file)]
        if section:
            bits.append(str(section))
        elif page is not None:
            bits.append(f"s. {page}")
        # Chunk-returning tools (semantic_search/keyword_search/get_document/
        # expand_context) carry the passage text itself, without it the
        # trace panel showed only "where" (file, section), not "what was
        # actually found there".
        text = row.get("text")
        if text:
            snippet = " ".join(str(text).split())[:160]
            bits.append(f"— {snippet}" + ("…" if len(str(text)) > 160 else ""))
        return " ".join(bits)

    labels = [_label(r) for r in rows if isinstance(r, dict)]
    return f"{len(rows)} result: " + "; ".join(labels)


def _print_event(event: dict) -> None:
    kind = event["type"]
    if kind == "tool_call":
        print(f"  · {event['name']}({event['args']})")
    elif kind == "tool_result":
        print(f"  → {event['name']}: {event['summary']}")
    elif kind == "thought":
        print(f"  💭 {event['text'].strip()}")
    elif kind == "chunk":
        print(event["text"], end="", flush=True)


def _build_pool(messages: list) -> tuple[dict[str, dict], dict[str, int | None], int]:
    """Two ingests from the same tool rows: the chunk pool (keyed by chunk_id,
    feeds citations) and a document-level map (keyed by doc_id,
    from rows that carry a doc_id but no chunk_id: `search_documents`,
    `get_document`, `list_documents` feed `documents_seen`).
    Runs over whatever prefix of `messages` is available, so the caller can
    call this after every tool result (a growing pool, for the sources
    stream) as well as once at the end (`_build_answer`)."""
    pool: dict[str, dict] = {}
    docs_only: dict[str, int | None] = {}
    tool_calls = 0
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        tool_calls += 1
        try:
            rows = json.loads(m.content)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict) or "error" in row:
                continue
            if "chunk_id" in row:
                cid = row["chunk_id"]
                if cid in pool:
                    continue
                pool[cid] = {
                    "doc_id": row.get("doc_id"),
                    "page": row.get("page"),
                    "section": row.get("section"),
                    "quote": row.get("text") or "",
                    "chunk_id": cid,
                }
            elif "doc_id" in row:
                did = row["doc_id"]
                if did is not None and did not in docs_only:
                    docs_only[did] = row.get("page")
    return pool, docs_only, tool_calls


def _build_answer(
    messages: list, doc_id_to_file: dict, *, truncated: bool, cancelled: bool = False
) -> AgentAnswer:
    pool, docs_only, tool_calls = _build_pool(messages)

    final_text = ""
    for m in reversed(messages):
        if isinstance(m, AIMessage) and isinstance(m.content, str) and m.content.strip():
            final_text = m.content
            break

    # Inline ⟦chunk_id⟧ marks are the citation scheme. Collect
    # them in first-appearance order, deduped. Marks stay in the text: the
    # UI turns them into source links.
    cited_ids: list[str] = list(dict.fromkeys(_MARK.findall(final_text)))

    # Backward tolerance: a model may still emit the old trailing
    # `KULLANILAN:` line. Strip it either way; use its ids only if there were
    # no inline marks.
    match = _KULLANILAN.search(final_text)
    if match is not None:
        if not cited_ids:
            cited_ids = [c.strip() for c in match.group(1).split(",") if c.strip()]
        final_text = _KULLANILAN.sub("", final_text).strip()

    # Resolve the cited ids against the pool first; fall back to the top-3
    # touched entries when there is no citation, it named nothing, or none of
    # the ids it named exist in the pool, UNLESS the
    # model explicitly abstained ("BULAMADIM", the system prompt's instruction for
    # "the answer isn't in the documents"). An abstaining answer citing
    # whatever it happened to touch is worse than citing nothing: the pool
    # holds search results the model itself judged irrelevant.
    resolved = [pool[cid] for cid in cited_ids if cid in pool]
    abstained = "BULAMADIM" in final_text.upper()
    sources = (
        []
        if abstained and not resolved
        else [_source(e, doc_id_to_file) for e in (resolved or list(pool.values())[:3])]
    )

    # `documents_seen`: distinct doc_ids from the chunk pool (with their page)
    # unioned with doc-only rows, first-seen order, deduped by doc_id. A doc
    # seen both ways keeps the chunk pool's page.
    doc_pages: dict[str, int | None] = {}
    for entry in pool.values():
        did = entry["doc_id"]
        if did is not None and did not in doc_pages:
            doc_pages[did] = entry["page"]
    for did, page in docs_only.items():
        if did not in doc_pages:
            doc_pages[did] = page
    documents_seen = [
        Source(file=doc_id_to_file.get(did, did), page=page, section=None, quote="", doc_id=did)
        for did, page in doc_pages.items()
    ]

    # The marker means "step limit exceeded", only true for an overflow, not
    # for a run the caller cancelled (which is also `truncated`, being partial).
    if truncated and not cancelled:
        final_text = _TRUNCATED_MARK + final_text

    return AgentAnswer(
        text=final_text,
        sources=sources,
        documents_seen=documents_seen,
        tool_calls=tool_calls,
        truncated=truncated,
        cancelled=cancelled,
    )


def _source(entry: dict, doc_id_to_file: dict) -> Source:
    return Source(
        file=doc_id_to_file.get(entry["doc_id"], entry["doc_id"]),
        page=entry["page"],
        section=entry["section"],
        quote=entry["quote"],
        chunk_id=entry.get("chunk_id"),
        doc_id=entry.get("doc_id"),
    )
