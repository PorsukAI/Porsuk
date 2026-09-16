"""Bridge run_agent's on_event callback to a generator for SSE.

`run_agent` is synchronous and reports steps through an `on_event(dict)`
callback. `agent_events` runs it on a worker thread and turns that push into a
pull: a plain generator that yields each step event as it arrives, then a
final `{"type": "answer", ...}` (or `{"type": "error", ...}`). The endpoint
wraps it in `sse_starlette.EventSourceResponse`, which iterates it off the
event loop.
"""

from __future__ import annotations

import dataclasses
import queue
import threading
from collections.abc import Iterator

_DONE = object()
_ERROR = object()


def agent_events(
    run,
    question: str,
    *,
    lang: str | None = None,
    scope: str | None = None,
    stop: threading.Event | None = None,
    history: list[dict[str, str]] | None = None,
) -> Iterator[dict]:
    """Yield each of the agent's step events, then one terminal event.

    `run` is `build_agent_runner`'s return: `run(question, *, lang=None,
    verbose=False, on_event=None, stop=None, scope=None, history=None) ->
    AgentAnswer`. Terminal event is `{"type": "answer", "answer": <AgentAnswer
    as dict>}` on success or `{"type": "error", "detail": <str>}` if `run`
    raised. `history` is the prior turns of this conversation, supplied by
    the caller (the API request body/query); nothing is stored
    server-side.

    `stop` is a cooperative-cancel flag threaded through to the agent loop
    ("bağlantı koparsa agent task iptal edilir"). The caller may
    pass its own so it can set it from outside: the endpoint does, because a
    sync generator's `finally` does not run promptly on client disconnect
    (sse-starlette 3.3+ only stops iterating; see `server.ask_stream`). When
    none is passed one is made here, and this generator's own `finally` sets
    it, enough for a caller that drives the generator to exhaustion or calls
    `.close()` on it directly.
    """
    q: queue.Queue = queue.Queue()
    stop = stop or threading.Event()

    def worker() -> None:
        try:
            result = run(
                question, lang=lang, scope=scope, on_event=q.put, stop=stop, history=history
            )
            q.put((_DONE, result))
        except Exception as exc:  # noqa: BLE001 - surfaced to the client as an SSE error event
            q.put((_ERROR, exc))

    threading.Thread(target=worker, daemon=True).start()

    try:
        while not stop.is_set():
            try:
                # Poll, not a bare `get()`: the caller cancels by setting `stop`
                # from another thread (see `server.ask_stream`), and this loop
                # must notice that even while no event is in flight. The wait is
                # also what lets `iterate_in_threadpool` (the endpoint's
                # off-loader) return to the event loop so a pending cancel can
                # be delivered into the async wrapper.
                item = q.get(timeout=0.25)
            except queue.Empty:
                continue
            if isinstance(item, tuple) and item[0] is _DONE:
                yield {"type": "answer", "answer": dataclasses.asdict(item[1])}
                return
            if isinstance(item, tuple) and item[0] is _ERROR:
                yield {"type": "error", "detail": str(item[1])}
                return
            yield item
    finally:
        # Generator exhausted or closed: signal the worker to stop.
        stop.set()
