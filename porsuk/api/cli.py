"""Command line entry point ('parse <dosya>' works).

argparse rather than typer or click: one command with three flags does not
justify a dependency, and abstraction is free only where two implementations
exist.

The default output shows the gate's reasoning rather than only its verdict.
The component that triggered an upgrade needs to be visible, and a
person running this on a file that parsed badly wants to know why.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from typing import Any

from porsuk.api._serialize import _PERSISTENT_STORES, _hit_dict
from porsuk.core.config import ConfigError, load_config
from porsuk.core.container import build_pipeline, build_retriever, build_router
from porsuk.ingestion.chunking import chunk
from porsuk.ingestion.quality import decide, document_script_validity

_DEFAULT_CONFIG = "config/local.yaml"
_DEFAULT_STATE = ".porsuk/state.db"
# How much of a chunk the human-readable output shows. Truncation is marked,
# so a reader can tell a cut chunk from a short one.
_PREVIEW_CHARS = 300


def _preview(text: str) -> str:
    """First _PREVIEW_CHARS of a chunk, with a marker when it was cut."""
    if len(text) <= _PREVIEW_CHARS:
        return text
    return text[:_PREVIEW_CHARS] + " […]"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="porsuk", description="Agentic document search.")
    sub = parser.add_subparsers(dest="command", required=True)
    parse_cmd = sub.add_parser("parse", help="Parse one file and report the quality gate.")
    parse_cmd.add_argument("path", help="File to parse.")
    parse_cmd.add_argument("--config", default=_DEFAULT_CONFIG, help="Profile YAML.")
    parse_cmd.add_argument("--json", action="store_true", help="Machine-readable output.")
    parse_cmd.add_argument("--chunks", action="store_true", help="Print the chunks too.")

    index_cmd = sub.add_parser("index", help="Index a folder into the vector store.")
    index_cmd.add_argument("folder", help="Folder of documents to index.")
    index_cmd.add_argument("--config", default=_DEFAULT_CONFIG, help="Profile YAML.")
    index_cmd.add_argument("--state", default=_DEFAULT_STATE, help="SQLite ingestion state.")

    search_cmd = sub.add_parser("search", help="Search the index and print ranked chunks.")
    search_cmd.add_argument("query", help="Natural-language query.")
    search_cmd.add_argument(
        "--strategy",
        choices=["semantic", "keyword", "hybrid"],
        default=None,
        help="Override the config's retrieval.strategy.",
    )
    search_cmd.add_argument(
        "--k",
        type=int,
        default=None,
        help="Chunks to return (default: retrieval.chunk_k from config).",
    )
    search_cmd.add_argument("--lang", default=None, help="Restrict to one language, e.g. 'tr'.")
    search_cmd.add_argument("--config", default=_DEFAULT_CONFIG, help="Profile YAML.")
    search_cmd.add_argument(
        "--state",
        default=_DEFAULT_STATE,
        help="SQLite ingestion state (unused by search; kept for symmetry).",
    )
    search_cmd.add_argument("--json", action="store_true", help="Machine-readable output.")

    ask_cmd = sub.add_parser("ask", help="Ask the agent a question and get a sourced answer.")
    ask_cmd.add_argument("question", help="Natural-language question.")
    ask_cmd.add_argument(
        "--lang", default=None, help="Restrict retrieval to one language, e.g. 'tr'."
    )
    ask_cmd.add_argument("--config", default=_DEFAULT_CONFIG, help="Profile YAML.")
    ask_cmd.add_argument("--verbose", action="store_true", help="Stream each tool call.")
    ask_cmd.add_argument("--json", action="store_true", help="Machine-readable output.")

    serve_cmd = sub.add_parser(
        "serve", help="Run the HTTP API: ask/search and index upload."
    )
    serve_cmd.add_argument("--config", default="config/vllm.yaml", help="Profile YAML.")
    serve_cmd.add_argument("--host", default="127.0.0.1", help="Bind address.")
    serve_cmd.add_argument("--port", type=int, default=8080, help="Bind port.")
    return parser


def _report(outcome, chunks, gate, path: str, show_chunks: bool) -> dict[str, Any]:
    components = dataclasses.asdict(outcome.components) if outcome.components is not None else None
    escalation = None
    if gate is not None and gate.escalate_to is not None:
        escalation = {
            "requested": gate.escalate_to,
            "trigger": gate.trigger,
            "reason": gate.reason,
        }
    report: dict[str, Any] = {
        # The path the user gave, not the document's: a failed parse has no
        # document, and printing None is least useful exactly when a batch run
        # needs to know which file failed.
        "path": outcome.document.path if outcome.document else path,
        "parser_used": outcome.parser_used,
        "status": outcome.status,
        "quality": round(outcome.quality, 4),
        "components": components,
        "escalation": escalation,
        "image_heavy": outcome.image_heavy,
        "error": outcome.error,
        "page_count": outcome.document.page_count if outcome.document else 0,
        "outline_source": outcome.document.outline.source if outcome.document else None,
        "chunk_count": len(chunks),
    }
    if show_chunks:
        # --chunks was silently ignored under --json, so the machine-readable
        # form could not answer a question the human-readable one could.
        report["chunks"] = [
            {
                "chunk_id": c.chunk_id,
                "section_path": c.section_path,
                "page_no": c.page_no,
                "language": c.language,
                "char_span": list(c.char_span),
                "text": c.text,
            }
            for c in chunks
        ]
    return report


def _print_human(report: dict[str, Any], chunks, show_chunks: bool) -> None:
    print(f"path        {report['path']}")
    print(f"parser      {report['parser_used']}")
    print(f"status      {report['status']}")
    print(f"quality     {report['quality']}")
    print(f"pages       {report['page_count']}")
    print(f"outline     {report['outline_source']}")
    print(f"chunks      {report['chunk_count']}")
    if report["image_heavy"]:
        print("image_heavy true")
    if report["components"]:
        print("components")
        for name, value in report["components"].items():
            print(f"  {name:<20} {value}")
    if report["escalation"]:
        print("escalation")
        print(f"  requested            {report['escalation']['requested']}")
        print(f"  trigger              {report['escalation']['trigger']}")
        print(f"  reason               {report['escalation']['reason']}")
    if report["error"]:
        print(f"error       {report['error']}")
    if show_chunks:
        print("---")
        for item in chunks:
            print(f"[{item.chunk_id}] {item.section_path or '(no section)'} ({item.language})")
            print(_preview(item.text))
            print()


def _parse_command(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(args.config)
    except (ConfigError, OSError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    outcome = build_router(cfg).parse(args.path)
    chunks = ()
    gate = None
    if outcome.document is not None:
        chunks = chunk(outcome.document, outcome.document.path, cfg.chunking)
        if outcome.components is not None:
            gate = decide(
                outcome.components,
                cfg.quality,
                image_count=outcome.document.image_count,
                script_validity=document_script_validity(outcome.document),
            )

    report = _report(outcome, chunks, gate, args.path, args.chunks)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_human(report, chunks, args.chunks)
    return 0 if outcome.status in {"ok", "degraded"} else 1


def _index_command(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(args.config)
    except (ConfigError, OSError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    from pathlib import Path

    Path(args.state).parent.mkdir(parents=True, exist_ok=True)
    pipeline = build_pipeline(cfg, args.state)

    last = None
    for event in pipeline.run(args.folder):
        last = event
        done = event.profiled + event.failed
        print(
            f"\r[{done}/{event.total}] "
            f"parsed·{event.parsed} embedded·{event.embedded} "
            f"profiled·{event.profiled} failed·{event.failed}",
            end="",
            flush=True,
        )
    print()
    if last is None:
        print("nothing to index (every file is already current)")
        return 0
    if last.failed == last.total and last.total > 0:
        print("every file failed", file=sys.stderr)
        return 1
    return 0


def _search_command(args: argparse.Namespace) -> int:
    from porsuk.core.models import Filters

    try:
        cfg = load_config(args.config)
    except (ConfigError, OSError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if cfg.store.provider not in _PERSISTENT_STORES:
        print(
            f"warning: store.provider {cfg.store.provider!r} (e.g. inmemory) does not "
            "persist between `porsuk index` and `porsuk search` runs, so search sees an "
            "empty index. Use a persistent qdrant store (see config/vllm.yaml) for real use.",
            file=sys.stderr,
        )

    k = args.k if args.k is not None else cfg.retrieval.chunk_k
    retriever = build_retriever(cfg)
    filters = Filters(languages=(args.lang,) if args.lang else ())
    hits = retriever.search(args.query, k=k, filters=filters, strategy=args.strategy)

    if args.json:
        print(json.dumps({"hits": [_hit_dict(h) for h in hits]}, ensure_ascii=False, indent=2))
        return 0

    if not hits:
        print("no results")
    for i, h in enumerate(hits, 1):
        score = h.rerank_score if h.rerank_score is not None else h.score
        print(f"[{i}] {score:.3f}  {h.chunk.section_path or '(no section)'}  ({h.chunk.language})")
        print(_preview(h.chunk.text))
        print()
    return 0


def _ask_command(args: argparse.Namespace) -> int:
    from porsuk.core.container import build_agent_runner

    try:
        cfg = load_config(args.config)
    except (ConfigError, OSError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if cfg.store.provider not in _PERSISTENT_STORES:
        print(
            f"warning: store.provider {cfg.store.provider!r} (e.g. inmemory) does not "
            "persist between `porsuk index` and `porsuk ask` runs, so the agent sees an "
            "empty index. Use a persistent qdrant store (see config/vllm.yaml) for real use.",
            file=sys.stderr,
        )

    run = build_agent_runner(cfg)
    answer = run(args.question, lang=args.lang, verbose=args.verbose)

    if args.json:
        print(json.dumps(dataclasses.asdict(answer), ensure_ascii=False, indent=2))
        return 0

    # Char class must match run.py's _MARK: a real chunk_id is `<uuid>:<5 digits>`.
    display_text = re.sub(r"\s*⟦[A-Za-z0-9_:.\-]+⟧", "", answer.text)
    print(f"CEVAP: {display_text}\n")
    print("KAYNAKLAR (agent'ın atıf verdiği):")
    if answer.sources:
        for i, s in enumerate(answer.sources, 1):
            loc = " · ".join(
                part
                for part in (
                    s.file,
                    f"s.{s.page}" if s.page is not None else None,
                    f'"{s.section}"' if s.section else None,
                )
                if part
            )
            print(f"  {i}. {loc}")
            if s.quote:
                print(f'     "{s.quote}"')
    else:
        print("  (yok)")

    print()
    print("BAKILAN DOKÜMANLAR (döngüde dokunulan):")
    if answer.documents_seen:
        for s in answer.documents_seen:
            loc = " · ".join(
                part for part in (s.file, f"s.{s.page}" if s.page is not None else None) if part
            )
            print(f"  · {loc}")
    else:
        print("  (yok)")

    if answer.truncated:
        print("\n[adım limiti aşıldı — cevap eldeki bilgiyle verildi]", file=sys.stderr)
    return 0


def _serve_command(args: argparse.Namespace) -> int:
    """Run the FastAPI app. `--config` becomes PORSUK_CONFIG, which
    the app's lifespan reads; PORSUK_API_KEY must already be in the environment
    or the app refuses to start."""
    import os

    import uvicorn

    if not os.environ.get("PORSUK_API_KEY"):
        print("PORSUK_API_KEY is not set; the API refuses to start without it", file=sys.stderr)
        return 2

    os.environ["PORSUK_CONFIG"] = args.config
    uvicorn.run("porsuk.api.server:app", host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "parse":
        return _parse_command(args)
    if args.command == "index":
        return _index_command(args)
    if args.command == "search":
        return _search_command(args)
    if args.command == "ask":
        return _ask_command(args)
    if args.command == "serve":
        return _serve_command(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
