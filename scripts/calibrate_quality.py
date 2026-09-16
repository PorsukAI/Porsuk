"""Measure the parse quality gate and chunking across both corpora, printing
one markdown table per document's components/decision and one for chunk
counts and section paths.

Run against the generated fixtures alone:

    uv run python -m scripts.calibrate_quality

and against real documents once `scripts/fixtures/fetch.py` has downloaded
them:

    uv run python -m scripts.fixtures.fetch
    uv run python -m scripts.calibrate_quality --root tests/fixtures/downloaded
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from porsuk.core.config import Config, load_config
from porsuk.core.container import build_router
from porsuk.ingestion.chunking import chunk as chunk_document
from porsuk.ingestion.quality import decide, document_script_validity
from porsuk.ingestion.router import ParserRouter

_SUFFIXES = {".pdf", ".docx", ".xlsx", ".pptx", ".txt", ".md", ".html"}


def measure_corpus(router: ParserRouter, root: Path, cfg: Config) -> list[dict[str, Any]]:
    """One row per document: every raw component, and the decision that
    followed.

    `cfg` is passed in rather than reloaded from a fixed path. Loading
    config/local.yaml here while `main` built the router from `--config` meant
    a run against another profile produced rows whose `quality` column came
    from one profile and whose `escalate_to`/`trigger` columns came from
    another, invisible only because the two shipped profiles happen to agree.
    """
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _SUFFIXES:
            continue
        outcome = router.parse(str(path))
        components = outcome.components
        gate = None
        chunks: tuple[Any, ...] = ()
        if components is not None and outcome.document is not None:
            gate = decide(
                components,
                cfg.quality,
                image_count=outcome.document.image_count,
                script_validity=document_script_validity(outcome.document),
            )
            chunks = chunk_document(outcome.document, path.stem, cfg.chunking)
        rows.append(
            {
                "path": str(path),
                "category": path.parent.name,
                "parser_used": outcome.parser_used,
                "status": outcome.status,
                "quality": round(outcome.quality, 3),
                "pages": outcome.document.page_count if outcome.document else None,
                "chars_per_page": round(components.chars_per_page, 1) if components else None,
                "bad_char_ratio": round(components.bad_char_ratio, 4) if components else None,
                "text_plausibility": (
                    round(components.text_plausibility, 3) if components else None
                ),
                "empty_table_blocks": components.empty_table_blocks if components else None,
                "escalate_to": gate.escalate_to if gate else None,
                "trigger": gate.trigger if gate else None,
                "chunk_count": len(chunks),
                # Distinct paths in document order. A repeated path means one
                # section split across several chunks, which is a size
                # decision, not a sectioning one.
                "section_paths": tuple(dict.fromkeys(c.section_path or "" for c in chunks)),
            }
        )
    return rows


_COLUMNS = (
    "category",
    "parser_used",
    "status",
    "quality",
    "pages",
    "chars_per_page",
    "bad_char_ratio",
    "text_plausibility",
    "empty_table_blocks",
    "escalate_to",
    "trigger",
    "chunk_count",
)


def as_markdown(rows: list[dict[str, Any]]) -> str:
    header = "| file | " + " | ".join(_COLUMNS) + " |"
    divider = "|---" * (len(_COLUMNS) + 1) + "|"
    lines = [header, divider]
    for row in rows:
        name = Path(row["path"]).name
        cells = ["" if row[c] is None else str(row[c]) for c in _COLUMNS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def sections_as_markdown(rows: list[dict[str, Any]], *, limit: int = 12) -> str:
    """Chunk count and the section paths tier 1/2/3 actually produced.

    Truncated per document because a long law yields one path per MADDE and
    the point of the listing is the *shape* of the sectioning, not a full
    index of it.
    """
    lines = [f"| file | chunks | section paths (first {limit} distinct) |", "|---|---|---|"]
    for row in rows:
        if row["status"] == "failed":
            continue
        paths = row["section_paths"]
        shown = ", ".join(f"`{p}`" if p else "_(none)_" for p in paths[:limit])
        if len(paths) > limit:
            shown += f", … (+{len(paths) - limit} more)"
        lines.append(f"| {Path(row['path']).name} | {row['chunk_count']} | {shown or '—'} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the parse quality gate.")
    parser.add_argument("--root", type=Path, default=Path("tests/fixtures/generated"))
    parser.add_argument("--config", default="config/local.yaml")
    args = parser.parse_args()

    if args.root == Path("tests/fixtures/generated"):
        from scripts.fixtures.generate import generate_fixtures

        generate_fixtures(args.root)

    cfg = load_config(args.config, env={})
    rows = measure_corpus(build_router(cfg), args.root, cfg)
    print(as_markdown(rows))
    print()
    print(sections_as_markdown(rows))


if __name__ == "__main__":
    main()
