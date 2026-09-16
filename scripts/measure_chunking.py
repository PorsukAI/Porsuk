"""Measure the semantic chunker's article-recall and misattribution rate
against real documents (`calibrate_quality` measures the parse quality gate;
this measures the chunker).

Two numbers, and the second is the one that matters:

  * **Recall** - how many of the article markers actually present in the text
    became sections. A miss costs granularity.
  * **Misattribution** - how many chunks quote one article while their
    `section_path` names a different one. A miss here produces a citation
    that is wrong: the user is shown a heading their answer did not come
    from.

    uv run python -m scripts.measure_chunking --root tests/fixtures/downloaded
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from porsuk.core.config import load_config
from porsuk.core.container import build_router
from porsuk.ingestion.chunking import chunk

# The marker as it appears in body text, wherever on the line it starts.
_MARKER = re.compile(r"\bMADDE\s+(\d+)", re.IGNORECASE)


def _markers(text: str) -> set[str]:
    return {m.group(1) for m in _MARKER.finditer(text)}


def measure(root: Path, config: str) -> list[dict]:
    cfg = load_config(config, env={})
    router = build_router(cfg)
    rows = []
    for path in sorted(root.rglob("*.pdf")):
        outcome = router.parse(str(path))
        if outcome.document is None:
            continue
        full = "\n".join(b.text for b in outcome.document.blocks)
        present = _markers(full)
        chunks = chunk(outcome.document, outcome.document.path, cfg.chunking)
        detected = {
            m.group(1) for c in chunks if c.section_path for m in _MARKER.finditer(c.section_path)
        }
        # A chunk is misattributed when its body quotes exactly one article and
        # its path names a different one. Restricting to one keeps the measure
        # conservative: a chunk spanning two articles is ambiguous, not wrong.
        wrong = 0
        for c in chunks:
            in_body = _markers(c.text)
            in_path = _markers(c.section_path or "")
            if len(in_body) == 1 and in_path and not (in_body & in_path):
                wrong += 1
        rows.append(
            {
                "file": path.name,
                "articles_in_text": len(present),
                "articles_sectioned": len(detected & present),
                "recall": len(detected & present) / len(present) if present else None,
                "chunks": len(chunks),
                "misattributed": wrong,
            }
        )
    return rows


def _table(rows: list[dict]) -> str:
    header = "| file | articles in text | sectioned | recall | chunks | misattributed |"
    lines = [header, "|---|---|---|---|---|---|"]
    for r in rows:
        recall = "-" if r["recall"] is None else f"{r['recall']:.0%}"
        lines.append(
            f"| {r['file']} | {r['articles_in_text']} | {r['articles_sectioned']} | "
            f"{recall} | {r['chunks']} | {r['misattributed']} |"
        )
    total_present = sum(r["articles_in_text"] for r in rows)
    total_found = sum(r["articles_sectioned"] for r in rows)
    total_wrong = sum(r["misattributed"] for r in rows)
    total_chunks = sum(r["chunks"] for r in rows)
    overall = f"{total_found / total_present:.0%}" if total_present else "-"
    lines.append(
        f"| **total** | **{total_present}** | **{total_found}** | **{overall}** | "
        f"**{total_chunks}** | **{total_wrong}** |"
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(prog="measure_chunking")
    parser.add_argument("--root", default="tests/fixtures/downloaded")
    parser.add_argument("--config", default="config/local.yaml")
    args = parser.parse_args()
    root = Path(args.root)
    if not root.exists():
        print(f"no corpus at {root}; fetch it with scripts.fixtures.fetch")
        return 1
    print(_table(measure(root, args.config)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
