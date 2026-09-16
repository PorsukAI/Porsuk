"""PDF primary parser.

PyMuPDF is first in the chain because it is 10-30x faster than Docling and
enough for most text-layer PDFs; over 5000 files that gap is hours. It never
decides a document needs more: that is the quality gate's job.

This module may import pymupdf: `tests/test_architecture.py` bans provider
libraries in core/, ingestion/ and retrieval/, and adapters are where they
belong.
"""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf

from porsuk.core.models import (
    Block,
    Outline,
    OutlineNode,
    ParsedDocument,
    ParseOutcome,
)
from porsuk.core.registry import register

_NAME = "pymupdf"

# A line carrying two or more cell separators reads as tabular. Deliberately
# crude: the gate only needs to know a table block was detected, and whether it
# came back empty. Real table structure is Docling's job.
#
# Note: not `re.compile(r"(\S\s*[|\t]\s*\S){2,}")`: that quantifier requires
# two *consecutive* separator matches with nothing in between, which a line
# like "Fatura No | Tutar | Vade" does not have (the second separator region
# starts mid-word, after "utar"). Counting non-overlapping hits instead is the
# form that actually matches real "a | b | c"-shaped lines.
_TABLE_SEPARATOR = re.compile(r"\S\s*[|\t]\s*\S")


def _is_tabular(text: str) -> bool:
    return any(len(_TABLE_SEPARATOR.findall(line)) >= 2 for line in text.splitlines())


def _failed(error: str) -> ParseOutcome:
    return ParseOutcome(document=None, status="failed", quality=0.0, parser_used=_NAME, error=error)


def _nest(entries: list[list]) -> tuple[OutlineNode, ...]:
    """Turn PyMuPDF's flat [level, title, page] list into a tree.

    Levels are 1-based and may skip; a child deeper than its predecessor
    attaches to the nearest shallower node, which is what get_toc's flat form
    encodes.
    """
    roots: list[dict] = []
    stack: list[dict] = []
    for level, title, page_no in entries:
        node = {"title": title, "level": level, "page_no": page_no, "children": []}
        while stack and stack[-1]["level"] >= level:
            stack.pop()
        if stack:
            stack[-1]["children"].append(node)
        else:
            roots.append(node)
        stack.append(node)

    def build(raw: dict) -> OutlineNode:
        return OutlineNode(
            title=raw["title"],
            level=raw["level"],
            page_no=raw["page_no"],
            children=tuple(build(c) for c in raw["children"]),
        )

    return tuple(build(r) for r in roots)


@register("parser", _NAME)
class PyMuPDFParser:
    name = _NAME
    cost = 10

    def can_handle(self, path: str) -> bool:
        return Path(path).suffix.lower() == ".pdf"

    def parse(self, path: str) -> ParseOutcome:
        try:
            document = self._parse(path)
        except Exception as exc:  # noqa: BLE001 - a Result, not an exception
            return _failed(f"{type(exc).__name__}: {exc}")
        return ParseOutcome(
            document=document, status="ok", quality=0.0, parser_used=_NAME, error=None
        )

    def _parse(self, path: str) -> ParsedDocument:
        with pymupdf.open(path) as doc:
            if doc.needs_pass:
                raise ValueError("PDF is password protected")
            blocks: list[Block] = []
            image_count = 0
            for page_index, page in enumerate(doc, start=1):
                image_count += len(page.get_images(full=True))
                # sort=True: MuPDF otherwise returns content-stream order,
                # which is usually but not always reading order. `Block`
                # promises reading order and the chunker relies on it: a
                # two-column layout can interleave without this. Measured as
                # a no-op on the current corpus (identical gate and chunking
                # tables), so it costs nothing to be correct here.
                for raw in page.get_text("blocks", sort=True):
                    text = raw[4].strip()
                    if not text:
                        continue
                    kind = "table" if _is_tabular(text) else "text"
                    blocks.append(Block(text=text, page_no=page_index, kind=kind))
            toc = doc.get_toc()
            outline = Outline(nodes=_nest(toc), source="toc") if toc else Outline(source="none")
            return ParsedDocument(
                path=path,
                doc_type="pdf",
                blocks=tuple(blocks),
                outline=outline,
                page_count=doc.page_count,
                image_count=image_count,
            )
