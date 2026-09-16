"""Complex-layout PDF parser (escalation target when a table/figure block was
detected but came back empty), run with full settings (table structure + OCR
on) and built lazily since `docling` pulls `torch` into the tree; falls back
to the `unavailable.py` stub without the `parsers` extra. Item mapping is
version-sensitive (pinned against docling 2.126.0 / docling-core 2.95.0) and
fails quietly rather than loudly on a mismatch, so check `item.label.value`
(not `str()`), `item.prov[0].page_no` (may be empty), and
`TableItem.export_to_dataframe(doc)` against the installed version before
upgrading docling.
"""

from __future__ import annotations

from pathlib import Path

from porsuk.core.registry import register

_NAME = "docling"
_COST = 200

try:
    import docling  # noqa: F401

    _AVAILABLE = True
except ImportError:  # pragma: no cover - the bare-install path
    _AVAILABLE = False


if _AVAILABLE:
    from porsuk.core.models import (
        Block,
        BlockKind,
        Outline,
        OutlineNode,
        ParsedDocument,
        ParseOutcome,
    )

    # Docling's label vocabulary is wider than ours. Anything unlisted is
    # body text: `Block.kind` has four values (the quality gate counts empty
    # *table* blocks specifically), and inventing a fifth here would not
    # reach the gate anyway.
    _HEADING_LABELS = frozenset({"title", "section_header", "page_header"})
    _TABLE_LABELS = frozenset({"table", "document_index"})
    _CAPTION_LABELS = frozenset({"caption"})

    def _failed(error: str) -> ParseOutcome:
        return ParseOutcome(
            document=None, status="failed", quality=0.0, parser_used=_NAME, error=error
        )

    def _label(item) -> str:
        label = getattr(item, "label", None)
        return getattr(label, "value", None) or str(label or "")

    def _kind(item) -> BlockKind:
        label = _label(item)
        if label in _TABLE_LABELS or type(item).__name__ == "TableItem":
            return "table"
        if label in _HEADING_LABELS:
            return "heading"
        if label in _CAPTION_LABELS:
            return "figure_caption"
        return "text"

    def _page_no(item) -> int:
        prov = getattr(item, "prov", None)
        if prov:
            return getattr(prov[0], "page_no", 1)
        return 1

    def _table_text(item, doc) -> str:
        """A table flattened to CSV.

        CSV rather than the source layout because a chunk is text: the column
        a value sits under has to survive as characters or it is lost to the
        embedder. Returns "" when Docling recognised a table but recovered no
        cells, which is exactly the empty table block the quality gate counts,
        so it must stay an empty string and not become an error.
        """
        try:
            return item.export_to_dataframe(doc).to_csv(index=False).strip()
        except Exception:  # noqa: BLE001 - a table we cannot flatten is an empty one
            return ""

    def _blocks(doc) -> tuple[list[Block], list[OutlineNode]]:
        blocks: list[Block] = []
        outline: list[OutlineNode] = []
        for item, level in doc.iterate_items():
            kind = _kind(item)
            text = (getattr(item, "text", "") or "").strip()
            if kind == "table":
                text = _table_text(item, doc)
                # Kept even when empty: an empty table block is the signal
                # the gate counts, and dropping it here would erase it.
                blocks.append(Block(text=text, page_no=_page_no(item), kind=kind))
                continue
            if not text:
                continue
            blocks.append(Block(text=text, page_no=_page_no(item), kind=kind))
            if kind == "heading":
                outline.append(OutlineNode(title=text, level=max(1, level), page_no=_page_no(item)))
        return blocks, outline

    @register("parser", _NAME)
    class DoclingParser:
        name = _NAME
        cost = _COST

        def __init__(self) -> None:
            self._converter = None

        def _build_converter(self):
            """Built on first use, not in __init__.

            `build_router` constructs every parser in the chain for every run,
            including the 4999 files that never escalate. Docling's converter
            loads layout and table-structure models, so doing it eagerly would
            put a torch model load into the startup path of a run that may
            never call this parser.
            """
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption

            options = PdfPipelineOptions()
            options.do_ocr = True
            options.do_table_structure = True
            return DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
            )

        def can_handle(self, path: str) -> bool:
            return Path(path).suffix.lower() == ".pdf"

        def parse(self, path: str) -> ParseOutcome:
            try:
                if self._converter is None:
                    self._converter = self._build_converter()
                result = self._converter.convert(path)
                doc = result.document
                blocks, outline = _blocks(doc)
            except Exception as exc:  # noqa: BLE001 - a Result, not an exception
                return _failed(f"{type(exc).__name__}: {exc}")

            if not blocks:
                return _failed("docling produced no content")

            document = ParsedDocument(
                path=path,
                doc_type="pdf",
                blocks=tuple(blocks),
                # "pattern", not "toc": Docling infers a heading hierarchy
                # from layout, it does not read an embedded table of contents
                # (a real TOC is authored and an inferred one is a guess).
                outline=Outline(nodes=tuple(outline), source="pattern"),
                page_count=len(getattr(doc, "pages", None) or {}) or 1,
            )
            # 0.0 deliberately - the router re-scores every document through
            # the gate; see ocr.py for why a parser must not grade itself.
            return ParseOutcome(
                document=document, status="ok", quality=0.0, parser_used=_NAME, error=None
            )

else:  # pragma: no cover - exercised only without the `parsers` extra
    # One registration site per name, real or stub. See ocr.py: two of them
    # would make the winner depend on the order ruff sorts the imports into.
    from porsuk.adapters.parsers.unavailable import DoclingParser as _DoclingStub

    DoclingParser = register("parser", _NAME)(_DoclingStub)
