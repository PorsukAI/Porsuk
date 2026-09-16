"""Placeholders for the OCR/Docling escalation targets on a bare install
(without the `parsers` extra): each fills its slot in the parser chain with a
failure naming what is missing, so the chain still runs end to end.
"""

from __future__ import annotations

from porsuk.core.models import ParseOutcome


class UnavailableParser:
    """Declares a slot in the chain and fails with a reason."""

    name = "unavailable"
    cost = 1000
    suffixes: tuple[str, ...] = ()
    reason = "not implemented"

    def can_handle(self, path: str) -> bool:
        return path.lower().endswith(self.suffixes)

    def parse(self, path: str) -> ParseOutcome:
        return ParseOutcome(
            document=None,
            status="failed",
            quality=0.0,
            parser_used=self.name,
            error=f"{self.name} parser is not available: {self.reason}",
        )


class OCRParser(UnavailableParser):
    name = "ocr"
    cost = 100
    suffixes = (".pdf",)
    reason = "install the 'parsers' extra for RapidOCR (uv sync --extra parsers)"


class DoclingParser(UnavailableParser):
    name = "docling"
    cost = 200
    suffixes = (".pdf",)
    reason = (
        "install the 'parsers' extra for Docling (uv sync --extra parsers); "
        "it pulls torch into the tree, which is why it is optional (spec 11)"
    )
