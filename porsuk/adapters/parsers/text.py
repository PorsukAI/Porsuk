"""Plain text, Markdown and HTML.

trafilatura is a known option for HTML. It is deliberately not used here: the
fixture corpus contains no HTML, so its benefit cannot be measured, and the
project's rule is to measure before adding. `html.parser` from the standard
library covers the corpus that exists. When a bulk fetch brings real
HTML, trafilatura drops in as a second parser at a higher cost and the gate
escalates to it.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

from porsuk.core.models import Block, Outline, OutlineNode, ParsedDocument, ParseOutcome
from porsuk.core.registry import register

_NAME = "text"
_SUFFIXES = {".txt", ".md", ".markdown", ".html", ".htm"}
_HTML_SUFFIXES = {".html", ".htm"}
_MARKDOWN_SUFFIXES = {".md", ".markdown"}
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def _failed(error: str) -> ParseOutcome:
    return ParseOutcome(document=None, status="failed", quality=0.0, parser_used=_NAME, error=error)


def _nest(entries: list[tuple[str, int]]) -> tuple[OutlineNode, ...]:
    roots: list[dict] = []
    stack: list[dict] = []
    for title, level in entries:
        node = {"title": title, "level": level, "children": []}
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
            page_no=None,
            children=tuple(build(c) for c in raw["children"]),
        )

    return tuple(build(r) for r in roots)


class _HTMLExtractor(HTMLParser):
    """Text and headings out of HTML, with script and style contents dropped.

    Without the skip set, `var x = 1;` from a <script> becomes document text
    and drags text_plausibility down on a page that is perfectly fine.
    """

    # <head> and its children are metadata, not body text. Without `title`
    # here, a page's title arrived glued to the front of its first block.
    _SKIP = {"script", "style", "noscript", "head", "title", "meta", "link"}

    # Tags after which running text cannot continue. Table cells are flush
    # points, or a row's cells concatenate into one unsearchable token.
    _FLUSH = {"p", "div", "li", "br", "td", "th", "tr", "table", "section", "article"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[tuple[str, str, int | None]] = []
        self._skip_depth = 0
        self._heading_level: int | None = None
        self._buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag == "br":
            # A void element: `handle_endtag` never fires for it, so flushing
            # only on the end tag glued the text either side of every <br>.
            self._flush()
        elif re.fullmatch(r"h[1-6]", tag):
            self._flush()
            self._heading_level = int(tag[1])

    def handle_startendtag(self, tag: str, attrs) -> None:
        """Explicitly self-closed forms: <br/> as well as <br>."""
        if tag == "br":
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif re.fullmatch(r"h[1-6]", tag) or tag in self._FLUSH:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._buffer.append(data)

    def _flush(self) -> None:
        text = " ".join("".join(self._buffer).split())
        self._buffer.clear()
        if not text:
            self._heading_level = None
            return
        if self._heading_level is not None:
            self.blocks.append(("heading", text, self._heading_level))
            self._heading_level = None
        else:
            self.blocks.append(("text", text, None))

    def close(self) -> None:
        super().close()
        self._flush()


def _read(path: Path) -> tuple[str, str]:
    """Decode as UTF-8, then cp1254, and only then Latin-1, saying which.

    `utf-8-sig` rather than `utf-8` because a Windows-authored file carries a
    BOM, and U+FEFF left at the front of the text defeats every anchored
    pattern downstream: a BOM'd Markdown file lost its outline entirely,
    because `_MD_HEADING` cannot match `\ufeff#`.

    cp1254 before Latin-1 because this is a Turkish corpus and cp1254 is the
    legacy encoding it will actually meet. The two differ in precisely the six
    letters Turkish adds, so decoding cp1254 bytes as Latin-1 turns `Sağlık`
    into `Saðlýk`, mojibake in which every character is individually valid,
    so `bad_char_ratio` sees nothing and only `script_validity` can. Latin-1
    stays as the final fallback because it never raises, which is what
    guarantees this function always returns.

    Recording which encoding was used keeps the choice diagnosable rather than
    silent.
    """
    raw = path.read_bytes()
    try:
        # Reported as plain "utf-8": stripping a BOM is not a different
        # decoding, and the label is a diagnostic, not a codec name.
        return raw.decode("utf-8-sig"), "utf-8"
    except UnicodeDecodeError:
        pass
    try:
        return raw.decode("cp1254"), "cp1254"
    except UnicodeDecodeError:
        return raw.decode("latin-1"), "latin-1"


@register("parser", _NAME)
class TextParser:
    name = _NAME
    cost = 5

    def can_handle(self, path: str) -> bool:
        return Path(path).suffix.lower() in _SUFFIXES

    def parse(self, path: str) -> ParseOutcome:
        try:
            document = self._parse(Path(path))
        except Exception as exc:  # noqa: BLE001
            return _failed(f"{type(exc).__name__}: {exc}")
        return ParseOutcome(
            document=document, status="ok", quality=0.0, parser_used=_NAME, error=None
        )

    def _parse(self, path: Path) -> ParsedDocument:
        text, encoding = _read(path)
        suffix = path.suffix.lower()
        if suffix in _HTML_SUFFIXES:
            blocks, headings = self._from_html(text)
        elif suffix in _MARKDOWN_SUFFIXES:
            blocks, headings = self._from_markdown(text)
        else:
            blocks, headings = self._from_plain(text)
        return ParsedDocument(
            path=str(path),
            doc_type=suffix.lstrip("."),
            blocks=tuple(blocks),
            outline=(
                Outline(nodes=_nest(headings), source="toc") if headings else Outline(source="none")
            ),
            page_count=1,
            metadata={"encoding": encoding},
        )

    @staticmethod
    def _from_plain(text: str) -> tuple[list[Block], list[tuple[str, int]]]:
        blocks = [
            Block(text=part.strip(), page_no=1, kind="text")
            for part in re.split(r"\n\s*\n", text)
            if part.strip()
        ]
        return blocks, []

    @staticmethod
    def _from_markdown(text: str) -> tuple[list[Block], list[tuple[str, int]]]:
        blocks: list[Block] = []
        headings: list[tuple[str, int]] = []
        buffer: list[str] = []

        def flush() -> None:
            body = "\n".join(buffer).strip()
            buffer.clear()
            if body:
                blocks.append(Block(text=body, page_no=1, kind="text"))

        for line in text.splitlines():
            match = _MD_HEADING.match(line)
            if match:
                flush()
                title, level = match.group(2), len(match.group(1))
                headings.append((title, level))
                blocks.append(Block(text=title, page_no=1, kind="heading", bold=True))
            else:
                buffer.append(line)
        flush()
        return blocks, headings

    @staticmethod
    def _from_html(text: str) -> tuple[list[Block], list[tuple[str, int]]]:
        extractor = _HTMLExtractor()
        extractor.feed(text)
        extractor.close()
        blocks: list[Block] = []
        headings: list[tuple[str, int]] = []
        for kind, body, level in extractor.blocks:
            if kind == "heading" and level is not None:
                headings.append((body, level))
                blocks.append(Block(text=body, page_no=1, kind="heading", bold=True))
            else:
                blocks.append(Block(text=body, page_no=1, kind="text"))
        return blocks, headings
