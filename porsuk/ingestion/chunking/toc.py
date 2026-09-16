"""Tier 1 chunking: section boundaries from a real outline (`Outline(source=
"toc")`). Since a PDF outline entry carries a title and page number but never
a character offset, an entry claims blocks from the first block whose text
starts with its title, searched from its page onward, until the next entry's
start.
"""

from __future__ import annotations

from porsuk.core.models import Block, OutlineNode, ParsedDocument
from porsuk.ingestion.chunking.splitter import PATH_JOIN, Section


def _flatten(
    nodes: tuple[OutlineNode, ...], ancestors: tuple[str, ...] = ()
) -> list[tuple[OutlineNode, str]]:
    """Depth-first, each entry paired with its full path."""
    out: list[tuple[OutlineNode, str]] = []
    for node in nodes:
        path = (*ancestors, node.title)
        out.append((node, PATH_JOIN.join(path)))
        out.extend(_flatten(node.children, path))
    return out


def _find(blocks: tuple[Block, ...], title: str, from_page: int | None, start: int) -> int | None:
    """Index of the block that opens `title`, or None.

    Matching is prefix-based on stripped text: a heading block usually holds
    exactly the title, but PDF extraction often glues the first body words on.
    """
    needle = title.strip().casefold()
    if not needle:
        return None
    for index in range(start, len(blocks)):
        block = blocks[index]
        if from_page is not None and block.page_no < from_page:
            continue
        if block.text.strip().casefold().startswith(needle):
            return index
    return None


def sections_from_outline(document: ParsedDocument) -> list[Section] | None:
    if document.outline.source != "toc" or not document.outline.nodes:
        return None
    if not document.blocks:
        return None

    entries = _flatten(document.outline.nodes)
    blocks = document.blocks

    # Locate every entry first, so a section can end where the next one begins.
    located: list[tuple[int, OutlineNode, str]] = []
    cursor = 0
    for node, path in entries:
        index = _find(blocks, node.title, node.page_no, cursor)
        if index is None:
            # The entry stays in descendants' paths through `_flatten`; it just
            # claims no blocks. Outlines naming sections whose heading is an
            # image are common enough that dropping the tier would be wrong.
            continue
        located.append((index, node, path))
        cursor = index + 1

    if not located:
        return None

    sections: list[Section] = []
    first = located[0][0]
    if first > 0:
        # A cover page, or anything else before the first entry. Untitled
        # rather than folded into section one, which it is not part of.
        sections.append(Section(title=None, path=None, blocks=blocks[:first]))

    for position, (index, node, path) in enumerate(located):
        end = located[position + 1][0] if position + 1 < len(located) else len(blocks)
        sections.append(Section(title=node.title, path=path, blocks=blocks[index:end]))
    return sections
