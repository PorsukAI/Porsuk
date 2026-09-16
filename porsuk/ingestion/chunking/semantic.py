"""Tier 3: no structure to go on.

The fallback every document reaches when there is neither an embedded outline
nor a heading pattern. It deliberately invents no structure: a fabricated
section path would be worse than an absent one, because retrieval would show
the user a heading that is not in their document.

Paragraph boundaries are preserved by `split_text`; this module only decides
that the whole document is one untitled section.
"""

from __future__ import annotations

from porsuk.core.models import ParsedDocument
from porsuk.ingestion.chunking.splitter import Section


def sections_semantic(document: ParsedDocument) -> list[Section]:
    if not document.blocks:
        return []
    return [Section(title=None, path=None, blocks=document.blocks)]
