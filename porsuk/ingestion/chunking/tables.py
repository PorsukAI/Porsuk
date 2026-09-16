"""Table blocks become markdown and stay whole.

Markdown because it is the one table representation both an embedder and an
LLM read without extra machinery, and because it survives being pasted into an
answer as a citation.
"""

from __future__ import annotations


def table_to_markdown(text: str) -> str:
    """Pipe- or tab-separated rows into a markdown table.

    The first row is treated as the header. Ragged rows are padded rather than
    dropped: a table that lost a cell is still worth indexing, and dropping
    rows silently would make the chunk lie about the document.
    """
    rows: list[list[str]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        separator = "\t" if "\t" in line else "|"
        rows.append([cell.strip() for cell in line.split(separator)])
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    header, *body = padded
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)
