"""Tests for the plain-text/Markdown/HTML parser adapter."""

import pytest

from porsuk.adapters.parsers.text import TextParser
from porsuk.core.ports import Parser

TURKISH = "Ödemeler fatura tarihinden itibaren 30 gün içinde yapılır."


@pytest.fixture
def parser():
    return TextParser()


def test_conforms_to_the_parser_protocol(parser):
    assert isinstance(parser, Parser)


def test_is_the_cheapest_parser_in_the_chain(parser):
    """The parser chain is ordered by cost; reading a text file is the least
    work any parser does."""
    from porsuk.adapters.parsers.pymupdf import PyMuPDFParser

    assert parser.cost < PyMuPDFParser().cost


@pytest.mark.parametrize("suffix", [".txt", ".md", ".html", ".htm", ".TXT"])
def test_handles_text_formats(parser, suffix):
    assert parser.can_handle(f"/tmp/a{suffix}")


@pytest.mark.parametrize("suffix", [".pdf", ".docx", ".xlsx"])
def test_rejects_binary_formats(parser, suffix):
    assert not parser.can_handle(f"/tmp/a{suffix}")


def test_plain_text_becomes_paragraph_blocks(parser, tmp_path):
    path = tmp_path / "a.txt"
    path.write_text(f"{TURKISH}\n\nİkinci paragraf burada.\n", encoding="utf-8")
    outcome = parser.parse(str(path))
    assert outcome.status == "ok"
    assert len(outcome.document.blocks) == 2
    assert outcome.document.outline.source == "none"


def test_markdown_headings_become_an_outline(parser, tmp_path):
    """Tier 1: Markdown headings are real structure."""
    path = tmp_path / "a.md"
    path.write_text(
        f"# Sözleşme\n\n## 1. Taraflar\n\n{TURKISH}\n\n## 2. Mali Hükümler\n\nMetin.\n",
        encoding="utf-8",
    )
    outcome = parser.parse(str(path))
    outline = outcome.document.outline
    assert outline.source == "toc"
    assert [n.title for n in outline.nodes] == ["Sözleşme"]
    assert [c.title for c in outline.nodes[0].children] == ["1. Taraflar", "2. Mali Hükümler"]


def test_html_tags_are_stripped_and_headings_kept(parser, tmp_path):
    path = tmp_path / "a.html"
    path.write_text(
        f"<html><body><h1>Sözleşme</h1><p>{TURKISH}</p><script>var x = 1;</script></body></html>",
        encoding="utf-8",
    )
    outcome = parser.parse(str(path))
    text = "\n".join(b.text for b in outcome.document.blocks)
    assert "Ödemeler" in text
    assert "<p>" not in text
    assert "var x" not in text, "script contents must not become document text"
    assert [n.title for n in outcome.document.outline.nodes] == ["Sözleşme"]


def test_non_utf8_content_is_decoded_without_raising(parser, tmp_path):
    """A file that is not valid UTF-8 must still parse. Guessing silently is
    worse than falling back visibly, so the fallback is recorded in metadata.

    cp1254 is tried before Latin-1 because this is a Turkish corpus; the two
    agree on every character in this fixture, so what is asserted here is that
    the fallback happened and was recorded, not which of the two won."""
    path = tmp_path / "a.txt"
    path.write_bytes("Ödemeler yapılır.".encode("latin-1", errors="replace"))
    outcome = parser.parse(str(path))
    assert outcome.status == "ok"
    assert outcome.document.metadata.get("encoding") in {"cp1254", "latin-1"}
    assert "Ödemeler" in outcome.document.blocks[0].text


def test_bytes_undefined_in_cp1254_still_reach_the_latin1_fallback(parser, tmp_path):
    """0x8D is undefined in cp1254. Latin-1 stays the final fallback precisely
    because it never raises, which is what makes `_read` total."""
    path = tmp_path / "a.txt"
    path.write_bytes(b"Odemeler \x8d yapilir.")
    outcome = parser.parse(str(path))
    assert outcome.status == "ok"
    assert outcome.document.metadata.get("encoding") == "latin-1"


def test_utf8_content_records_its_encoding(parser, tmp_path):
    path = tmp_path / "a.txt"
    path.write_text(TURKISH, encoding="utf-8")
    outcome = parser.parse(str(path))
    assert outcome.document.metadata.get("encoding") == "utf-8"


def test_an_empty_file_parses_to_zero_blocks(parser, tmp_path):
    """Empty is not an error: the gate decides what to do about it."""
    path = tmp_path / "a.txt"
    path.write_text("", encoding="utf-8")
    outcome = parser.parse(str(path))
    assert outcome.status == "ok"
    assert outcome.document.blocks == ()


def test_a_missing_file_fails_without_raising(parser, tmp_path):
    outcome = parser.parse(str(tmp_path / "nope.txt"))
    assert outcome.status == "failed"
    assert outcome.error


def test_a_bom_does_not_destroy_markdown_structure(tmp_path):
    """A UTF-8 BOM leaks U+FEFF into the first block, and `_MD_HEADING` is
    anchored, so a BOM'd Windows-authored Markdown file loses its outline
    entirely and falls all the way to tier 3."""
    path = tmp_path / "bom.md"
    path.write_bytes("﻿# Tedarik Sözleşmesi\n\nBir paragraf.\n".encode())
    outcome = TextParser().parse(str(path))
    assert outcome.document.outline.source == "toc"
    assert "﻿" not in outcome.document.blocks[0].text


def test_turkish_legacy_encoding_is_decoded_as_cp1254(tmp_path):
    """Latin-1 never raises, so every non-UTF-8 file became Latin-1, and for
    Turkish that is precisely wrong: cp1254 differs from Latin-1 in exactly the
    six Turkish letters, so `Sağlık` came back as `Saðlýk`. Every character is
    individually valid, so `bad_char_ratio` sees nothing and the damage is
    invisible to the gate."""
    path = tmp_path / "legacy.txt"
    path.write_bytes("Sağlık İstanbul için şirket ödemeleri.".encode("cp1254"))
    outcome = TextParser().parse(str(path))
    text = outcome.document.blocks[0].text
    assert "Sağlık" in text
    assert "şirket" in text


def test_html_does_not_glue_adjacent_words_together(parser, tmp_path):
    """`handle_endtag` never fires for a void <br/>, and table cells were not
    flush points, so 'Bir<br>iki' arrived as 'Biriki': unretrievable, and it
    depresses text_plausibility on a perfectly good page."""
    path = tmp_path / "a.html"
    path.write_text(
        "<html><head><title>Gizli Başlık</title></head><body>"
        "<table><tr><td>A</td><td>B</td></tr></table>"
        "<p>Bir<br>iki</p></body></html>",
        encoding="utf-8",
    )
    outcome = parser.parse(str(path))
    text = " ".join(b.text for b in outcome.document.blocks)
    assert "Biriki" not in text
    assert "AB" not in text
    assert "Bir" in text and "iki" in text


def test_html_title_does_not_leak_into_the_body(parser, tmp_path):
    """<head> content is not body text; it arrived glued to the first block."""
    path = tmp_path / "a.html"
    path.write_text(
        "<html><head><title>Gizli Başlık</title></head><body><p>Gövde metni.</p></body></html>",
        encoding="utf-8",
    )
    outcome = parser.parse(str(path))
    assert outcome.document.blocks[0].text.strip() == "Gövde metni."
