"""Generate the fixture corpus deterministically and offline: CI needs no
network, the repo stays small, and there are no licensing questions about
redistributing someone else's documents.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf as fitz

# Fixture text uses real Turkish orthography, which requires an embedded
# Unicode font: PyMuPDF's default base-14 Helvetica is Latin-1 and has no
# glyphs for ş/ğ/ı/İ, so those round-trip out of a generated PDF as "?".
# `notos` (Noto Sans, SIL OFL 1.1) ships with the pymupdf-fonts dev
# dependency and covers Turkish fully. This matters because the parse
# quality gate scores `text_plausibility` on Turkish + English function
# words and alphabet share (not a dictionary, agglutination makes one
# useless): ASCII-fied text would make `text_pdf` (the "PyMuPDF happy path"
# category) score as damaged under the metric that is supposed to treat it
# as clean, inverting any weight calibrated against it.
#
# KNOWN GAP: `bad_char_ratio` counts "U+FFFD and control
# characters", but PyMuPDF's text API cannot round-trip U+FFFD:
# insert_textbox, insert_text and insert_htmlbox each re-map it (to "?",
# "\u00b7", "\u023c" and "\u037d" respectively). The U+FFFD round-trip itself
# remains impossible through the text API; a fixture for it would need a
# hand-built ToUnicode CMap.
#
# What `mojibake.pdf` adds is the *damage* U+FFFD would have stood for. A
# document that lost its encoding is the real-world case behind the
# replacement character, and mojibake reaches it from the other side: UTF-8
# read as Latin-1 produces text in which every character is individually
# valid, so `bad_char_ratio` sees nothing at all, and `script_validity`,
# the only signal that can, is what catches it. `control_chars.pdf` still
# exercises the half of `bad_char_ratio` that is reachable.
_FONT = "notos"

TURKISH_BODY = (
    "İşbu sözleşme, ABC Lojistik A.Ş. ile XYZ Sanayi Ltd. Şti. arasında "
    "tedarik hizmetlerinin sağlanması amacıyla akdedilmiştir. Ödemeler fatura "
    "tarihinden itibaren 30 gün içinde yapılır."
)
ENGLISH_BODY = (
    "This agreement is entered into between the parties for the provision of "
    "supply services. Payments shall be made within 30 days of the invoice date."
)


def _text_pdf(path: Path, pages: list[str], toc: list[list] | None = None) -> None:
    doc = fitz.open()
    for body in pages:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 550, 750), body, fontsize=11, fontname=_FONT)
    if toc:
        doc.set_toc(toc)
    doc.save(path)
    doc.close()


def _rasterise(source: Path, target: Path) -> None:
    """Render every page to an image and rebuild as an image-only PDF."""
    src = fitz.open(source)
    out = fitz.open()
    for page in src:
        pixmap = page.get_pixmap(dpi=110)
        new_page = out.new_page(width=page.rect.width, height=page.rect.height)
        new_page.insert_image(new_page.rect, pixmap=pixmap)
    out.save(target)
    out.close()
    src.close()


def _docx(path: Path) -> None:
    from docx import Document

    document = Document()
    document.add_heading("Tedarik Sözleşmesi", level=1)
    document.add_heading("1. Taraflar", level=2)
    document.add_paragraph(TURKISH_BODY)
    document.add_heading("2. Mali Hükümler", level=2)
    document.add_paragraph(TURKISH_BODY)
    document.save(path)


def _docx_with_table(path: Path) -> None:
    """A Word table, which is where corporate documents keep their numbers.

    python-docx's `paragraphs` property returns only top-level body
    paragraphs and never descends into tables, so a parser that iterates it
    drops this content entirely, silently, because no table block is emitted
    for the quality gate to count as empty.
    """
    from docx import Document

    document = Document()
    document.add_heading("Fatura Özeti", level=1)
    document.add_paragraph(TURKISH_BODY)
    table = document.add_table(rows=3, cols=3)
    for row, cells in enumerate(
        [
            ["Fatura No", "Açıklama", "Tutar"],
            ["F-001", "Nakliye bedeli", "12.500 TL"],
            ["F-002", "Depolama bedeli", "8.750 TL"],
        ]
    ):
        for column, value in enumerate(cells):
            table.cell(row, column).text = value
    document.add_paragraph("Ödemeler fatura tarihinden itibaren 30 gün içinde yapılır.")
    document.save(path)


def _xlsx(path: Path) -> None:
    from openpyxl import Workbook

    book = Workbook()
    first = book.active
    first.title = "Ödemeler"
    first.append(["Fatura No", "Tutar", "Vade"])
    first.append(["F-001", 15000, "2024-04-15"])
    second = book.create_sheet("Teslimat")
    second.append(["Tarih", "Miktar"])
    second.append(["2024-03-01", 120])
    book.save(path)


def _pptx(path: Path) -> None:
    from pptx import Presentation

    presentation = Presentation()
    for title, body in [("Tedarik Süreci", TURKISH_BODY), ("Supply Process", ENGLISH_BODY)]:
        slide = presentation.slides.add_slide(presentation.slide_layouts[1])
        slide.shapes.title.text = title
        slide.placeholders[1].text = body
    presentation.save(path)


def _pptx_with_table(path: Path) -> None:
    """A PowerPoint table lives in a GraphicFrame, which has no text frame.

    A parser that keeps only shapes where `has_text_frame` is true discards
    every table on every slide.
    """
    from pptx import Presentation
    from pptx.util import Inches

    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[5])
    slide.shapes.title.text = "Teslimat Kalemleri"
    shape = slide.shapes.add_table(3, 3, Inches(0.5), Inches(2.0), Inches(9.0), Inches(2.0))
    for row, cells in enumerate(
        [
            ["Kalem", "Miktar", "Tutar"],
            ["Nakliye", "120", "12.500 TL"],
            ["Depolama", "80", "8.750 TL"],
        ]
    ):
        for column, value in enumerate(cells):
            shape.table.cell(row, column).text = value
    presentation.save(path)


def _ruled_table_pdf(path: Path) -> None:
    """A table with drawn cell borders, not pipe-separated text.

    table_like.pdf is the cheap case: "a | b | c" lines that PyMuPDF's
    separator regex calls tabular and that read fine as plain text. Docling
    looks at it and sees prose, correctly - there is no visual structure to
    recover.

    This is the case Docling exists for: real cells, real rules,
    where the reading order of the text alone loses which value belongs to
    which column. Drawn with explicit rectangles so the structure is in the
    page, not merely implied by whitespace.
    """
    doc = fitz.open()
    page = doc.new_page()
    rows = [
        ["Fatura No", "Tutar", "Vade"],
        ["F-001", "15000", "2024-04-01"],
        ["F-002", "23750", "2024-05-15"],
        ["F-003", "9800", "2024-06-30"],
    ]
    left, top, cell_width, row_height = 60, 80, 140, 28
    page.insert_text((left, 60), "Odeme Tablosu", fontsize=14)
    for row_index, row in enumerate(rows):
        for column_index, cell in enumerate(row):
            rect = fitz.Rect(
                left + column_index * cell_width,
                top + row_index * row_height,
                left + (column_index + 1) * cell_width,
                top + (row_index + 1) * row_height,
            )
            page.draw_rect(rect, color=(0, 0, 0), width=0.8)
            page.insert_text((rect.x0 + 6, rect.y0 + 18), cell, fontsize=11)
    doc.save(path)
    doc.close()


def _damaged_text_pdf(path: Path) -> None:
    """Text carrying control characters, for the parse quality gate's `bad_char_ratio`.

    Real extraction damage shows up as replacement and control characters.
    Only the control-character half is reproducible here (see the module
    note), but it is the half that gives the component a signal at all.
    """
    damaged = (
        "Tedarik s\x01zle\x02mesi \x03 metin \x04 bozulmu\x05 durumda. "
        "\x06\x07 Ödemeler fatura tarihinden itibaren 30 gün içinde yapılır."
    )
    _text_pdf(path, [damaged])


def _scan_with_garbage_layer(source: Path, target: Path) -> None:
    """A rasterised page carrying a plausible bad OCR text layer.

    The plain rasterised scans carry exactly zero characters per
    page because we made them that way, so any threshold above zero passes
    them. Real scans usually carry a garbage text layer from a previous OCR
    pass, and calibrating the low-text branch needs that case to exist.

    The string is tuned, not decorative. `text_plausibility` has to read it
    as non-language with margin (measured 0.32 against the 0.4 ceiling
    `tests/test_fixtures.py` asserts); the shape it imitates is real OCR
    failure on Turkish: `rn` for `m`, `vv` for `w`, `c` for `e`, dotted/
    dotless `i` confusion, and stretches of repeated keystrokes.
    """
    garbage = (
        "Iıl ııı tlıc rn1 vvhh zzz Ödcmclcr fatuı-a tzırihindcn "
        "itibaı-cn 3O gtin içiııdc yapılıı- lll nnn kkk ııı lıl mmm ttt vvv"
    )
    src = fitz.open(source)
    out = fitz.open()
    for page in src:
        pixmap = page.get_pixmap(dpi=110)
        new_page = out.new_page(width=page.rect.width, height=page.rect.height)
        new_page.insert_image(new_page.rect, pixmap=pixmap)
        new_page.insert_textbox(fitz.Rect(50, 50, 550, 200), garbage, fontsize=9, fontname=_FONT)
    out.save(target)
    out.close()
    src.close()


def _mojibake_pdf(path: Path) -> None:
    """UTF-8 Turkish decoded as Latin-1.

    The damage `bad_char_ratio` cannot see: every resulting character is
    individually valid, so only `script_validity` catches it. This is the
    half of the KNOWN GAP that U+FFFD cannot reach through PyMuPDF's text API.

    The source sentence is restricted to Turkish letters whose UTF-8
    continuation byte is not itself a C1 control character. That excludes
    Ö, Ş, ş, ğ, Ğ, Ü, Ç and Â: a two-byte UTF-8 sequence carries
    `0x80 | (codepoint & 0x3F)` as its second byte, so any codepoint whose
    low six bits are under 0x20 mojibakes into U+0080-U+009F, which
    `unicodedata.category` reports as `Cc`. One "Ö" would therefore give this
    fixture a non-zero `bad_char_ratio` (measured 0.027 for a sentence
    opening on "Ödemeler") and destroy the only thing it exists to
    demonstrate: that mojibake is invisible to that component. The letters
    left in play (ç, ö, ü, ı, İ, â) are Turkish enough to carry the point.
    """
    original = (
        "İçindekiler bölümü üçüncü ölçüm çözümü: ödemeler için görüntüleme "
        "müdüriyeti üçüncü tarafın onayını içerir. ölçütü tanımlı olan çözüm türü."
    )
    mojibake = original.encode("utf-8").decode("latin-1")
    _text_pdf(path, [mojibake])


def _hyphenated_pdf(path: Path) -> None:
    """Turkish hyphenated across line breaks, parsed perfectly correctly.

    A negative control: if hyphenation alone trips the gate, clean mevzuat
    PDFs get escalated to OCR, which is the most expensive wrong answer the
    gate can give.
    """
    text = (
        "Ödemeler fatura tarihinden iti-\nbaren otuz gün içinde yapı-\nlır. "
        "Sözleşmenin feshi hâlin-\nde taraflar mutabık kalır."
    )
    _text_pdf(path, [text])


def _broken(directory: Path) -> None:
    (directory / "zero_byte.pdf").write_bytes(b"")
    (directory / "truncated.pdf").write_bytes(b"%PDF-1.7\n1 0 obj\n<< /Type /Catalog")

    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(50, 50, 550, 750), TURKISH_BODY, fontsize=11)
    doc.save(
        directory / "password_protected.pdf",
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="owner",
        user_pw="user",
    )
    doc.close()


def generate_fixtures(target: Path) -> dict[str, list[Path]]:
    """Create every fixture category under `target`. Idempotent."""
    categories = [
        "text_pdf",
        "outlined_pdf",
        "pattern_pdf",
        "table_pdf",
        "damaged_text_pdf",
        "scanned_pdf",
        "mixed_language_pdf",
        "docx",
        "xlsx",
        "pptx",
        "broken",
    ]
    for name in categories:
        (target / name).mkdir(parents=True, exist_ok=True)

    _text_pdf(target / "text_pdf" / "simple_tr.pdf", [TURKISH_BODY, TURKISH_BODY])
    _text_pdf(target / "text_pdf" / "simple_en.pdf", [ENGLISH_BODY])

    _text_pdf(
        target / "outlined_pdf" / "with_toc.pdf",
        ["Taraflar\n\n" + TURKISH_BODY, "Mali Hükümler\n\n" + TURKISH_BODY],
        toc=[[1, "Taraflar", 1], [1, "Mali Hükümler", 2], [2, "Ödeme Koşulları", 2]],
    )

    _text_pdf(
        target / "pattern_pdf" / "madde_numbering.pdf",
        [
            "BİRİNCİ BÖLÜM\n\nMADDE 1 - " + TURKISH_BODY,
            "MADDE 2 - " + TURKISH_BODY + "\n\nEK-1 Teslimat Programı",
        ],
    )
    _text_pdf(
        target / "pattern_pdf" / "numeric_sections.pdf",
        ["1. Genel Hükümler\n\n" + TURKISH_BODY, "1.1 Kapsam\n\n" + TURKISH_BODY],
    )

    _text_pdf(
        target / "table_pdf" / "table_like.pdf",
        ["Fatura No | Tutar | Vade\nF-001 | 15000 | 2024-04-15\nF-002 | 22000 | 2024-05-01"],
    )

    _ruled_table_pdf(target / "table_pdf" / "ruled_table.pdf")

    _text_pdf(
        target / "mixed_language_pdf" / "tr_with_en_annex.pdf",
        [TURKISH_BODY, "ANNEX A\n\n" + ENGLISH_BODY],
    )

    _damaged_text_pdf(target / "damaged_text_pdf" / "control_chars.pdf")

    _rasterise(target / "text_pdf" / "simple_tr.pdf", target / "scanned_pdf" / "scan_tr.pdf")
    _rasterise(target / "text_pdf" / "simple_en.pdf", target / "scanned_pdf" / "scan_en.pdf")
    _scan_with_garbage_layer(
        target / "text_pdf" / "simple_tr.pdf",
        target / "scanned_pdf" / "scan_with_garbage_layer.pdf",
    )
    _mojibake_pdf(target / "damaged_text_pdf" / "mojibake.pdf")
    _hyphenated_pdf(target / "damaged_text_pdf" / "hyphenated.pdf")

    _docx(target / "docx" / "styled_headings.docx")
    _docx_with_table(target / "docx" / "with_table.docx")
    _xlsx(target / "xlsx" / "two_sheets.xlsx")
    _pptx(target / "pptx" / "two_slides.pptx")
    _pptx_with_table(target / "pptx" / "with_table.pptx")
    _broken(target / "broken")

    return {name: sorted((target / name).glob("*")) for name in categories}


if __name__ == "__main__":
    root = Path("tests/fixtures/generated")
    created = generate_fixtures(root)
    for name, paths in created.items():
        print(f"{name}: {len(paths)} file(s)")
