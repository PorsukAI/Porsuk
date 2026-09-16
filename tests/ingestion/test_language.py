from porsuk.ingestion.language import detect_language


def test_turkish():
    assert (
        detect_language(
            "Bu sözleşme taraflar arasında akdedilmiş olup ödeme koşulları ekte belirtilmiştir."
        )
        == "tr"
    )


def test_english():
    assert (
        detect_language(
            "This agreement is made between the parties and payment terms are set out in the annex."
        )
        == "en"
    )


def test_too_short_is_none():
    assert detect_language("ok") is None
    assert detect_language("   ") is None
    assert detect_language("") is None


def test_newlines_do_not_break_detection():
    assert detect_language("Tedarik sözleşmesi\nödeme koşulları\nfesih hükümleri gibi") == "tr"
