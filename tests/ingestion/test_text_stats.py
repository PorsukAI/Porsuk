"""Tests for the dictionary-free text plausibility and language statistics."""

import pytest

from porsuk.ingestion.text_stats import (
    bad_char_ratio,
    detect_language,
    function_word_rate,
    script_validity,
    text_plausibility,
    tokenize,
    vowel_rate,
)

CLEAN_TR = (
    "İşbu sözleşme, ABC Lojistik A.Ş. ile XYZ Sanayi Ltd. Şti. arasında "
    "tedarik hizmetlerinin sağlanması amacıyla akdedilmiştir. Ödemeler fatura "
    "tarihinden itibaren 30 gün içinde yapılır."
)
CLEAN_EN = (
    "This agreement is entered into between the parties for the provision of "
    "supply services. Payments shall be made within 30 days of the invoice date."
)
OCR_GARBAGE = "Iıl ııı tlıc rn1 vvhh zzz ttt lll nnn kkk ııı lıl ııl"
MOJIBAKE = "Ã–demeler faturaÃ§ tarihinden Åžti. itibaren Ã¼Ã§ gÃ¼n iÃ§inde"


def test_agglutinated_turkish_scores_as_plausible():
    """The whole reason dictionary lookup was dropped.

    None of these surface forms appear in any word list, and Turkish legal
    prose is made of them. A measure that scores this low would escalate clean
    mevzuat PDFs to OCR.
    """
    agglutinated = (
        "Değerlendirilmesinde uygulanmasına ilişkin kitaplarımızdan "
        "yararlanılabilmesi için başvurulduğunda karşılaştırmalarımızın "
        "sonuçlandırılmasıyla birlikte gerçekleştirilebilecektir."
    )
    assert text_plausibility(agglutinated) > 0.6


def test_clean_text_outscores_ocr_garbage_in_both_languages():
    assert text_plausibility(CLEAN_TR) > text_plausibility(OCR_GARBAGE)
    assert text_plausibility(CLEAN_EN) > text_plausibility(OCR_GARBAGE)


def test_ocr_garbage_scores_low():
    assert text_plausibility(OCR_GARBAGE) < 0.4


def test_mojibake_is_detected_by_script_validity():
    """Latin-1-decoded UTF-8 stays inside the Latin block, so it survives
    bad_char_ratio. The alphabet check is what catches it."""
    assert script_validity(MOJIBAKE) < script_validity(CLEAN_TR)


def test_turkish_diacritics_are_not_penalised():
    """A naive ASCII-only alphabet check would score Turkish as foreign."""
    assert script_validity("ğüşiöçİĞÜŞÖÇ ıI") == pytest.approx(1.0)


def test_tokenize_splits_on_non_letters_and_lowercases():
    assert tokenize("Ödemeler, 30 gün: içinde!") == ["ödemeler", "gün", "içinde"]


def test_vowel_rate_flags_vowelless_tokens():
    assert vowel_rate(tokenize("kitap masa kalem")) == pytest.approx(1.0)
    assert vowel_rate(tokenize("ttt kkk lll")) == pytest.approx(0.0)


def test_function_word_rate_finds_both_languages():
    assert function_word_rate(tokenize(CLEAN_TR)) > 0.0
    assert function_word_rate(tokenize(CLEAN_EN)) > 0.0
    assert function_word_rate(tokenize(OCR_GARBAGE)) == pytest.approx(0.0)


def test_function_words_are_immune_to_agglutination():
    """Function words do not inflect, which is why they are the lexical half."""
    assert function_word_rate(tokenize("ve bir bu için ile")) == pytest.approx(1.0)


def test_capitalised_turkish_function_words_are_still_recognised():
    """Sentence-initial capitalisation must not defeat the lookup.

    Plain str.lower() turns Turkish 'İ' into 'i' + COMBINING DOT ABOVE (two
    characters), not plain 'i'. A naive tokenizer would fold 'İçin' to
    'i̇çin' and silently miss the function-word table on every sentence that
    opens with one.
    """
    for capitalised, lowercase in [("İçin", "için"), ("İle", "ile"), ("İse", "ise")]:
        tokens = tokenize(capitalised)
        assert tokens == [lowercase]
        assert function_word_rate(tokens) == pytest.approx(1.0)


def test_bad_char_ratio_counts_control_and_replacement_characters():
    assert bad_char_ratio("temiz metin") == pytest.approx(0.0)
    assert bad_char_ratio("a\x01b\x02c�") == pytest.approx(0.5)


def test_bad_char_ratio_allows_ordinary_whitespace():
    """Newlines and tabs are control characters by category but are normal in
    extracted text; counting them would flag every multi-line document."""
    assert bad_char_ratio("satır bir\nsatır iki\tsekme\r\n") == pytest.approx(0.0)


def test_detect_language_separates_turkish_from_english():
    assert detect_language(CLEAN_TR) == "tr"
    assert detect_language(CLEAN_EN) == "en"


def test_detect_language_returns_none_when_there_is_nothing_to_go_on():
    assert detect_language("") is None
    assert detect_language("123 456 789") is None


def test_empty_text_scores_zero_rather_than_dividing_by_zero():
    assert text_plausibility("") == 0.0
    assert script_validity("") == 0.0
    assert bad_char_ratio("") == 0.0
