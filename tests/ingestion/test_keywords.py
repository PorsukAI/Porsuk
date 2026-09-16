from porsuk.ingestion.keywords import keyword_terms


def test_extracts_salient_terms_from_a_turkish_contract():
    text = (
        "İşbu tedarik sözleşmesi ABC Lojistik ile XYZ Sanayi arasında "
        "akdedilmiştir. Ödeme koşulları, gecikme faizi ve fesih hükümleri "
        "aşağıda düzenlenmiştir. Teslimat programı EK-1'de yer almaktadır."
    )
    terms = keyword_terms(text, top_k=10, language="tr")
    joined = " ".join(terms)
    assert "tedarik" in joined
    assert "sözleşme" in joined or "sözleşmesi" in joined
    assert len(terms) <= 10
    assert terms == [t.lower() for t in terms]
    assert len(terms) == len(set(terms))  # deduplicated


def test_turkish_stopwords_are_filtered():
    text = (
        "İşbu belge ile ve veya için gibi kadar daha çok en de da ki "
        "ancak fakat sözleşme tedarik ödeme"
    )
    terms = keyword_terms(text, top_k=15, language="tr")
    assert "ve" not in terms
    assert "veya" not in terms
    assert "için" not in terms


def test_empty_or_blank_text_returns_empty():
    assert keyword_terms("", top_k=10) == []
    assert keyword_terms("   ", top_k=10) == []


def test_english_text_without_language_hint():
    text = (
        "This supply agreement between the parties sets out the payment terms, "
        "late payment interest and termination clauses."
    )
    terms = keyword_terms(text, top_k=8)
    assert any("payment" in t or "agreement" in t or "supply" in t for t in terms)
