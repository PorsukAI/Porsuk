from porsuk.ingestion.entities import extract_entities


def test_extracts_mixed_entities():
    text = (
        "Sözleşme No: 2024/A-137, tarih 15/03/2024. Bedel 1.250.000,00 TL. "
        "İrtibat: info@abclojistik.com.tr. Taraf: ABC Lojistik Anonim Şirketi. "
        "VKN 1234567890."
    )
    ents = extract_entities(text)
    assert "2024/A-137" in ents
    assert "15/03/2024" in ents
    assert "info@abclojistik.com.tr" in ents
    assert "1234567890" in ents
    assert any("ABC Lojistik" in e for e in ents)
    assert any("1.250.000,00" in e for e in ents)


def test_deduplicates_preserving_first_appearance():
    text = "15/03/2024 ... yine 15/03/2024 ... sonra 16/03/2024"
    assert extract_entities(text) == ["15/03/2024", "16/03/2024"]


def test_empty_text_returns_empty():
    assert extract_entities("") == []
    assert extract_entities("   \n  ") == []


def test_no_false_positive_on_plain_lowercase_prose():
    text = "bu bir sözleşme metnidir ve herhangi bir tarih ya da tutar içermez"
    assert extract_entities(text) == []
