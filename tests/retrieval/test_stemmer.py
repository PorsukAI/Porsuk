from porsuk.retrieval.stemmer import stem_terms


def test_agglutinated_forms_share_a_stem():
    a = stem_terms("sözleşmelerdeki")
    b = stem_terms("sözleşme")
    assert a[0] == b[0]


def test_multiple_terms():
    stems = stem_terms("Ödeme koşulları ve gecikme faizi")
    assert len(stems) >= 4
    assert all(s == s.lower() for s in stems)


def test_empty():
    assert stem_terms("") == []
    assert stem_terms("   ") == []
