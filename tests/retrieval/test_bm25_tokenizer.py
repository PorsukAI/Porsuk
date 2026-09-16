"""BM25 tokenizer (porsuk/retrieval/bm25.py)."""

from __future__ import annotations

from porsuk.retrieval import bm25


def test_tokenize_lowercases_turkish_dotted_I():
    # "İ".lower() is "i" + combining dot in default Python; the tokenizer must
    # fold "İ" -> "i" so query "istanbul" matches document "İSTANBUL".
    doc = bm25.tokenize("İSTANBUL İ")
    q = bm25.query_terms("istanbul")
    assert set(q) <= set(doc)


def test_tokenize_folds_dotless_I():
    doc = bm25.tokenize("IŞIK")  # -> "ışık"
    q = bm25.query_terms("ışık")
    assert set(q) <= set(doc)


def test_tokenize_counts_term_frequency():
    tf = bm25.tokenize("doktor doktor doçent")
    assert sorted(tf.values()) == [1, 2]  # doçent x1, doktor x2


def test_tokenize_drops_stopwords():
    assert bm25.tokenize("ve ile veya") == {}


def test_query_terms_are_a_set_not_frequencies():
    q = bm25.query_terms("doktor doktor doktor")
    assert set(q.values()) == {1.0}
    assert len(q) == 1


def test_empty_input():
    assert bm25.tokenize("") == {}
    assert bm25.tokenize("   \n ") == {}
    assert bm25.query_terms("") == {}


def test_token_id_is_stable_and_32bit():
    a = bm25._token_id("sözleşme")
    b = bm25._token_id("sözleşme")
    assert a == b
    assert 0 <= a < 2**32


def test_stem_stage_folds_inflections():
    # "sözleşmelerdeki" and "sözleşme" -> same stem -> same token id
    with_stem = bm25.tokenize("sözleşmelerdeki sözleşme", stem=True)
    assert len(with_stem) == 1


def test_no_stem_keeps_inflections_apart():
    without_stem = bm25.tokenize("sözleşmelerdeki sözleşme", stem=False)
    assert len(without_stem) == 2
