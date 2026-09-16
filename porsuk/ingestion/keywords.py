"""YAKE keyword extraction.

Statistical - a term's position, frequency, sentence spread and context
variety - so it needs no training and is language-independent bar a stopword
list. KeyBERT scores better but runs an embedding pass, which is the wrong
tool in the cheap profile (doküman profili) layer.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yake

_STOPWORDS_FILE = Path(__file__).with_name("tr_stopwords.txt")


@lru_cache(maxsize=1)
def _tr_stopwords() -> set[str]:
    return set(_STOPWORDS_FILE.read_text(encoding="utf-8").split())


def keyword_terms(text: str, *, top_k: int = 15, language: str | None = None) -> list[str]:
    """Up to ``top_k`` keyword phrases, lowercased, most significant first,
    deduplicated. ``[]`` for empty or blank text."""
    if not text.strip():
        return []
    lang = language or "en"
    extractor = yake.KeywordExtractor(
        lan=lang,
        n=2,  # unigrams and bigrams
        top=top_k * 3,  # over-extract, then dedupe and trim
        dedupLim=0.8,
        stopwords=_tr_stopwords() if lang == "tr" else None,
    )
    seen: set[str] = set()
    out: list[str] = []
    for phrase, _score in sorted(extractor.extract_keywords(text), key=lambda kv: kv[1]):
        low = phrase.lower().strip()
        if low and low not in seen:
            seen.add(low)
            out.append(low)
        if len(out) >= top_k:
            break
    return out
