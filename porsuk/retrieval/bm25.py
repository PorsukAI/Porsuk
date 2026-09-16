"""BM25 tokenisation for keyword search: turns text into the term-frequency
vectors Qdrant scores with corpus IDF. Turkish-aware: `İ`/`I` casefolding is
done explicitly because Python's default `str.lower()` maps `İ` to `i` plus a
combining dot, which would make query "istanbul" miss document "İstanbul".
`tokenize` builds the document-side term-frequency vector; `query_terms`
builds the query-side term set (term-id → 1.0).
"""

from __future__ import annotations

import re
import zlib

from porsuk.retrieval.stemmer import stem_terms

# High-frequency Turkish function words that carry no retrieval signal in
# legal/corporate text: conjunctions, postpositions, pronouns, question
# particles, common auxiliaries. Not a linguistic inventory, just the words
# that only add noise to a lexical match.
TURKISH_STOPWORDS: frozenset[str] = frozenset(
    """
    ve veya ile ya da ki de da mi mı mu mü ise için gibi kadar göre
    bu şu o bir birçok her hangi hiç çok az en daha ne nasıl niçin neden
    olan olarak olup ancak fakat ama yani iken üzere dolayı itibaren
    değil hem ilgili sonra önce içinde üzerine kendi
    """.split()
)

_WORD = re.compile(r"\w+", re.UNICODE)
_UPPER_MAP = str.maketrans({"İ": "i", "I": "ı"})


def _casefold_tr(text: str) -> str:
    """Lowercase with Turkish dotted/dotless I handled before `str.lower()`."""
    return text.translate(_UPPER_MAP).lower()


def _token_id(term: str) -> int:
    """Stable 32-bit id for a term.

    `zlib.crc32` is deterministic across processes and Python runs, unlike
    the built-in `hash()` (salted per process). Collisions at 32 bits over a
    realistic vocabulary are rare enough to ignore for lexical ranking.
    """
    return zlib.crc32(term.encode("utf-8"))


def _terms(text: str, *, stem: bool) -> list[str]:
    folded = _casefold_tr(text)
    tokens = [m.group(0) for m in _WORD.finditer(folded)]
    if stem:
        # stem_terms lowercases again (harmless, already folded) and splits
        # on its own \w+; feeding it a space-joined string is fine.
        tokens = stem_terms(" ".join(tokens))
    return [t for t in tokens if t and t not in TURKISH_STOPWORDS]


def tokenize(text: str, *, stem: bool = False) -> dict[int, int]:
    """Document side: term-id → term frequency, stopwords dropped."""
    tf: dict[int, int] = {}
    for term in _terms(text, stem=stem):
        tid = _token_id(term)
        tf[tid] = tf.get(tid, 0) + 1
    return tf


def tokenize_terms(text: str, *, stem: bool = False) -> list[str]:
    """The distinct terms of `text` as strings, for the plain word-membership
    backend (`keyword_backend=text`), which matches on the words themselves,
    not on hashed ids.
    """
    return list(dict.fromkeys(_terms(text, stem=stem)))


def query_terms(text: str, *, stem: bool = False) -> dict[int, float]:
    """Query side: term-id → 1.0 for each distinct term."""
    return {tid: 1.0 for tid in tokenize(text, stem=stem)}
