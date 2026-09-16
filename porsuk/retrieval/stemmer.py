"""Turkish stemming for keyword search.

Turkish is agglutinative: `sözleşmelerdeki` and `sözleşme` are the same
lexical item, and a keyword search that treats them as different terms
misses obvious matches. Snowball, not Zemberek: Zemberek needs a JVM and is
heavy; Snowball's Turkish stemmer is pure Python via PyStemmer and enough
for term normalisation. Whether this is applied to sparse tokens or to query
expansion is a measured decision, not an assumed one.
"""

from __future__ import annotations

import re

import Stemmer

_STEMMER = Stemmer.Stemmer("turkish")
_TOKEN = re.compile(r"\w+", re.UNICODE)


def stem_terms(text: str) -> list[str]:
    tokens = [m.group(0).lower() for m in _TOKEN.finditer(text)]
    return _STEMMER.stemWords(tokens) if tokens else []
