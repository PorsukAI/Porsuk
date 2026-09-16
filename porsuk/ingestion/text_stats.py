"""Pure text measures behind `text_plausibility` and chunk language: no config
and no I/O by design, so a calibrated number stays stable across runs.
`text_plausibility` is `mean(script_validity, vowel_rate, function_word_rate)
* (1 - repetition_penalty)`, not a plain unweighted mean: the multiplicative
`_repetition_penalty` term (share of tokens with 3+ of the same letter in a
row) is needed because `script_validity` alone cannot detect OCR-garbage
tokens built from legitimate Turkish letters, and reweighting the other three
signals alone cannot fix that without breaking mojibake detection or
agglutinated-Turkish recall.
"""

from __future__ import annotations

import re
import unicodedata

# Latin plus the six letters Turkish adds. Written out rather than derived so
# that a naive ASCII-only check can never creep back in: scoring 'ğ' as foreign
# would mark every correctly-parsed Turkish document as damaged.
_TURKISH_EXTRA = "çğıöşüÇĞİÖŞÜ"
# Both cases listed explicitly rather than derived via .lower(): Python's
# Unicode casefold turns Turkish 'İ' (dotted capital I, U+0130) into a
# *two*-character string ('i' + COMBINING DOT ABOVE, U+0307), which then
# never matches a single-character set. Comparing letters against both cases
# directly avoids that decomposition entirely.
_ALPHABET = (
    set("abcdefghijklmnopqrstuvwxyz") | set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") | set(_TURKISH_EXTRA)
)
_VOWELS = set("aeıioöuü")

# High-frequency function words, Turkish and English together. Function words
# are the lexical signal that survives agglutination: `kitaplarımızdan` is
# unmatchable, but `ve`, `bir` and `için` are not inflected. Kept small on
# purpose: this detects whether text is language at all, not what it says.
_FUNCTION_WORDS = frozenset(
    """
    ve veya ile için gibi kadar sonra önce ancak fakat çünkü ise da de ki mi
    bu şu o bir birer her hiç bazı tüm daha en çok az var yok olan olarak
    tarafından üzere göre içinde arasında hakkında ilgili ayrıca
    the a an and or of to in on for with by from as at is are was were be been
    that this these those which not but if then than such shall may will
    """.split()
)

_TOKEN_PATTERN = re.compile(rf"[a-zA-Z{_TURKISH_EXTRA}]+")

# Whitespace that is normal in extracted text. Counting these as damage would
# flag every multi-line document.
_ALLOWED_CONTROL = frozenset("\n\r\t\f\v")

# Three or more of the same letter in a row. No Turkish or English word does
# this ('zzz', 'ttt', 'ııı' are keyboard mashing or OCR noise, not words); see
# the module docstring for why this exists alongside the three original signals.
_RUN_PATTERN = re.compile(r"(.)\1\1")

# Same pitfall as _ALPHABET above, reached through .lower() instead of a set
# comparison this time: plain str.lower() turns Turkish 'İ' into a *two*-
# character sequence ('i' + COMBINING DOT ABOVE), so a capitalised Turkish
# function word like 'İçin' would tokenize to 'i̇çin' and silently miss every
# _FUNCTION_WORDS lookup: sentence-initial function words are common, not an
# edge case. Folding 'İ' to plain 'i' before the general .lower() keeps this
# a single codepoint. Checked against every letter in our alphabet (Turkish
# extras plus ASCII); 'İ' is the only one whose casefold isn't already a
# single character, so dotless 'ı' and ASCII 'I' are untouched by this and
# still round-trip through .upper()/.lower() the ordinary way.
_FOLD_DOTTED_I = str.maketrans({"İ": "i"})


def tokenize(text: str) -> list[str]:
    """Alphabetic tokens, lowercased. Digits and punctuation are not tokens."""
    return [
        match.group(0).translate(_FOLD_DOTTED_I).lower() for match in _TOKEN_PATTERN.finditer(text)
    ]


def script_validity(text: str) -> float:
    """Share of letters that belong to the expected alphabet.

    Catches alphabet bleed (Cyrillic, CJK) and mojibake. Mojibake matters
    because UTF-8 read as Latin-1 stays inside the Latin block and therefore
    survives `bad_char_ratio` untouched: 'Ödemeler' becomes 'Ã–demeler',
    which is all perfectly ordinary characters.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    inside = sum(1 for c in letters if c in _ALPHABET)
    return inside / len(letters)


def vowel_rate(tokens: list[str]) -> float:
    """Share of tokens containing at least one vowel.

    OCR garbage tends to produce consonant runs ('tlıc', 'rn1', 'vvhh') that
    are individually valid letters but cannot be words.
    """
    candidates = [t for t in tokens if len(t) >= 2]
    if not candidates:
        return 0.0
    return sum(1 for t in candidates if any(c in _VOWELS for c in t)) / len(candidates)


def function_word_rate(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in _FUNCTION_WORDS) / len(tokens)


def _repetition_penalty(tokens: list[str]) -> float:
    # Known and accepted: Roman numerals (III, VIII -> iii after folding) and
    # "www" also match a three-letter run, and both occur in Turkish legal
    # PDFs. At document scale their share is negligible - measured, a
    # Roman-numeral-heavy *snippet* scores 0.408, but a whole document does
    # not move. Noted so it is not rediscovered as a bug.
    """Share of tokens with a 3+ run of the same letter, see module docstring."""
    candidates = [t for t in tokens if len(t) >= 2]
    if not candidates:
        return 0.0
    bad = sum(1 for t in candidates if _RUN_PATTERN.search(t))
    return bad / len(candidates)


def text_plausibility(text: str) -> float:
    """Does this look like language, rather than damage?

    The mean of the three signals above, discounted by the share of tokens
    that are letter-run noise. See the module docstring for the measured
    numbers behind this combination and why a plain unweighted mean does not
    separate the OCR-garbage and agglutinated-Turkish fixtures.
    """
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    base = (script_validity(text) + vowel_rate(tokens) + function_word_rate(tokens)) / 3.0
    return base * (1.0 - _repetition_penalty(tokens))


def bad_char_ratio(text: str) -> float:
    """Replacement (U+FFFD) and control characters."""
    if not text:
        return 0.0
    bad = sum(
        1
        for c in text
        if c == "�" or (unicodedata.category(c) == "Cc" and c not in _ALLOWED_CONTROL)
    )
    return bad / len(text)


def detect_language(text: str) -> str | None:
    """Chunk-level language (the chunk `language` field).

    A placeholder behind a stable signature. The real detector belongs in
    `retrieval/language.py`, where the cross-lingual claim
    gets measured; building a second one now would be a guess competing with a
    measurement.
    """
    tokens = tokenize(text)
    if not tokens:
        return None
    if any(c in _TURKISH_EXTRA for c in text):
        return "tr"
    turkish_hits = sum(1 for t in tokens if t in _FUNCTION_WORDS and _looks_turkish(t))
    english_hits = sum(1 for t in tokens if t in _FUNCTION_WORDS and not _looks_turkish(t))
    if turkish_hits == english_hits == 0:
        return None
    return "tr" if turkish_hits > english_hits else "en"


_TURKISH_FUNCTION_WORDS = frozenset(
    """
    ve veya ile için gibi kadar sonra önce ancak fakat çünkü ise da de ki mi
    bu şu o bir birer her hiç bazı tüm daha en çok az var yok olan olarak
    tarafından üzere göre içinde arasında hakkında ilgili ayrıca
    """.split()
)


def _looks_turkish(token: str) -> bool:
    return token in _TURKISH_FUNCTION_WORDS
