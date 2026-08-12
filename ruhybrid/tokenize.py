"""Tokenization for mixed Russian/English text: normalize, drop stopwords, stem."""

from __future__ import annotations

import re

from nltk.stem.snowball import SnowballStemmer

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_CYRILLIC = re.compile(r"[а-я]")
_LATIN = re.compile(r"[a-z]")

_ru_stemmer = SnowballStemmer("russian")
_en_stemmer = SnowballStemmer("english")

# character n-grams are marked so they never collide with a real word stem
_NGRAM_PREFIX = "#"
_NGRAM_MIN_LEN = 5

_stopwords_cache: set[str] | None = None


def _load_stopwords() -> set[str]:
    import nltk
    from nltk.corpus import stopwords

    try:
        words = stopwords.words("russian") + stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)
        words = stopwords.words("russian") + stopwords.words("english")
    # normalize the same way tokens are normalized, so matching is consistent
    return {w.replace("ё", "е") for w in words}


def _stopwords() -> set[str]:
    global _stopwords_cache
    if _stopwords_cache is None:
        _stopwords_cache = _load_stopwords()
    return _stopwords_cache


def normalize(text: str) -> str:
    return text.lower().replace("ё", "е")


def _stem(word: str) -> str:
    if _CYRILLIC.search(word):
        return _ru_stemmer.stem(word)
    if _LATIN.search(word):
        return _en_stemmer.stem(word)
    return word


def _char_ngrams(word: str, n: int) -> list[str]:
    if len(word) < n:
        return []
    return [word[i : i + n] for i in range(len(word) - n + 1)]


def _base_tokens(text: str, ngram_size: int) -> list[str]:
    stop = _stopwords()
    tokens = []
    for word in _WORD_RE.findall(normalize(text)):
        if word in stop:
            continue
        tokens.append(_stem(word))
        # bridge the stemmer's split of verb/noun pairs (ускорить/ускорение)
        # via shared character n-grams; only for long Russian words, since
        # the stemmer already handles English inflection and ngrams just bloat.
        if len(word) >= _NGRAM_MIN_LEN and _CYRILLIC.search(word):
            tokens.extend(_NGRAM_PREFIX + g for g in _char_ngrams(word, ngram_size))
    return tokens


def tokenize_document(text: str, ngram_size: int = 4) -> list[str]:
    return _base_tokens(text, ngram_size)


def tokenize_query(text: str, ngram_size: int = 4) -> list[str]:
    return _base_tokens(text, ngram_size)
