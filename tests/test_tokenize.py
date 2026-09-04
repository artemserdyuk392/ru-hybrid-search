from ruhybrid.tokenize import _NGRAM_PREFIX, tokenize_document, tokenize_query


def _has_ngram(tokens):
    return any(t.startswith(_NGRAM_PREFIX) for t in tokens)


def test_long_russian_word_emits_stem_and_ngrams():
    tokens = tokenize_document("ускорение")
    assert "ускорен" in tokens
    assert _has_ngram(tokens)
    assert _NGRAM_PREFIX + "уско" in tokens


def test_verb_and_noun_share_ngrams_not_stems():
    query = set(tokenize_query("ускорить"))
    doc = set(tokenize_document("ускорение"))
    shared = query & doc
    assert shared, "verb and noun must share at least one token"
    # the stems differ; the overlap can only come from the character n-grams
    assert all(t.startswith(_NGRAM_PREFIX) for t in shared)


def test_english_word_has_only_a_stem():
    tokens = tokenize_document("optimization")
    assert tokens
    assert not _has_ngram(tokens)


def test_short_russian_word_has_no_ngrams():
    tokens = tokenize_document("кот")
    assert "кот" in tokens
    assert not _has_ngram(tokens)


def test_ngram_size_zero_disables_ngrams():
    tokens = tokenize_document("ускорение", ngram_size=0)
    assert "ускорен" in tokens
    assert not _has_ngram(tokens)


def test_stopwords_are_removed():
    assert tokenize_document("и в на под") == []
