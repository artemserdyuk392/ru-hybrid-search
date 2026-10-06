from ruhybrid.tokenize import _NGRAM_PREFIX, tokenize_document, tokenize_query


def _has_ngram(tokens):
    return any(t.startswith(_NGRAM_PREFIX) for t in tokens)


def test_long_russian_word_emits_stem_and_ngrams():
    tokens = tokenize_document("ускорение", ngram_size=4)
    assert "ускорен" in tokens
    assert _has_ngram(tokens)
    assert _NGRAM_PREFIX + "уско" in tokens


def test_verb_and_noun_share_ngrams_not_stems():
    query = set(tokenize_query("ускорить", ngram_size=4))
    doc = set(tokenize_document("ускорение", ngram_size=4))
    shared = query & doc
    assert shared, "verb and noun must share at least one token"
    # the stems differ; the overlap can only come from the character n-grams
    assert all(t.startswith(_NGRAM_PREFIX) for t in shared)


def test_english_word_has_only_a_stem():
    tokens = tokenize_document("optimization", ngram_size=4)
    assert tokens
    assert not _has_ngram(tokens)


def test_short_russian_word_has_no_ngrams():
    tokens = tokenize_document("кот", ngram_size=4)
    assert "кот" in tokens
    assert not _has_ngram(tokens)


def test_ngrams_off_by_default():
    tokens = tokenize_document("ускорение")
    assert "ускорен" in tokens
    assert not _has_ngram(tokens)


def test_stopwords_are_removed():
    assert tokenize_document("и в на под") == []


def test_an_ngram_never_matches_a_word_stem():
    # the query stem is also a 4-gram inside the document word; the two must stay
    # distinct tokens, or a document matches a word it does not contain
    query = set(tokenize_query("мост", ngram_size=4))
    doc = set(tokenize_document("помостки", ngram_size=4))
    assert query == {"мост"}
    assert len(doc) > 1, "the document word must emit n-grams for this to mean anything"
    assert query.isdisjoint(doc)
