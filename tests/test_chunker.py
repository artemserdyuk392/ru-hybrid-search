import re

from ruhybrid.chunker import chunk_document


def test_short_document_is_single_chunk_with_fields():
    chunks = chunk_document(0, "Короткий текст про поиск.", title="T", source="src")
    assert len(chunks) == 1
    assert chunks[0].doc_id == 0
    assert chunks[0].chunk_index == 0
    assert chunks[0].title == "T"
    assert chunks[0].source == "src"


def test_long_document_is_split_with_overlap():
    text = " ".join(f"segment{i} content marker." for i in range(200))
    chunks = chunk_document(1, text, max_chars=200, overlap=60)
    assert len(chunks) > 1
    for a, b in zip(chunks, chunks[1:], strict=False):
        a_markers = set(re.findall(r"segment\d+", a.text))
        b_markers = set(re.findall(r"segment\d+", b.text))
        assert a_markers & b_markers, "consecutive chunks must overlap"


def test_document_exactly_at_window_is_one_chunk():
    text = "a" * 300
    chunks = chunk_document(2, text, max_chars=300, overlap=50)
    assert len(chunks) == 1
    assert len(chunks[0].text) == 300


def test_empty_document_yields_no_chunks():
    assert chunk_document(3, "   \n\n  ") == []
