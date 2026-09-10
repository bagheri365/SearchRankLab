import pytest

from searchranklab.retrieval import BM25Retriever, tokenize


def test_tokenize_normalizes_case_and_punctuation():
    assert tokenize("Alpha, BETA!") == ["alpha", "beta"]


def test_bm25_returns_matching_document_first():
    corpus = {
        "d1": {"title": "", "text": "cats sleep on sofas"},
        "d2": {"title": "", "text": "quantum particle physics"},
        "d3": {"title": "", "text": "dogs play outside"},
    }

    retriever = BM25Retriever(corpus)
    results = retriever.search("quantum physics", k=2)

    assert results[0].doc_id == "d2"
    assert len(results) == 2


def test_bm25_rejects_nonpositive_k():
    retriever = BM25Retriever({"d1": {"title": "", "text": "hello"}})

    with pytest.raises(ValueError, match="positive"):
        retriever.search("hello", k=0)
