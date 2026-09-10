import numpy as np
import pytest

from searchranklab.retrieval import DenseRetriever


class FakeEncoder:
    def __init__(self, mapping):
        self.mapping = mapping

    def encode(
        self,
        sentences,
        *,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ):
        if isinstance(sentences, str):
            sentences = [sentences]
        vectors = np.asarray([self.mapping[text] for text in sentences], dtype=np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.where(norms == 0, 1.0, norms)
        return vectors


def test_dense_retriever_returns_most_similar_document_first():
    corpus = {
        "d1": {"title": "", "text": "cats"},
        "d2": {"title": "", "text": "physics"},
        "d3": {"title": "", "text": "dogs"},
    }
    encoder = FakeEncoder(
        {
            "cats": [1.0, 0.0],
            "physics": [0.0, 1.0],
            "dogs": [0.8, 0.2],
            "quantum": [0.0, 1.0],
        }
    )

    retriever = DenseRetriever(corpus, encoder=encoder)
    results = retriever.search("quantum", k=2)

    assert results[0].doc_id == "d2"
    assert len(results) == 2


def test_dense_batch_search_returns_all_queries():
    corpus = {
        "d1": {"title": "", "text": "alpha"},
        "d2": {"title": "", "text": "beta"},
    }
    encoder = FakeEncoder(
        {
            "alpha": [1.0, 0.0],
            "beta": [0.0, 1.0],
            "query alpha": [1.0, 0.0],
            "query beta": [0.0, 1.0],
        }
    )

    retriever = DenseRetriever(corpus, encoder=encoder)
    run = retriever.batch_search(
        {"q1": "query alpha", "q2": "query beta"},
        k=1,
    )

    assert run["q1"][0].doc_id == "d1"
    assert run["q2"][0].doc_id == "d2"


def test_dense_retriever_rejects_nonpositive_k():
    encoder = FakeEncoder({"hello": [1.0, 0.0]})
    retriever = DenseRetriever(
        {"d1": {"title": "", "text": "hello"}},
        encoder=encoder,
    )

    with pytest.raises(ValueError, match="positive"):
        retriever.search("hello", k=0)
