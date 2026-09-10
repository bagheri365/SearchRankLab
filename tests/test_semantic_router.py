import numpy as np
import pytest

from searchranklab.routing import SemanticUtilityRouter, semantic_query_features


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
        vectors = np.asarray([self.mapping[text] for text in sentences], dtype=np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.where(norms == 0, 1.0, norms)
        return vectors


def test_semantic_query_features_return_matrix():
    encoder = FakeEncoder({"a": [1.0, 0.0], "b": [0.0, 1.0]})

    matrix = semantic_query_features(["a", "b"], encoder)

    assert matrix.shape == (2, 2)
    assert np.isfinite(matrix).all()


def test_semantic_router_learns_simple_utility_pattern():
    encoder = FakeEncoder(
        {
            "a": [1.0, 0.0],
            "b": [0.9, 0.1],
            "c": [0.0, 1.0],
            "d": [0.1, 0.9],
            "left": [1.0, 0.0],
            "right": [0.0, 1.0],
        }
    )
    queries = ["a", "b", "c", "d"]
    utilities = {
        "bm25": [0.9, 0.8, 0.1, 0.2],
        "dense": [0.1, 0.2, 0.9, 0.8],
        "hybrid": [0.3, 0.3, 0.3, 0.3],
    }

    router = SemanticUtilityRouter(
        encoder=encoder,
        alpha=0.1,
    ).fit(queries, utilities)

    assert router.predict(["left", "right"]) == ["bm25", "dense"]


def test_semantic_router_validates_alpha():
    encoder = FakeEncoder({"a": [1.0, 0.0]})

    with pytest.raises(ValueError, match="nonnegative"):
        SemanticUtilityRouter(encoder=encoder, alpha=-1.0)
