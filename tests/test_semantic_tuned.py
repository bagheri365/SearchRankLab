import numpy as np
import pytest

from searchranklab.routing import (
    SemanticConfig,
    TunedSemanticUtilityRouter,
    select_semantic_config,
)


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
        return np.asarray([self.mapping[text] for text in sentences], dtype=np.float32)


def test_select_semantic_config_returns_candidate():
    features = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.1, 0.9, 0.0, 0.0],
            [1.0, 0.0, 0.1, 0.0],
            [0.0, 1.0, 0.0, 0.1],
        ],
        dtype=np.float64,
    )
    utilities = {
        "bm25": np.asarray([0.9, 0.8, 0.1, 0.2, 0.85, 0.1]),
        "dense": np.asarray([0.1, 0.2, 0.9, 0.8, 0.2, 0.85]),
        "hybrid": np.asarray([0.3] * 6),
    }

    config = select_semantic_config(
        features,
        utilities,
        pca_components=(1, 2),
        alphas=(1.0, 10.0),
        cv_splits=3,
    )

    assert isinstance(config, SemanticConfig)
    assert config.pca_components in {1, 2}
    assert config.alpha in {1.0, 10.0}


def test_tuned_semantic_router_predicts_after_fit():
    encoder = FakeEncoder(
        {
            "a": [1.0, 0.0, 0.0],
            "b": [0.9, 0.1, 0.0],
            "c": [0.0, 1.0, 0.0],
            "d": [0.1, 0.9, 0.0],
            "e": [1.0, 0.0, 0.1],
            "f": [0.0, 1.0, 0.1],
            "left": [1.0, 0.0, 0.0],
            "right": [0.0, 1.0, 0.0],
        }
    )
    queries = ["a", "b", "c", "d", "e", "f"]
    utilities = {
        "bm25": [0.9, 0.8, 0.1, 0.2, 0.85, 0.1],
        "dense": [0.1, 0.2, 0.9, 0.8, 0.2, 0.85],
        "hybrid": [0.3] * 6,
    }

    router = TunedSemanticUtilityRouter(
        encoder=encoder,
        pca_components=(1, 2),
        alphas=(1.0,),
        cv_splits=3,
    ).fit(queries, utilities)

    assert router.predict(["left", "right"]) == ["bm25", "dense"]


def test_tuned_router_requires_fit_before_predict():
    router = TunedSemanticUtilityRouter(
        encoder=FakeEncoder({"a": [1.0, 0.0]}),
        pca_components=(1,),
        alphas=(1.0,),
        cv_splits=2,
    )

    with pytest.raises(ValueError, match="fit before prediction"):
        router.predict(["a"])
