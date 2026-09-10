import pytest

from searchranklab.routing import (
    PreRetrievalRouter,
    always_strategy,
    length_heuristic,
    random_strategy,
)


def test_always_strategy_routes_every_query_identically():
    assert always_strategy(["a", "b"], "bm25") == ["bm25", "bm25"]


def test_random_strategy_is_deterministic_for_seed():
    queries = ["a", "b", "c", "d"]
    first = random_strategy(queries, seed=7)
    second = random_strategy(queries, seed=7)

    assert first == second
    assert len(first) == len(queries)


def test_length_heuristic_routes_by_token_count():
    queries = [
        "one two",
        " ".join(["word"] * 12),
        " ".join(["word"] * 20),
    ]

    assert length_heuristic(queries) == ["bm25", "dense", "hybrid"]


def test_length_heuristic_validates_thresholds():
    with pytest.raises(ValueError, match="positive"):
        length_heuristic(["query"], dense_threshold=0)

    with pytest.raises(ValueError, match="greater"):
        length_heuristic(["query"], dense_threshold=10, hybrid_threshold=10)


def test_router_supports_unweighted_logistic_regression():
    router = PreRetrievalRouter(class_weight=None)

    assert router.class_weight is None
