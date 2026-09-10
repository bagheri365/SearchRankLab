import pytest

from searchranklab.routing import UtilityRegressionRouter


def test_utility_router_learns_per_strategy_targets():
    queries = [
        "short",
        "tiny",
        "a considerably longer scientific query",
        "another considerably longer biomedical query",
    ]
    utilities = {
        "bm25": [0.9, 0.8, 0.2, 0.1],
        "dense": [0.1, 0.2, 0.8, 0.9],
        "hybrid": [0.3, 0.3, 0.4, 0.4],
    }

    router = UtilityRegressionRouter(alpha=0.0).fit(queries, utilities)
    predictions = router.predict(["brief", "a substantially longer scientific query"])

    assert predictions == ["bm25", "dense"]


def test_utility_router_returns_predicted_utility_for_each_strategy():
    queries = ["one", "two", "three"]
    utilities = {
        "bm25": [0.1, 0.2, 0.3],
        "dense": [0.3, 0.2, 0.1],
        "hybrid": [0.2, 0.2, 0.2],
    }

    router = UtilityRegressionRouter().fit(queries, utilities)
    predicted = router.predict_utilities(["one"])

    assert set(predicted) == {"bm25", "dense", "hybrid"}
    assert all(values.shape == (1,) for values in predicted.values())


def test_utility_router_validates_targets():
    router = UtilityRegressionRouter()

    with pytest.raises(ValueError, match="every strategy"):
        router.fit(
            ["query"],
            {"bm25": [0.1]},
        )

    with pytest.raises(ValueError, match="number of queries"):
        router.fit(
            ["query"],
            {
                "bm25": [],
                "dense": [0.1],
                "hybrid": [0.1],
            },
        )


def test_utility_router_rejects_negative_alpha():
    with pytest.raises(ValueError, match="nonnegative"):
        UtilityRegressionRouter(alpha=-1.0)
