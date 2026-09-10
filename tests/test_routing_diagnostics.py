import pytest

from searchranklab.routing import feature_diagnostics, strongest_effects


def test_feature_diagnostics_returns_each_label_feature_pair():
    queries = [
        "short query",
        "tiny query",
        "this is a much longer scientific statement",
        "another considerably longer scientific statement",
    ]
    labels = ["bm25", "bm25", "dense", "dense"]

    diagnostics = feature_diagnostics(queries, labels)

    assert len(diagnostics) == 14
    assert {item.label for item in diagnostics} == {"bm25", "dense"}


def test_length_difference_produces_opposite_effect_directions():
    queries = [
        "short",
        "tiny",
        "this is a much longer query with many words",
        "another long scientific query containing many tokens",
    ]
    labels = ["bm25", "bm25", "dense", "dense"]

    diagnostics = feature_diagnostics(queries, labels)
    token_effects = {
        item.label: item.standardized_effect
        for item in diagnostics
        if item.feature == "token_count"
    }

    assert token_effects["bm25"] < 0
    assert token_effects["dense"] > 0


def test_strongest_effects_orders_by_absolute_effect():
    queries = [
        "short",
        "tiny",
        "this is a very long query with many many tokens",
        "another very long query with many many words",
    ]
    labels = ["bm25", "bm25", "dense", "dense"]

    diagnostics = feature_diagnostics(queries, labels)
    strongest = strongest_effects(diagnostics, n=3)

    magnitudes = [abs(item.standardized_effect) for item in strongest]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_feature_diagnostics_validates_inputs():
    with pytest.raises(ValueError, match="same length"):
        feature_diagnostics(["query"], [])

    with pytest.raises(ValueError, match="must not be empty"):
        feature_diagnostics([], [])

    with pytest.raises(ValueError, match="two distinct labels"):
        feature_diagnostics(["a", "b"], ["bm25", "bm25"])
