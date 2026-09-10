import numpy as np
import pytest

from searchranklab.analysis import RunRecord
from searchranklab.routing import (
    FEATURE_NAMES,
    PreRetrievalRouter,
    evaluate_routing,
    featurize_queries,
    query_features,
)


def _record(query_id, ndcg, recall):
    return RunRecord(
        query_id=query_id,
        query="example",
        recall_at_100=recall,
        mrr_at_10=0.0,
        ndcg_at_10=ndcg,
        relevant_doc_ids=["d1"],
        retrieved_doc_ids=[],
    )


def test_query_features_have_fixed_shape_and_are_query_only():
    values = query_features("Does TP53 affect 2 pathways?")

    assert values.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(values).all()
    assert values[0] == 5
    assert values[-1] == 1.0


def test_featurize_queries_returns_matrix():
    matrix = featurize_queries(["short query", "A much longer query 123"])

    assert matrix.shape == (2, len(FEATURE_NAMES))


def test_router_learns_simple_separable_pattern():
    queries = [
        "short",
        "tiny",
        "brief",
        "this is a much longer scientific query",
        "another substantially longer scientific question",
        "many words appear in this long biomedical statement",
    ]
    labels = ["bm25", "bm25", "bm25", "dense", "dense", "dense"]

    router = PreRetrievalRouter().fit(queries, labels)
    predictions = router.predict(["small", "this query contains many different words"])

    assert predictions == ["bm25", "dense"]


def test_evaluate_routing_computes_quality_cost_and_utility():
    runs = {
        "bm25": {
            "q1": _record("q1", 0.8, 1.0),
            "q2": _record("q2", 0.6, 1.0),
        },
        "dense": {
            "q1": _record("q1", 0.9, 1.0),
            "q2": _record("q2", 0.5, 0.0),
        },
        "hybrid": {
            "q1": _record("q1", 0.85, 1.0),
            "q2": _record("q2", 0.7, 1.0),
        },
    }

    summary = evaluate_routing(
        query_ids=["q1", "q2"],
        predicted_strategies=["dense", "hybrid"],
        runs=runs,
        costs={"bm25": 1.0, "dense": 1.0, "hybrid": 2.0},
        lambda_value=0.1,
    )

    assert summary.mean_ndcg_at_10 == pytest.approx(0.8)
    assert summary.mean_recall_at_100 == 1.0
    assert summary.mean_cost == 1.5
    assert summary.mean_utility == pytest.approx(0.65)
