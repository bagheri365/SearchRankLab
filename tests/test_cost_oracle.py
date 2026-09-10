import pytest

from searchranklab.analysis import (
    RunRecord,
    compute_cost_aware_oracle,
    summarize_cost_aware_oracle,
)


def _record(query_id, ndcg, recall, query="example"):
    return RunRecord(
        query_id=query_id,
        query=query,
        recall_at_100=recall,
        mrr_at_10=0.0,
        ndcg_at_10=ndcg,
        relevant_doc_ids=["d1"],
        retrieved_doc_ids=[],
    )


def test_zero_lambda_matches_best_relevance_choice():
    bm25 = {"q1": _record("q1", 0.7, 1.0)}
    dense = {"q1": _record("q1", 0.9, 1.0)}
    hybrid = {"q1": _record("q1", 0.8, 1.0)}

    decisions = compute_cost_aware_oracle(
        bm25=bm25,
        dense=dense,
        hybrid=hybrid,
        costs={"bm25": 1.0, "dense": 1.0, "hybrid": 2.0},
        lambda_value=0.0,
    )

    assert decisions[0].strategy == "dense"


def test_cost_penalty_can_flip_hybrid_to_bm25():
    bm25 = {"q1": _record("q1", 0.80, 1.0)}
    dense = {"q1": _record("q1", 0.60, 1.0)}
    hybrid = {"q1": _record("q1", 0.85, 1.0)}

    decisions = compute_cost_aware_oracle(
        bm25=bm25,
        dense=dense,
        hybrid=hybrid,
        costs={"bm25": 1.0, "dense": 1.0, "hybrid": 2.0},
        lambda_value=0.10,
    )

    assert decisions[0].strategy == "bm25"


def test_summary_reports_mean_cost_and_selection_counts():
    bm25 = {
        "q1": _record("q1", 0.8, 1.0),
        "q2": _record("q2", 0.4, 0.0),
    }
    dense = {
        "q1": _record("q1", 0.5, 1.0),
        "q2": _record("q2", 0.9, 1.0),
    }
    hybrid = {
        "q1": _record("q1", 0.7, 1.0),
        "q2": _record("q2", 0.8, 1.0),
    }

    decisions = compute_cost_aware_oracle(
        bm25=bm25,
        dense=dense,
        hybrid=hybrid,
        costs={"bm25": 1.0, "dense": 1.0, "hybrid": 2.0},
        lambda_value=0.0,
    )
    summary = summarize_cost_aware_oracle(decisions, lambda_value=0.0)

    assert summary.queries == 2
    assert summary.mean_cost == 1.0
    assert summary.bm25_selected == 1
    assert summary.dense_selected == 1
    assert summary.hybrid_selected == 0


def test_cost_oracle_validates_inputs():
    empty = {}

    with pytest.raises(ValueError, match="costs"):
        compute_cost_aware_oracle(
            bm25=empty,
            dense=empty,
            hybrid=empty,
            costs={"bm25": 1.0},
            lambda_value=0.0,
        )

    with pytest.raises(ValueError, match="nonnegative"):
        compute_cost_aware_oracle(
            bm25=empty,
            dense=empty,
            hybrid=empty,
            costs={"bm25": 1.0, "dense": 1.0, "hybrid": 2.0},
            lambda_value=-0.1,
        )
