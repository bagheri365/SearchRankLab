import pytest

from searchranklab.analysis import RunRecord, compute_oracle_decisions, summarize_oracle


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


def test_oracle_selects_best_ndcg_strategy():
    bm25 = {"q1": _record("q1", 0.7, 1.0)}
    dense = {"q1": _record("q1", 0.9, 1.0)}
    hybrid = {"q1": _record("q1", 0.8, 1.0)}

    decisions = compute_oracle_decisions(
        bm25=bm25,
        dense=dense,
        hybrid=hybrid,
    )

    assert decisions[0].strategy == "dense"
    assert decisions[0].ndcg_at_10 == 0.9


def test_oracle_uses_deterministic_tie_order():
    bm25 = {"q1": _record("q1", 1.0, 1.0)}
    dense = {"q1": _record("q1", 1.0, 1.0)}
    hybrid = {"q1": _record("q1", 1.0, 1.0)}

    decisions = compute_oracle_decisions(
        bm25=bm25,
        dense=dense,
        hybrid=hybrid,
    )

    assert decisions[0].strategy == "bm25"


def test_oracle_summary_aggregates_counts_and_metrics():
    bm25 = {
        "q1": _record("q1", 0.8, 1.0),
        "q2": _record("q2", 0.3, 0.0),
    }
    dense = {
        "q1": _record("q1", 0.5, 1.0),
        "q2": _record("q2", 0.9, 1.0),
    }
    hybrid = {
        "q1": _record("q1", 0.7, 1.0),
        "q2": _record("q2", 0.8, 1.0),
    }

    summary = summarize_oracle(
        compute_oracle_decisions(
            bm25=bm25,
            dense=dense,
            hybrid=hybrid,
        )
    )

    assert summary.queries == 2
    assert summary.mean_ndcg_at_10 == pytest.approx(0.85)
    assert summary.mean_recall_at_100 == 1.0
    assert summary.bm25_selected == 1
    assert summary.dense_selected == 1
    assert summary.hybrid_selected == 0


def test_oracle_rejects_invalid_tie_order():
    with pytest.raises(ValueError, match="tie_order"):
        compute_oracle_decisions(
            bm25={},
            dense={},
            hybrid={},
            tie_order=("bm25", "dense", "dense"),
        )
