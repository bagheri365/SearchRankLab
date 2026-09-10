from searchranklab.analysis import (
    RunRecord,
    compare_runs,
    largest_bm25_wins,
    largest_dense_wins,
    summarize_disagreements,
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


def test_compare_runs_assigns_winners():
    bm25 = {
        "q1": _record("q1", 0.8, 1.0),
        "q2": _record("q2", 0.2, 0.0),
        "q3": _record("q3", 0.5, 1.0),
    }
    dense = {
        "q1": _record("q1", 0.3, 1.0),
        "q2": _record("q2", 0.7, 1.0),
        "q3": _record("q3", 0.5, 1.0),
    }

    comparisons = compare_runs(bm25, dense)

    assert [item.winner for item in comparisons] == ["bm25", "dense", "tie"]


def test_summary_counts_recall_recoveries():
    bm25 = {
        "q1": _record("q1", 0.0, 0.0),
        "q2": _record("q2", 0.7, 1.0),
    }
    dense = {
        "q1": _record("q1", 0.6, 1.0),
        "q2": _record("q2", 0.0, 0.0),
    }

    summary = summarize_disagreements(compare_runs(bm25, dense))

    assert summary.dense_recall_recoveries == 1
    assert summary.bm25_recall_recoveries == 1


def test_largest_win_helpers_order_by_delta():
    bm25 = {
        "q1": _record("q1", 0.8, 1.0),
        "q2": _record("q2", 0.1, 1.0),
        "q3": _record("q3", 0.5, 1.0),
    }
    dense = {
        "q1": _record("q1", 0.2, 1.0),
        "q2": _record("q2", 0.9, 1.0),
        "q3": _record("q3", 0.6, 1.0),
    }

    comparisons = compare_runs(bm25, dense)

    assert largest_dense_wins(comparisons, n=1)[0].query_id == "q2"
    assert largest_bm25_wins(comparisons, n=1)[0].query_id == "q1"
