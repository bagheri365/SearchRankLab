import pytest

from searchranklab.evaluation import (
    evaluate_run,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from searchranklab.retrieval import SearchResult


def _results(*doc_ids):
    return [SearchResult(doc_id=doc_id, score=1.0) for doc_id in doc_ids]


def test_recall_at_k():
    qrels = {"d1": 1, "d2": 1}
    assert recall_at_k(_results("d1", "d3"), qrels, 2) == 0.5


def test_reciprocal_rank_at_k():
    qrels = {"d2": 1}
    assert reciprocal_rank_at_k(_results("d1", "d2"), qrels, 2) == 0.5


def test_ndcg_at_k_perfect_ranking_is_one():
    qrels = {"d1": 2, "d2": 1}
    assert ndcg_at_k(_results("d1", "d2"), qrels, 2) == pytest.approx(1.0)


def test_evaluate_run_averages_queries():
    run = {
        "q1": _results("d1"),
        "q2": _results("d3"),
    }
    qrels = {
        "q1": {"d1": 1},
        "q2": {"d2": 1},
    }

    metrics = evaluate_run(run, qrels, k=1)

    assert metrics.recall_at_k == 0.5
    assert metrics.mrr_at_k == 0.5
    assert metrics.ndcg_at_k == 0.5
