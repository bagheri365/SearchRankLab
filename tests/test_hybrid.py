import pytest

from searchranklab.retrieval import SearchResult, fuse_runs, reciprocal_rank_fusion


def _ranking(*doc_ids):
    return [
        SearchResult(doc_id=doc_id, score=float(len(doc_ids) - index))
        for index, doc_id in enumerate(doc_ids)
    ]


def test_rrf_rewards_documents_present_in_both_rankings():
    fused = reciprocal_rank_fusion(
        _ranking("d1", "d2", "d3"),
        _ranking("d2", "d4", "d1"),
        k=60,
    )

    assert fused[0].doc_id == "d2"


def test_rrf_top_n_limits_output():
    fused = reciprocal_rank_fusion(
        _ranking("d1", "d2"),
        _ranking("d3", "d4"),
        top_n=2,
    )

    assert len(fused) == 2


def test_fuse_runs_uses_common_queries_only():
    bm25 = {
        "q1": _ranking("d1"),
        "q2": _ranking("d2"),
    }
    dense = {
        "q1": _ranking("d3"),
        "q3": _ranking("d4"),
    }

    fused = fuse_runs(bm25, dense)

    assert set(fused) == {"q1"}


def test_rrf_rejects_invalid_parameters():
    with pytest.raises(ValueError, match="positive"):
        reciprocal_rank_fusion(_ranking("d1"), k=0)

    with pytest.raises(ValueError, match="positive"):
        reciprocal_rank_fusion(_ranking("d1"), top_n=0)
