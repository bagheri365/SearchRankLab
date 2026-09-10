"""Rank-fusion utilities for hybrid sparse+dense retrieval."""

from __future__ import annotations

from collections import defaultdict

from .bm25 import SearchResult


def reciprocal_rank_fusion(
    *rankings: list[SearchResult],
    k: int = 60,
    top_n: int | None = None,
) -> list[SearchResult]:
    """Fuse ranked lists with Reciprocal Rank Fusion (RRF).

    Each document receives:

        sum(1 / (k + rank))

    where rank starts at 1 for each input ranking.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    if top_n is not None and top_n <= 0:
        raise ValueError("top_n must be positive when provided")

    scores: dict[str, float] = defaultdict(float)

    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            scores[result.doc_id] += 1.0 / (k + rank)

    fused = sorted(
        (
            SearchResult(doc_id=doc_id, score=score)
            for doc_id, score in scores.items()
        ),
        key=lambda result: (-result.score, result.doc_id),
    )

    return fused if top_n is None else fused[:top_n]


def fuse_runs(
    bm25_run: dict[str, list[SearchResult]],
    dense_run: dict[str, list[SearchResult]],
    *,
    k: int = 60,
    top_n: int = 100,
) -> dict[str, list[SearchResult]]:
    """Fuse BM25 and dense runs for query ids present in both."""

    query_ids = sorted(set(bm25_run) & set(dense_run))
    return {
        query_id: reciprocal_rank_fusion(
            bm25_run[query_id],
            dense_run[query_id],
            k=k,
            top_n=top_n,
        )
        for query_id in query_ids
    }
