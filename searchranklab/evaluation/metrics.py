"""Ranking metrics for per-query and aggregate retrieval evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass

from searchranklab.retrieval import SearchResult


@dataclass(frozen=True)
class AggregateMetrics:
    recall_at_k: float
    mrr_at_k: float
    ndcg_at_k: float


def _relevant(qrels: dict[str, int]) -> set[str]:
    return {doc_id for doc_id, score in qrels.items() if score > 0}


def recall_at_k(results: list[SearchResult], qrels: dict[str, int], k: int) -> float:
    relevant = _relevant(qrels)
    if not relevant:
        return 0.0
    retrieved = {result.doc_id for result in results[:k]}
    return len(retrieved & relevant) / len(relevant)


def reciprocal_rank_at_k(results: list[SearchResult], qrels: dict[str, int], k: int) -> float:
    relevant = _relevant(qrels)
    for rank, result in enumerate(results[:k], start=1):
        if result.doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(results: list[SearchResult], qrels: dict[str, int], k: int) -> float:
    gains = [qrels.get(result.doc_id, 0) for result in results[:k]]
    dcg = sum((2**gain - 1) / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))

    ideal_gains = sorted((score for score in qrels.values() if score > 0), reverse=True)[:k]
    idcg = sum(
        (2**gain - 1) / math.log2(rank + 1)
        for rank, gain in enumerate(ideal_gains, start=1)
    )
    return dcg / idcg if idcg else 0.0


def evaluate_run(
    run: dict[str, list[SearchResult]],
    qrels: dict[str, dict[str, int]],
    *,
    k: int,
) -> AggregateMetrics:
    query_ids = [query_id for query_id in qrels if query_id in run]
    if not query_ids:
        raise ValueError("run and qrels have no queries in common")

    recalls = []
    mrrs = []
    ndcgs = []

    for query_id in query_ids:
        results = run[query_id]
        judgments = qrels[query_id]
        recalls.append(recall_at_k(results, judgments, k))
        mrrs.append(reciprocal_rank_at_k(results, judgments, k))
        ndcgs.append(ndcg_at_k(results, judgments, k))

    n = len(query_ids)
    return AggregateMetrics(
        recall_at_k=sum(recalls) / n,
        mrr_at_k=sum(mrrs) / n,
        ndcg_at_k=sum(ndcgs) / n,
    )
