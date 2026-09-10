"""Oracle per-query strategy selection analysis."""

from __future__ import annotations

from dataclasses import dataclass

from .disagreement import RunRecord


@dataclass(frozen=True)
class OracleDecision:
    query_id: str
    query: str
    strategy: str
    ndcg_at_10: float
    recall_at_100: float


@dataclass(frozen=True)
class OracleSummary:
    queries: int
    mean_ndcg_at_10: float
    mean_recall_at_100: float
    bm25_selected: int
    dense_selected: int
    hybrid_selected: int


def compute_oracle_decisions(
    *,
    bm25: dict[str, RunRecord],
    dense: dict[str, RunRecord],
    hybrid: dict[str, RunRecord],
    tie_order: tuple[str, ...] = ("bm25", "dense", "hybrid"),
) -> list[OracleDecision]:
    """Select the highest-NDCG strategy for each query.

    Ties are resolved deterministically according to ``tie_order``.
    """

    valid = {"bm25", "dense", "hybrid"}
    if set(tie_order) != valid:
        raise ValueError("tie_order must contain bm25, dense, and hybrid exactly once")

    common = sorted(set(bm25) & set(dense) & set(hybrid))
    decisions: list[OracleDecision] = []

    for query_id in common:
        records = {
            "bm25": bm25[query_id],
            "dense": dense[query_id],
            "hybrid": hybrid[query_id],
        }

        query_texts = {record.query for record in records.values()}
        if len(query_texts) != 1:
            raise ValueError(f"query text mismatch for query_id={query_id}")

        best_ndcg = max(record.ndcg_at_10 for record in records.values())
        strategy = next(
            name
            for name in tie_order
            if records[name].ndcg_at_10 == best_ndcg
        )
        chosen = records[strategy]

        decisions.append(
            OracleDecision(
                query_id=query_id,
                query=chosen.query,
                strategy=strategy,
                ndcg_at_10=chosen.ndcg_at_10,
                recall_at_100=chosen.recall_at_100,
            )
        )

    return decisions


def summarize_oracle(decisions: list[OracleDecision]) -> OracleSummary:
    """Aggregate oracle quality and selection counts."""

    if not decisions:
        raise ValueError("decisions must not be empty")

    n = len(decisions)
    return OracleSummary(
        queries=n,
        mean_ndcg_at_10=sum(item.ndcg_at_10 for item in decisions) / n,
        mean_recall_at_100=sum(item.recall_at_100 for item in decisions) / n,
        bm25_selected=sum(item.strategy == "bm25" for item in decisions),
        dense_selected=sum(item.strategy == "dense" for item in decisions),
        hybrid_selected=sum(item.strategy == "hybrid" for item in decisions),
    )
