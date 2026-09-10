"""Cost-aware oracle routing analysis."""

from __future__ import annotations

from dataclasses import dataclass

from .disagreement import RunRecord


@dataclass(frozen=True)
class CostAwareDecision:
    query_id: str
    query: str
    strategy: str
    ndcg_at_10: float
    recall_at_100: float
    cost: float
    utility: float


@dataclass(frozen=True)
class CostAwareSummary:
    lambda_value: float
    queries: int
    mean_ndcg_at_10: float
    mean_recall_at_100: float
    mean_cost: float
    mean_utility: float
    bm25_selected: int
    dense_selected: int
    hybrid_selected: int


def compute_cost_aware_oracle(
    *,
    bm25: dict[str, RunRecord],
    dense: dict[str, RunRecord],
    hybrid: dict[str, RunRecord],
    costs: dict[str, float],
    lambda_value: float,
    tie_order: tuple[str, ...] = ("bm25", "dense", "hybrid"),
) -> list[CostAwareDecision]:
    """Choose the strategy maximizing NDCG@10 - lambda * cost."""

    strategies = {"bm25", "dense", "hybrid"}

    if set(costs) != strategies:
        raise ValueError("costs must define bm25, dense, and hybrid")
    if any(cost < 0 for cost in costs.values()):
        raise ValueError("costs must be nonnegative")
    if lambda_value < 0:
        raise ValueError("lambda_value must be nonnegative")
    if set(tie_order) != strategies:
        raise ValueError("tie_order must contain bm25, dense, and hybrid exactly once")

    common = sorted(set(bm25) & set(dense) & set(hybrid))
    decisions: list[CostAwareDecision] = []

    for query_id in common:
        records = {
            "bm25": bm25[query_id],
            "dense": dense[query_id],
            "hybrid": hybrid[query_id],
        }

        query_texts = {record.query for record in records.values()}
        if len(query_texts) != 1:
            raise ValueError(f"query text mismatch for query_id={query_id}")

        utilities = {
            name: record.ndcg_at_10 - lambda_value * costs[name]
            for name, record in records.items()
        }
        best_utility = max(utilities.values())
        strategy = next(
            name
            for name in tie_order
            if utilities[name] == best_utility
        )
        chosen = records[strategy]

        decisions.append(
            CostAwareDecision(
                query_id=query_id,
                query=chosen.query,
                strategy=strategy,
                ndcg_at_10=chosen.ndcg_at_10,
                recall_at_100=chosen.recall_at_100,
                cost=costs[strategy],
                utility=utilities[strategy],
            )
        )

    return decisions


def summarize_cost_aware_oracle(
    decisions: list[CostAwareDecision],
    *,
    lambda_value: float,
) -> CostAwareSummary:
    if not decisions:
        raise ValueError("decisions must not be empty")

    n = len(decisions)
    return CostAwareSummary(
        lambda_value=lambda_value,
        queries=n,
        mean_ndcg_at_10=sum(item.ndcg_at_10 for item in decisions) / n,
        mean_recall_at_100=sum(item.recall_at_100 for item in decisions) / n,
        mean_cost=sum(item.cost for item in decisions) / n,
        mean_utility=sum(item.utility for item in decisions) / n,
        bm25_selected=sum(item.strategy == "bm25" for item in decisions),
        dense_selected=sum(item.strategy == "dense" for item in decisions),
        hybrid_selected=sum(item.strategy == "hybrid" for item in decisions),
    )
