"""Evaluation helpers for routed retrieval decisions."""

from __future__ import annotations

from dataclasses import dataclass

from searchranklab.analysis import RunRecord


@dataclass(frozen=True)
class RoutedSummary:
    queries: int
    mean_ndcg_at_10: float
    mean_recall_at_100: float
    mean_cost: float
    mean_utility: float


def evaluate_routing(
    *,
    query_ids: list[str],
    predicted_strategies: list[str],
    runs: dict[str, dict[str, RunRecord]],
    costs: dict[str, float],
    lambda_value: float,
) -> RoutedSummary:
    if len(query_ids) != len(predicted_strategies):
        raise ValueError("query_ids and predicted_strategies must have the same length")
    if not query_ids:
        raise ValueError("routing evaluation must not be empty")

    ndcg = []
    recall = []
    selected_costs = []
    utilities = []

    for query_id, strategy in zip(query_ids, predicted_strategies, strict=True):
        if strategy not in runs:
            raise ValueError(f"unknown strategy: {strategy}")
        record = runs[strategy][query_id]
        cost = costs[strategy]
        ndcg.append(record.ndcg_at_10)
        recall.append(record.recall_at_100)
        selected_costs.append(cost)
        utilities.append(record.ndcg_at_10 - lambda_value * cost)

    n = len(query_ids)
    return RoutedSummary(
        queries=n,
        mean_ndcg_at_10=sum(ndcg) / n,
        mean_recall_at_100=sum(recall) / n,
        mean_cost=sum(selected_costs) / n,
        mean_utility=sum(utilities) / n,
    )
