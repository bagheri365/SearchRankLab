"""Aggregation helpers for routing robustness experiments."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev


@dataclass(frozen=True)
class SeedResult:
    seed: int
    router: str
    ndcg_at_10: float
    recall_at_100: float
    mean_cost: float
    mean_utility: float


@dataclass(frozen=True)
class RobustnessSummary:
    router: str
    runs: int
    mean_ndcg_at_10: float
    std_ndcg_at_10: float
    mean_recall_at_100: float
    std_recall_at_100: float
    mean_cost: float
    std_cost: float
    mean_utility: float
    std_utility: float


def summarize_seed_results(results: list[SeedResult]) -> list[RobustnessSummary]:
    """Aggregate routing metrics by router across deterministic split seeds."""

    if not results:
        raise ValueError("results must not be empty")

    summaries: list[RobustnessSummary] = []
    routers = sorted({result.router for result in results})

    for router in routers:
        rows = [result for result in results if result.router == router]

        summaries.append(
            RobustnessSummary(
                router=router,
                runs=len(rows),
                mean_ndcg_at_10=mean(row.ndcg_at_10 for row in rows),
                std_ndcg_at_10=pstdev(row.ndcg_at_10 for row in rows),
                mean_recall_at_100=mean(row.recall_at_100 for row in rows),
                std_recall_at_100=pstdev(row.recall_at_100 for row in rows),
                mean_cost=mean(row.mean_cost for row in rows),
                std_cost=pstdev(row.mean_cost for row in rows),
                mean_utility=mean(row.mean_utility for row in rows),
                std_utility=pstdev(row.mean_utility for row in rows),
            )
        )

    return summaries
