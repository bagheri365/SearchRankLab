"""Per-query comparison utilities for retrieval systems."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunRecord:
    query_id: str
    query: str
    recall_at_100: float
    mrr_at_10: float
    ndcg_at_10: float
    relevant_doc_ids: list[str]
    retrieved_doc_ids: list[str]


@dataclass(frozen=True)
class QueryComparison:
    query_id: str
    query: str
    bm25_ndcg_at_10: float
    dense_ndcg_at_10: float
    ndcg_delta_dense_minus_bm25: float
    bm25_recall_at_100: float
    dense_recall_at_100: float
    recall_delta_dense_minus_bm25: float
    winner: str


@dataclass(frozen=True)
class DisagreementSummary:
    queries: int
    bm25_wins: int
    dense_wins: int
    ties: int
    dense_recall_recoveries: int
    bm25_recall_recoveries: int


def load_run_records(path: str | Path) -> dict[str, RunRecord]:
    """Load JSONL per-query results keyed by query id."""

    records: dict[str, RunRecord] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            record = RunRecord(
                query_id=str(row["query_id"]),
                query=row["query"],
                recall_at_100=float(row["recall_at_100"]),
                mrr_at_10=float(row["mrr_at_10"]),
                ndcg_at_10=float(row["ndcg_at_10"]),
                relevant_doc_ids=list(row["relevant_doc_ids"]),
                retrieved_doc_ids=list(row["retrieved_doc_ids"]),
            )
            records[record.query_id] = record
    return records


def compare_runs(
    bm25: dict[str, RunRecord],
    dense: dict[str, RunRecord],
    *,
    tie_tolerance: float = 1e-12,
) -> list[QueryComparison]:
    """Compare two runs on queries present in both."""

    comparisons: list[QueryComparison] = []

    for query_id in sorted(set(bm25) & set(dense)):
        sparse = bm25[query_id]
        semantic = dense[query_id]

        if sparse.query != semantic.query:
            raise ValueError(f"query text mismatch for query_id={query_id}")

        ndcg_delta = semantic.ndcg_at_10 - sparse.ndcg_at_10
        recall_delta = semantic.recall_at_100 - sparse.recall_at_100

        if ndcg_delta > tie_tolerance:
            winner = "dense"
        elif ndcg_delta < -tie_tolerance:
            winner = "bm25"
        else:
            winner = "tie"

        comparisons.append(
            QueryComparison(
                query_id=query_id,
                query=sparse.query,
                bm25_ndcg_at_10=sparse.ndcg_at_10,
                dense_ndcg_at_10=semantic.ndcg_at_10,
                ndcg_delta_dense_minus_bm25=ndcg_delta,
                bm25_recall_at_100=sparse.recall_at_100,
                dense_recall_at_100=semantic.recall_at_100,
                recall_delta_dense_minus_bm25=recall_delta,
                winner=winner,
            )
        )

    return comparisons


def summarize_disagreements(
    comparisons: list[QueryComparison],
) -> DisagreementSummary:
    """Summarize winner counts and recall-only recoveries."""

    return DisagreementSummary(
        queries=len(comparisons),
        bm25_wins=sum(item.winner == "bm25" for item in comparisons),
        dense_wins=sum(item.winner == "dense" for item in comparisons),
        ties=sum(item.winner == "tie" for item in comparisons),
        dense_recall_recoveries=sum(
            item.bm25_recall_at_100 == 0.0 and item.dense_recall_at_100 > 0.0
            for item in comparisons
        ),
        bm25_recall_recoveries=sum(
            item.dense_recall_at_100 == 0.0 and item.bm25_recall_at_100 > 0.0
            for item in comparisons
        ),
    )


def largest_dense_wins(
    comparisons: list[QueryComparison],
    *,
    n: int = 10,
) -> list[QueryComparison]:
    return sorted(
        comparisons,
        key=lambda item: (-item.ndcg_delta_dense_minus_bm25, item.query_id),
    )[:n]


def largest_bm25_wins(
    comparisons: list[QueryComparison],
    *,
    n: int = 10,
) -> list[QueryComparison]:
    return sorted(
        comparisons,
        key=lambda item: (item.ndcg_delta_dense_minus_bm25, item.query_id),
    )[:n]
