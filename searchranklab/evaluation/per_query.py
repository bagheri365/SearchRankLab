"""Per-query evaluation records and JSONL export."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from searchranklab.retrieval import SearchResult

from .metrics import ndcg_at_k, recall_at_k, reciprocal_rank_at_k


@dataclass(frozen=True)
class QueryEvaluation:
    query_id: str
    query: str
    recall_at_100: float
    mrr_at_10: float
    ndcg_at_10: float
    relevant_doc_ids: list[str]
    retrieved_doc_ids: list[str]


def evaluate_queries(
    *,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    run: dict[str, list[SearchResult]],
) -> list[QueryEvaluation]:
    """Compute one deterministic evaluation record per judged query."""

    records: list[QueryEvaluation] = []
    for query_id in sorted(qrels):
        if query_id not in queries or query_id not in run:
            continue

        judgments = qrels[query_id]
        results = run[query_id]

        records.append(
            QueryEvaluation(
                query_id=query_id,
                query=queries[query_id],
                recall_at_100=recall_at_k(results, judgments, 100),
                mrr_at_10=reciprocal_rank_at_k(results, judgments, 10),
                ndcg_at_10=ndcg_at_k(results, judgments, 10),
                relevant_doc_ids=sorted(
                    doc_id for doc_id, score in judgments.items() if score > 0
                ),
                retrieved_doc_ids=[result.doc_id for result in results[:100]],
            )
        )

    return records


def write_query_evaluations_jsonl(
    records: list[QueryEvaluation],
    path: str | Path,
) -> Path:
    """Write per-query evaluation records as JSON Lines."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")

    return path
