"""Print the consolidated headline results for SearchRankLab."""

from collections import Counter

from searchranklab.analysis import load_run_records
from searchranklab.datasets import load_scifact
from searchranklab.datasets.fiqa import load_fiqa


RUNS = {
    "SciFact": {
        "bm25": "results/scifact/bm25_per_query.jsonl",
        "dense": "results/scifact/dense_per_query.jsonl",
        "hybrid": "results/scifact/hybrid_rrf_per_query.jsonl",
        "reranked": "results/scifact/hybrid_cross_encoder_per_query.jsonl",
        "dataset": load_scifact,
    },
    "FiQA": {
        "bm25": "results/fiqa/bm25_per_query.jsonl",
        "dense": "results/fiqa/dense_per_query.jsonl",
        "hybrid": "results/fiqa/hybrid_rrf_per_query.jsonl",
        "reranked": "results/fiqa/hybrid_cross_encoder_per_query.jsonl",
        "dataset": load_fiqa,
    },
}

STRATEGIES = ("bm25", "dense", "hybrid")
COSTS = {"bm25": 1.0, "dense": 1.0, "hybrid": 2.0}
LAMBDA_VALUE = 0.10


def _common_query_ids(runs):
    return sorted(set.intersection(*(set(run) for run in runs.values())))


def _mean(records, query_ids, attribute):
    return sum(getattr(records[query_id], attribute) for query_id in query_ids) / len(query_ids)


def _oracle_summary(runs, query_ids):
    selected = []
    for query_id in query_ids:
        selected.append(
            max(
                STRATEGIES,
                key=lambda strategy: (
                    runs[strategy][query_id].ndcg_at_10
                    - LAMBDA_VALUE * COSTS[strategy]
                ),
            )
        )

    mean_ndcg = sum(
        runs[strategy][query_id].ndcg_at_10
        for query_id, strategy in zip(query_ids, selected, strict=True)
    ) / len(query_ids)
    mean_recall = sum(
        runs[strategy][query_id].recall_at_100
        for query_id, strategy in zip(query_ids, selected, strict=True)
    ) / len(query_ids)
    mean_cost = sum(COSTS[strategy] for strategy in selected) / len(selected)
    utility = mean_ndcg - LAMBDA_VALUE * mean_cost

    return mean_ndcg, mean_recall, mean_cost, utility, Counter(selected)


def _record_doc_ids(record):
    for attribute in ("retrieved_doc_ids", "doc_ids", "ranking", "ranked_doc_ids"):
        value = getattr(record, attribute, None)
        if value is not None:
            return list(value)
    raise AttributeError("No ranked-document field found")


def _failure_counts(dataset, hybrid, reranked):
    counts = Counter()
    query_ids = sorted(set(hybrid) & set(reranked) & set(dataset.qrels))

    for query_id in query_ids:
        relevant = {
            doc_id
            for doc_id, relevance in dataset.qrels[query_id].items()
            if relevance > 0
        }
        if not relevant:
            continue

        hybrid_ids = _record_doc_ids(hybrid[query_id])[:100]
        reranked_ids = _record_doc_ids(reranked[query_id])[:100]

        hybrid_rank = next(
            (rank for rank, doc_id in enumerate(hybrid_ids, start=1) if doc_id in relevant),
            None,
        )
        reranked_rank = next(
            (rank for rank, doc_id in enumerate(reranked_ids, start=1) if doc_id in relevant),
            None,
        )

        if hybrid_rank is None:
            counts["retrieval failure"] += 1
        elif reranked_rank is not None and reranked_rank <= 10:
            if hybrid_rank <= 10:
                counts["already top-10"] += 1
            else:
                counts["ranking fixed"] += 1
        else:
            counts["ranking failure"] += 1

    return counts


def main() -> None:
    print("SearchRankLab consolidated results")
    print(f"cost-aware lambda: {LAMBDA_VALUE}")
    print()

    for dataset_name, config in RUNS.items():
        runs = {
            name: load_run_records(path)
            for name, path in config.items()
            if name != "dataset"
        }
        query_ids = _common_query_ids(runs)

        print(dataset_name)
        print(
            "system                 NDCG@10   MRR@10   Recall@100"
        )
        for system in ("bm25", "dense", "hybrid", "reranked"):
            records = runs[system]
            print(
                f"{system:<22} "
                f"{_mean(records, query_ids, 'ndcg_at_10'):>7.4f}   "
                f"{_mean(records, query_ids, 'mrr_at_10'):>7.4f}   "
                f"{_mean(records, query_ids, 'recall_at_100'):>10.4f}"
            )

        oracle_ndcg, oracle_recall, oracle_cost, oracle_utility, oracle_counts = _oracle_summary(
            {strategy: runs[strategy] for strategy in STRATEGIES},
            query_ids,
        )
        print()
        print(
            f"cost-aware oracle: NDCG@10={oracle_ndcg:.4f}, "
            f"Recall@100={oracle_recall:.4f}, "
            f"mean_cost={oracle_cost:.4f}, "
            f"utility={oracle_utility:.4f}"
        )
        print(f"oracle selections: {dict(oracle_counts)}")

        dataset = config["dataset"]()
        failures = _failure_counts(dataset, runs["hybrid"], runs["reranked"])
        total = sum(failures.values())
        print("failure attribution:")
        for label in (
            "retrieval failure",
            "ranking failure",
            "ranking fixed",
            "already top-10",
        ):
            count = failures[label]
            pct = 100.0 * count / total if total else 0.0
            print(f"  {label:<18} {count:>4} ({pct:>5.1f}%)")

        print()
        print(
            f"reranking delta vs hybrid: "
            f"NDCG@10="
            f"{_mean(runs['reranked'], query_ids, 'ndcg_at_10') - _mean(runs['hybrid'], query_ids, 'ndcg_at_10'):+.4f}, "
            f"MRR@10="
            f"{_mean(runs['reranked'], query_ids, 'mrr_at_10') - _mean(runs['hybrid'], query_ids, 'mrr_at_10'):+.4f}"
        )
        print("-" * 72)


if __name__ == "__main__":
    main()
