"""Compute relevance-only and cost-aware routing ceilings on FiQA."""

from collections import Counter

from searchranklab.analysis import load_run_records


BM25_PATH = "results/fiqa/bm25_per_query.jsonl"
DENSE_PATH = "results/fiqa/dense_per_query.jsonl"
HYBRID_PATH = "results/fiqa/hybrid_rrf_per_query.jsonl"

STRATEGIES = ("bm25", "dense", "hybrid")
COSTS = {"bm25": 1.0, "dense": 1.0, "hybrid": 2.0}
LAMBDAS = (0.0, 0.01, 0.025, 0.05, 0.10, 0.20)


def _load_runs():
    return {
        "bm25": load_run_records(BM25_PATH),
        "dense": load_run_records(DENSE_PATH),
        "hybrid": load_run_records(HYBRID_PATH),
    }


def _common_query_ids(runs):
    return sorted(set.intersection(*(set(run) for run in runs.values())))


def _choose_strategy(runs, query_id, lambda_value):
    return max(
        STRATEGIES,
        key=lambda strategy: (
            runs[strategy][query_id].ndcg_at_10
            - lambda_value * COSTS[strategy]
        ),
    )


def _summarize(runs, query_ids, lambda_value):
    selected = [
        _choose_strategy(runs, query_id, lambda_value)
        for query_id in query_ids
    ]

    mean_ndcg = sum(
        runs[strategy][query_id].ndcg_at_10
        for query_id, strategy in zip(query_ids, selected, strict=True)
    ) / len(query_ids)

    mean_recall = sum(
        runs[strategy][query_id].recall_at_100
        for query_id, strategy in zip(query_ids, selected, strict=True)
    ) / len(query_ids)

    mean_cost = sum(COSTS[strategy] for strategy in selected) / len(selected)
    mean_utility = mean_ndcg - lambda_value * mean_cost
    counts = Counter(selected)

    return mean_ndcg, mean_recall, mean_cost, mean_utility, counts


def main() -> None:
    runs = _load_runs()
    query_ids = _common_query_ids(runs)

    print("Oracle routing ceiling / FiQA")
    print(f"queries: {len(query_ids)}")
    print()
    print(
        "lambda   NDCG@10   Recall@100   mean_cost   utility   "
        "BM25   Dense   Hybrid"
    )

    for lambda_value in LAMBDAS:
        ndcg, recall, cost, utility, counts = _summarize(
            runs,
            query_ids,
            lambda_value,
        )
        print(
            f"{lambda_value:>5.3f}    "
            f"{ndcg:>7.4f}      "
            f"{recall:>7.4f}      "
            f"{cost:>7.4f}    "
            f"{utility:>7.4f}   "
            f"{counts['bm25']:>4}   "
            f"{counts['dense']:>5}   "
            f"{counts['hybrid']:>6}"
        )


if __name__ == "__main__":
    main()
