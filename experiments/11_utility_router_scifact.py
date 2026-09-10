"""Evaluate utility regression for pre-retrieval routing on SciFact."""

from collections import Counter

from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split

from searchranklab.analysis import load_run_records
from searchranklab.datasets import load_scifact
from searchranklab.routing import (
    UtilityRegressionRouter,
    build_lexical_statistics,
    evaluate_routing,
)


BM25_PATH = "results/scifact/bm25_per_query.jsonl"
DENSE_PATH = "results/scifact/dense_per_query.jsonl"
HYBRID_PATH = "results/scifact/hybrid_rrf_per_query.jsonl"

COSTS = {"bm25": 1.0, "dense": 1.0, "hybrid": 2.0}
STRATEGIES = ("bm25", "dense", "hybrid")
LAMBDA_VALUE = 0.10
TEST_SIZE = 0.30
RANDOM_STATE = 42


def _utility(record, strategy):
    return record.ndcg_at_10 - LAMBDA_VALUE * COSTS[strategy]


def main() -> None:
    dataset = load_scifact()
    lexical_stats = build_lexical_statistics(dataset.corpus)

    runs = {
        "bm25": load_run_records(BM25_PATH),
        "dense": load_run_records(DENSE_PATH),
        "hybrid": load_run_records(HYBRID_PATH),
    }

    query_ids = sorted(set.intersection(*(set(run) for run in runs.values())))
    queries = [runs["bm25"][query_id].query for query_id in query_ids]

    oracle_labels = [
        max(
            STRATEGIES,
            key=lambda strategy: _utility(runs[strategy][query_id], strategy),
        )
        for query_id in query_ids
    ]

    (
        train_ids,
        test_ids,
        train_queries,
        test_queries,
        train_labels,
        test_labels,
    ) = train_test_split(
        query_ids,
        queries,
        oracle_labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=oracle_labels,
    )

    train_utilities = {
        strategy: [
            _utility(runs[strategy][query_id], strategy)
            for query_id in train_ids
        ]
        for strategy in STRATEGIES
    }

    surface_router = UtilityRegressionRouter(
        strategies=STRATEGIES,
        lexical_stats=None,
    ).fit(train_queries, train_utilities)

    lexical_router = UtilityRegressionRouter(
        strategies=STRATEGIES,
        lexical_stats=lexical_stats,
    ).fit(train_queries, train_utilities)

    surface_predictions = surface_router.predict(test_queries)
    lexical_predictions = lexical_router.predict(test_queries)

    print("Utility-regression routing / SciFact")
    print(f"lambda: {LAMBDA_VALUE}")
    print(f"train queries: {len(train_ids)}")
    print(f"test queries: {len(test_ids)}")
    print(f"test oracle labels: {dict(Counter(test_labels))}")
    print()
    print(
        "router                    accuracy   balanced   NDCG@10   "
        "Recall@100   mean_cost   utility"
    )

    for name, predictions in (
        ("always BM25", ["bm25"] * len(test_ids)),
        ("utility surface", surface_predictions),
        ("utility lexical", lexical_predictions),
        ("oracle", test_labels),
    ):
        summary = evaluate_routing(
            query_ids=test_ids,
            predicted_strategies=predictions,
            runs=runs,
            costs=COSTS,
            lambda_value=LAMBDA_VALUE,
        )
        print(
            f"{name:<24} "
            f"{accuracy_score(test_labels, predictions):>8.4f}   "
            f"{balanced_accuracy_score(test_labels, predictions):>8.4f}   "
            f"{summary.mean_ndcg_at_10:>7.4f}   "
            f"{summary.mean_recall_at_100:>10.4f}   "
            f"{summary.mean_cost:>9.4f}   "
            f"{summary.mean_utility:>7.4f}"
        )

    print()
    print(f"utility surface: {dict(Counter(surface_predictions))}")
    print(f"utility lexical: {dict(Counter(lexical_predictions))}")


if __name__ == "__main__":
    main()
