"""Evaluate routing robustness across multiple deterministic SciFact splits."""

from sklearn.model_selection import train_test_split

from searchranklab.analysis import load_run_records
from searchranklab.datasets import load_scifact
from searchranklab.routing import (
    SeedResult,
    UtilityRegressionRouter,
    build_lexical_statistics,
    evaluate_routing,
    summarize_seed_results,
)


BM25_PATH = "results/scifact/bm25_per_query.jsonl"
DENSE_PATH = "results/scifact/dense_per_query.jsonl"
HYBRID_PATH = "results/scifact/hybrid_rrf_per_query.jsonl"

COSTS = {"bm25": 1.0, "dense": 1.0, "hybrid": 2.0}
STRATEGIES = ("bm25", "dense", "hybrid")
LAMBDA_VALUE = 0.10
TEST_SIZE = 0.30
SEEDS = (7, 19, 42, 73, 101, 137, 211, 307, 401, 509)


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

    results: list[SeedResult] = []

    for seed in SEEDS:
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
            random_state=seed,
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
        ).fit(train_queries, train_utilities)

        lexical_router = UtilityRegressionRouter(
            strategies=STRATEGIES,
            lexical_stats=lexical_stats,
        ).fit(train_queries, train_utilities)

        predictions = {
            "always BM25": ["bm25"] * len(test_ids),
            "utility surface": surface_router.predict(test_queries),
            "utility lexical": lexical_router.predict(test_queries),
            "oracle": list(test_labels),
        }

        for name, selected in predictions.items():
            summary = evaluate_routing(
                query_ids=test_ids,
                predicted_strategies=selected,
                runs=runs,
                costs=COSTS,
                lambda_value=LAMBDA_VALUE,
            )
            results.append(
                SeedResult(
                    seed=seed,
                    router=name,
                    ndcg_at_10=summary.mean_ndcg_at_10,
                    recall_at_100=summary.mean_recall_at_100,
                    mean_cost=summary.mean_cost,
                    mean_utility=summary.mean_utility,
                )
            )

    print("Utility router robustness / SciFact")
    print(f"lambda: {LAMBDA_VALUE}")
    print(f"seeds: {SEEDS}")
    print(f"runs per router: {len(SEEDS)}")
    print()
    print(
        "router              NDCG@10 mean±std   Recall@100 mean±std   "
        "cost mean±std     utility mean±std"
    )

    for summary in summarize_seed_results(results):
        print(
            f"{summary.router:<18} "
            f"{summary.mean_ndcg_at_10:.4f}±{summary.std_ndcg_at_10:.4f}      "
            f"{summary.mean_recall_at_100:.4f}±{summary.std_recall_at_100:.4f}         "
            f"{summary.mean_cost:.4f}±{summary.std_cost:.4f}   "
            f"{summary.mean_utility:.4f}±{summary.std_utility:.4f}"
        )

    lexical_rows = [
        result for result in results if result.router == "utility lexical"
    ]
    bm25_rows = [
        result for result in results if result.router == "always BM25"
    ]
    lexical_wins = sum(
        lexical.mean_utility > bm25.mean_utility
        for lexical, bm25 in zip(lexical_rows, bm25_rows, strict=True)
    )
    utility_deltas = [
        lexical.mean_utility - bm25.mean_utility
        for lexical, bm25 in zip(lexical_rows, bm25_rows, strict=True)
    ]

    print()
    print(
        f"lexical utility beats always-BM25 on "
        f"{lexical_wins}/{len(SEEDS)} splits"
    )
    print(
        f"mean lexical utility delta vs BM25: "
        f"{sum(utility_deltas) / len(utility_deltas):+.4f}"
    )


if __name__ == "__main__":
    main()
