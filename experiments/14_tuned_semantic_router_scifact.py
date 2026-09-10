"""Nested-CV tuning for semantic utility routing on SciFact."""

from collections import Counter

from sklearn.model_selection import train_test_split

from searchranklab.analysis import load_run_records
from searchranklab.routing import (
    DEFAULT_ROUTER_MODEL,
    SeedResult,
    TunedSemanticUtilityRouter,
    evaluate_routing,
    load_query_encoder,
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

    encoder = load_query_encoder(DEFAULT_ROUTER_MODEL, device="cpu")
    results: list[SeedResult] = []
    configs = []

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

        router = TunedSemanticUtilityRouter(
            encoder=encoder,
            strategies=STRATEGIES,
            pca_components=(8, 16, 32, 64),
            alphas=(1.0, 10.0, 100.0),
            cv_splits=5,
            random_state=seed,
        ).fit(train_queries, train_utilities)
        configs.append(router.config)

        predictions = {
            "always BM25": ["bm25"] * len(test_ids),
            "semantic tuned": router.predict(test_queries),
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

    print("Tuned semantic utility router / SciFact")
    print(f"model: {DEFAULT_ROUTER_MODEL}")
    print(f"lambda: {LAMBDA_VALUE}")
    print(f"seeds: {SEEDS}")
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

    tuned_rows = [r for r in results if r.router == "semantic tuned"]
    bm25_rows = [r for r in results if r.router == "always BM25"]
    wins = sum(
        tuned.mean_utility > bm25.mean_utility
        for tuned, bm25 in zip(tuned_rows, bm25_rows, strict=True)
    )
    deltas = [
        tuned.mean_utility - bm25.mean_utility
        for tuned, bm25 in zip(tuned_rows, bm25_rows, strict=True)
    ]

    print()
    print(f"semantic tuned beats always-BM25 on {wins}/{len(SEEDS)} splits")
    print(f"mean semantic tuned utility delta vs BM25: {sum(deltas)/len(deltas):+.4f}")
    print(f"selected configs: {dict(Counter(configs))}")


if __name__ == "__main__":
    main()
