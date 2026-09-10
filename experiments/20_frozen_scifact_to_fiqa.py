"""Frozen cross-domain routing evaluation: SciFact -> FiQA."""

from collections import Counter

from searchranklab.analysis import load_run_records
from searchranklab.datasets import load_scifact
from searchranklab.routing import (
    UtilityRegressionRouter,
    build_lexical_statistics,
    evaluate_routing,
)


SCIFACT_RUNS = {
    "bm25": "results/scifact/bm25_per_query.jsonl",
    "dense": "results/scifact/dense_per_query.jsonl",
    "hybrid": "results/scifact/hybrid_rrf_per_query.jsonl",
}

FIQA_RUNS = {
    "bm25": "results/fiqa/bm25_per_query.jsonl",
    "dense": "results/fiqa/dense_per_query.jsonl",
    "hybrid": "results/fiqa/hybrid_rrf_per_query.jsonl",
}

STRATEGIES = ("bm25", "dense", "hybrid")
COSTS = {"bm25": 1.0, "dense": 1.0, "hybrid": 2.0}
LAMBDA_VALUE = 0.10


def _load_runs(paths):
    return {
        strategy: load_run_records(path)
        for strategy, path in paths.items()
    }


def _common_query_ids(runs):
    return sorted(set.intersection(*(set(run) for run in runs.values())))


def _utility(record, strategy):
    return record.ndcg_at_10 - LAMBDA_VALUE * COSTS[strategy]


def _oracle_labels(runs, query_ids):
    return [
        max(
            STRATEGIES,
            key=lambda strategy: _utility(
                runs[strategy][query_id],
                strategy,
            ),
        )
        for query_id in query_ids
    ]


def main() -> None:
    source_runs = _load_runs(SCIFACT_RUNS)
    target_runs = _load_runs(FIQA_RUNS)

    source_ids = _common_query_ids(source_runs)
    target_ids = _common_query_ids(target_runs)

    source_queries = [
        source_runs["bm25"][query_id].query
        for query_id in source_ids
    ]
    target_queries = [
        target_runs["bm25"][query_id].query
        for query_id in target_ids
    ]

    source_utilities = {
        strategy: [
            _utility(source_runs[strategy][query_id], strategy)
            for query_id in source_ids
        ]
        for strategy in STRATEGIES
    }

    # Frozen lexical statistics are built from the source corpus only.
    # No FiQA corpus statistics, qrels, labels, or thresholds are used
    # to train or configure either learned router.
    scifact = load_scifact()
    source_lexical_stats = build_lexical_statistics(scifact.corpus)

    surface_router = UtilityRegressionRouter(
        strategies=STRATEGIES,
    ).fit(source_queries, source_utilities)

    lexical_router = UtilityRegressionRouter(
        strategies=STRATEGIES,
        lexical_stats=source_lexical_stats,
    ).fit(source_queries, source_utilities)

    target_oracle = _oracle_labels(target_runs, target_ids)

    predictions = {
        "always BM25": ["bm25"] * len(target_ids),
        "always dense": ["dense"] * len(target_ids),
        "always hybrid": ["hybrid"] * len(target_ids),
        "frozen surface": surface_router.predict(target_queries),
        "frozen lexical": lexical_router.predict(target_queries),
        "target oracle": target_oracle,
    }

    print("Frozen routing transfer / SciFact -> FiQA")
    print(f"lambda: {LAMBDA_VALUE}")
    print(f"source training queries: {len(source_ids)}")
    print(f"target evaluation queries: {len(target_ids)}")
    print("target tuning: none")
    print("lexical statistics: frozen from SciFact corpus")
    print()
    print(
        "strategy             NDCG@10   Recall@100   "
        "mean_cost   utility"
    )

    for name, selected in predictions.items():
        summary = evaluate_routing(
            query_ids=target_ids,
            predicted_strategies=selected,
            runs=target_runs,
            costs=COSTS,
            lambda_value=LAMBDA_VALUE,
        )
        print(
            f"{name:<20} "
            f"{summary.mean_ndcg_at_10:>7.4f}      "
            f"{summary.mean_recall_at_100:>7.4f}      "
            f"{summary.mean_cost:>7.4f}    "
            f"{summary.mean_utility:>7.4f}"
        )

    print()
    for name in ("frozen surface", "frozen lexical", "target oracle"):
        print(f"{name}: {dict(Counter(predictions[name]))}")


if __name__ == "__main__":
    main()
