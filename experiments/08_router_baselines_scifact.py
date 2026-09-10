"""Compare simple pre-retrieval routing baselines on SciFact."""

from collections import Counter

from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split

from searchranklab.analysis import compute_cost_aware_oracle, load_run_records
from searchranklab.routing import (
    PreRetrievalRouter,
    always_strategy,
    evaluate_routing,
    length_heuristic,
    random_strategy,
)


BM25_PATH = "results/scifact/bm25_per_query.jsonl"
DENSE_PATH = "results/scifact/dense_per_query.jsonl"
HYBRID_PATH = "results/scifact/hybrid_rrf_per_query.jsonl"

COSTS = {"bm25": 1.0, "dense": 1.0, "hybrid": 2.0}
LAMBDA_VALUE = 0.10
TEST_SIZE = 0.30
RANDOM_STATE = 42


def _evaluate(name, predictions, *, test_ids, test_labels, runs):
    summary = evaluate_routing(
        query_ids=test_ids,
        predicted_strategies=predictions,
        runs=runs,
        costs=COSTS,
        lambda_value=LAMBDA_VALUE,
    )
    accuracy = accuracy_score(test_labels, predictions)
    balanced = balanced_accuracy_score(test_labels, predictions)
    print(
        f"{name:<18} "
        f"{accuracy:>8.4f}   "
        f"{balanced:>8.4f}   "
        f"{summary.mean_ndcg_at_10:>7.4f}   "
        f"{summary.mean_recall_at_100:>10.4f}   "
        f"{summary.mean_cost:>9.4f}   "
        f"{summary.mean_utility:>7.4f}"
    )
    return summary


def main() -> None:
    bm25 = load_run_records(BM25_PATH)
    dense = load_run_records(DENSE_PATH)
    hybrid = load_run_records(HYBRID_PATH)

    oracle = compute_cost_aware_oracle(
        bm25=bm25,
        dense=dense,
        hybrid=hybrid,
        costs=COSTS,
        lambda_value=LAMBDA_VALUE,
    )

    query_ids = [item.query_id for item in oracle]
    queries = [item.query for item in oracle]
    labels = [item.strategy for item in oracle]

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
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    runs = {"bm25": bm25, "dense": dense, "hybrid": hybrid}

    predictions = {
        "always BM25": always_strategy(test_queries, "bm25"),
        "random": random_strategy(test_queries, seed=RANDOM_STATE),
        "length heuristic": length_heuristic(test_queries),
        "logreg balanced": (
            PreRetrievalRouter(class_weight="balanced")
            .fit(train_queries, train_labels)
            .predict(test_queries)
        ),
        "logreg unweighted": (
            PreRetrievalRouter(class_weight=None)
            .fit(train_queries, train_labels)
            .predict(test_queries)
        ),
        "oracle": list(test_labels),
    }

    print("Router baselines / SciFact")
    print(f"lambda: {LAMBDA_VALUE}")
    print(f"train queries: {len(train_ids)}")
    print(f"test queries: {len(test_ids)}")
    print(f"test labels: {dict(Counter(test_labels))}")
    print()
    print(
        "router              accuracy   balanced   NDCG@10   "
        "Recall@100   mean_cost   utility"
    )
    for name, predicted in predictions.items():
        _evaluate(
            name,
            predicted,
            test_ids=test_ids,
            test_labels=test_labels,
            runs=runs,
        )

    print()
    for name, predicted in predictions.items():
        print(f"{name:<18}: {dict(Counter(predicted))}")


if __name__ == "__main__":
    main()
