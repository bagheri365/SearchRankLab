"""Train a first query-only pre-retrieval router on SciFact."""

from collections import Counter

from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split

from searchranklab.analysis import compute_cost_aware_oracle, load_run_records
from searchranklab.routing import PreRetrievalRouter, evaluate_routing


BM25_PATH = "results/scifact/bm25_per_query.jsonl"
DENSE_PATH = "results/scifact/dense_per_query.jsonl"
HYBRID_PATH = "results/scifact/hybrid_rrf_per_query.jsonl"

COSTS = {"bm25": 1.0, "dense": 1.0, "hybrid": 2.0}
LAMBDA_VALUE = 0.10
TEST_SIZE = 0.30
RANDOM_STATE = 42


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

    router = PreRetrievalRouter().fit(train_queries, train_labels)
    predictions = router.predict(test_queries)

    runs = {"bm25": bm25, "dense": dense, "hybrid": hybrid}
    routed = evaluate_routing(
        query_ids=test_ids,
        predicted_strategies=predictions,
        runs=runs,
        costs=COSTS,
        lambda_value=LAMBDA_VALUE,
    )
    always_bm25 = evaluate_routing(
        query_ids=test_ids,
        predicted_strategies=["bm25"] * len(test_ids),
        runs=runs,
        costs=COSTS,
        lambda_value=LAMBDA_VALUE,
    )
    oracle_test = evaluate_routing(
        query_ids=test_ids,
        predicted_strategies=test_labels,
        runs=runs,
        costs=COSTS,
        lambda_value=LAMBDA_VALUE,
    )

    print("Pre-retrieval router / SciFact")
    print(f"lambda: {LAMBDA_VALUE}")
    print(f"train queries: {len(train_ids)}")
    print(f"test queries: {len(test_ids)}")
    print(f"train labels: {dict(Counter(train_labels))}")
    print(f"test labels: {dict(Counter(test_labels))}")
    print(f"predicted labels: {dict(Counter(predictions))}")
    print(f"accuracy: {accuracy_score(test_labels, predictions):.4f}")
    print(
        f"balanced accuracy: "
        f"{balanced_accuracy_score(test_labels, predictions):.4f}"
    )
    print()
    print("strategy       NDCG@10   Recall@100   mean_cost   utility")
    print(
        f"always BM25    {always_bm25.mean_ndcg_at_10:.4f}     "
        f"{always_bm25.mean_recall_at_100:.4f}       "
        f"{always_bm25.mean_cost:.4f}     {always_bm25.mean_utility:.4f}"
    )
    print(
        f"learned router {routed.mean_ndcg_at_10:.4f}     "
        f"{routed.mean_recall_at_100:.4f}       "
        f"{routed.mean_cost:.4f}     {routed.mean_utility:.4f}"
    )
    print(
        f"oracle         {oracle_test.mean_ndcg_at_10:.4f}     "
        f"{oracle_test.mean_recall_at_100:.4f}       "
        f"{oracle_test.mean_cost:.4f}     {oracle_test.mean_utility:.4f}"
    )


if __name__ == "__main__":
    main()
