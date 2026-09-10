"""Evaluate a cost-aware oracle routing frontier on SciFact."""

from searchranklab.analysis import (
    compute_cost_aware_oracle,
    load_run_records,
    summarize_cost_aware_oracle,
)


BM25_PATH = "results/scifact/bm25_per_query.jsonl"
DENSE_PATH = "results/scifact/dense_per_query.jsonl"
HYBRID_PATH = "results/scifact/hybrid_rrf_per_query.jsonl"

# Simple implementation-independent relative cost proxy:
# running one retriever costs 1 unit; hybrid runs both and costs 2.
COSTS = {
    "bm25": 1.0,
    "dense": 1.0,
    "hybrid": 2.0,
}

LAMBDA_VALUES = (0.0, 0.01, 0.025, 0.05, 0.10, 0.20)


def main() -> None:
    bm25 = load_run_records(BM25_PATH)
    dense = load_run_records(DENSE_PATH)
    hybrid = load_run_records(HYBRID_PATH)

    print("Cost-aware oracle frontier / SciFact")
    print(f"costs: {COSTS}")
    print()
    print(
        "lambda   NDCG@10   Recall@100   mean_cost   "
        "BM25   Dense   Hybrid"
    )

    for lambda_value in LAMBDA_VALUES:
        decisions = compute_cost_aware_oracle(
            bm25=bm25,
            dense=dense,
            hybrid=hybrid,
            costs=COSTS,
            lambda_value=lambda_value,
        )
        summary = summarize_cost_aware_oracle(
            decisions,
            lambda_value=lambda_value,
        )

        print(
            f"{summary.lambda_value:>6.3f}   "
            f"{summary.mean_ndcg_at_10:>7.4f}   "
            f"{summary.mean_recall_at_100:>10.4f}   "
            f"{summary.mean_cost:>9.4f}   "
            f"{summary.bm25_selected:>4}   "
            f"{summary.dense_selected:>5}   "
            f"{summary.hybrid_selected:>6}"
        )


if __name__ == "__main__":
    main()
