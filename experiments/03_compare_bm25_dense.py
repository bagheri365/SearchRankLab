"""Compare BM25 and dense SciFact runs at the query level."""

from searchranklab.analysis import (
    compare_runs,
    largest_bm25_wins,
    largest_dense_wins,
    load_run_records,
    summarize_disagreements,
)


BM25_PATH = "results/scifact/bm25_per_query.jsonl"
DENSE_PATH = "results/scifact/dense_per_query.jsonl"


def _print_examples(title, examples):
    print()
    print(title)
    for item in examples:
        print(
            f"{item.query_id}: "
            f"ΔNDCG={item.ndcg_delta_dense_minus_bm25:+.4f} | "
            f"BM25={item.bm25_ndcg_at_10:.4f} | "
            f"Dense={item.dense_ndcg_at_10:.4f}"
        )
        print(f"  {item.query}")


def main() -> None:
    bm25 = load_run_records(BM25_PATH)
    dense = load_run_records(DENSE_PATH)

    comparisons = compare_runs(bm25, dense)
    summary = summarize_disagreements(comparisons)

    print("BM25 vs Dense / SciFact")
    print(f"queries compared: {summary.queries}")
    print(f"BM25 wins by NDCG@10: {summary.bm25_wins}")
    print(f"Dense wins by NDCG@10: {summary.dense_wins}")
    print(f"ties: {summary.ties}")
    print(f"Dense-only Recall@100 recoveries: {summary.dense_recall_recoveries}")
    print(f"BM25-only Recall@100 recoveries: {summary.bm25_recall_recoveries}")

    _print_examples(
        "Largest dense wins",
        largest_dense_wins(comparisons, n=5),
    )
    _print_examples(
        "Largest BM25 wins",
        largest_bm25_wins(comparisons, n=5),
    )


if __name__ == "__main__":
    main()
