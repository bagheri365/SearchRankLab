"""Measure the oracle routing ceiling on SciFact."""

from searchranklab.analysis import (
    compute_oracle_decisions,
    load_run_records,
    summarize_oracle,
)


BM25_PATH = "results/scifact/bm25_per_query.jsonl"
DENSE_PATH = "results/scifact/dense_per_query.jsonl"
HYBRID_PATH = "results/scifact/hybrid_rrf_per_query.jsonl"


def _mean(records, field):
    return sum(getattr(record, field) for record in records.values()) / len(records)


def main() -> None:
    bm25 = load_run_records(BM25_PATH)
    dense = load_run_records(DENSE_PATH)
    hybrid = load_run_records(HYBRID_PATH)

    decisions = compute_oracle_decisions(
        bm25=bm25,
        dense=dense,
        hybrid=hybrid,
    )
    oracle = summarize_oracle(decisions)

    print("Oracle routing ceiling / SciFact")
    print(f"queries: {oracle.queries}")
    print()
    print("Reference systems")
    print(
        f"BM25   NDCG@10={_mean(bm25, 'ndcg_at_10'):.4f} "
        f"Recall@100={_mean(bm25, 'recall_at_100'):.4f}"
    )
    print(
        f"Dense  NDCG@10={_mean(dense, 'ndcg_at_10'):.4f} "
        f"Recall@100={_mean(dense, 'recall_at_100'):.4f}"
    )
    print(
        f"Hybrid NDCG@10={_mean(hybrid, 'ndcg_at_10'):.4f} "
        f"Recall@100={_mean(hybrid, 'recall_at_100'):.4f}"
    )
    print()
    print("Oracle")
    print(f"NDCG@10: {oracle.mean_ndcg_at_10:.4f}")
    print(f"Recall@100: {oracle.mean_recall_at_100:.4f}")
    print(f"BM25 selected: {oracle.bm25_selected}")
    print(f"Dense selected: {oracle.dense_selected}")
    print(f"Hybrid selected: {oracle.hybrid_selected}")


if __name__ == "__main__":
    main()
