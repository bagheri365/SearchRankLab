"""Run BM25+dense Reciprocal Rank Fusion on SciFact."""

from time import perf_counter

from searchranklab.datasets import load_scifact, validate_dataset
from searchranklab.evaluation import (
    evaluate_queries,
    evaluate_run,
    write_query_evaluations_jsonl,
)
from searchranklab.retrieval import (
    BM25Retriever,
    DEFAULT_MODEL,
    DenseRetriever,
    fuse_runs,
)


OUTPUT_PATH = "results/scifact/hybrid_rrf_per_query.jsonl"
RRF_K = 60


def main() -> None:
    dataset = load_scifact()
    validate_dataset(dataset)

    bm25 = BM25Retriever(dataset.corpus)
    dense = DenseRetriever(
        dataset.corpus,
        model_name=DEFAULT_MODEL,
        device="cpu",
    )

    started = perf_counter()
    bm25_run = bm25.batch_search(dataset.queries, k=100)
    dense_run = dense.batch_search(dataset.queries, k=100)
    hybrid_run = fuse_runs(
        bm25_run,
        dense_run,
        k=RRF_K,
        top_n=100,
    )
    search_seconds = perf_counter() - started

    recall100 = evaluate_run(hybrid_run, dataset.qrels, k=100)
    top10 = evaluate_run(hybrid_run, dataset.qrels, k=10)

    records = evaluate_queries(
        queries=dataset.queries,
        qrels=dataset.qrels,
        run=hybrid_run,
    )
    output_path = write_query_evaluations_jsonl(records, OUTPUT_PATH)

    print("Hybrid RRF / SciFact")
    print(f"model: BM25 + {DEFAULT_MODEL}")
    print(f"RRF k: {RRF_K}")
    print(f"documents: {len(dataset.corpus):,}")
    print(f"queries: {len(dataset.queries):,}")
    print(f"combined search time: {search_seconds:.3f}s")
    print(f"mean query latency: {(search_seconds / len(dataset.queries)) * 1000:.2f} ms")
    print(f"Recall@100: {recall100.recall_at_k:.4f}")
    print(f"MRR@10: {top10.mrr_at_k:.4f}")
    print(f"NDCG@10: {top10.ndcg_at_k:.4f}")
    print(f"per-query results: {output_path}")


if __name__ == "__main__":
    main()
