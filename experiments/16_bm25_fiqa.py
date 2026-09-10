"""Run the BM25 baseline on FiQA."""

from time import perf_counter

from searchranklab.datasets import validate_dataset
from searchranklab.datasets.fiqa import load_fiqa
from searchranklab.evaluation import (
    evaluate_queries,
    evaluate_run,
    write_query_evaluations_jsonl,
)
from searchranklab.retrieval import BM25Retriever


OUTPUT_PATH = "results/fiqa/bm25_per_query.jsonl"


def main() -> None:
    dataset = load_fiqa()
    validate_dataset(dataset)

    started = perf_counter()
    retriever = BM25Retriever(dataset.corpus)
    index_seconds = perf_counter() - started

    started = perf_counter()
    run = retriever.batch_search(dataset.queries, k=100)
    search_seconds = perf_counter() - started

    recall100 = evaluate_run(run, dataset.qrels, k=100)
    top10 = evaluate_run(run, dataset.qrels, k=10)

    records = evaluate_queries(
        queries=dataset.queries,
        qrels=dataset.qrels,
        run=run,
    )
    output_path = write_query_evaluations_jsonl(records, OUTPUT_PATH)

    print("BM25 / FiQA")
    print(f"documents: {len(dataset.corpus):,}")
    print(f"queries: {len(dataset.queries):,}")
    print(f"index time: {index_seconds:.3f}s")
    print(f"search time: {search_seconds:.3f}s")
    print(
        f"mean query latency: "
        f"{(search_seconds / len(dataset.queries)) * 1000:.2f} ms"
    )
    print(f"Recall@100: {recall100.recall_at_k:.4f}")
    print(f"MRR@10: {top10.mrr_at_k:.4f}")
    print(f"NDCG@10: {top10.ndcg_at_k:.4f}")
    print(f"per-query results: {output_path}")


if __name__ == "__main__":
    main()
