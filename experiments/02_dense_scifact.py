"""Run the first dense retrieval baseline on SciFact."""

from time import perf_counter

from searchranklab.datasets import load_scifact, validate_dataset
from searchranklab.evaluation import (
    evaluate_queries,
    evaluate_run,
    write_query_evaluations_jsonl,
)
from searchranklab.retrieval import DEFAULT_MODEL, DenseRetriever


OUTPUT_PATH = "results/scifact/dense_per_query.jsonl"


def main() -> None:
    dataset = load_scifact()
    validate_dataset(dataset)

    started = perf_counter()
    retriever = DenseRetriever(
        dataset.corpus,
        model_name=DEFAULT_MODEL,
        device="cpu",
    )
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

    print("Dense / SciFact")
    print(f"model: {DEFAULT_MODEL}")
    print(f"documents: {len(dataset.corpus):,}")
    print(f"queries: {len(dataset.queries):,}")
    print(f"embedding/index time: {index_seconds:.3f}s")
    print(f"search time: {search_seconds:.3f}s")
    print(f"mean query latency: {(search_seconds / len(dataset.queries)) * 1000:.2f} ms")
    print(f"Recall@100: {recall100.recall_at_k:.4f}")
    print(f"MRR@10: {top10.mrr_at_k:.4f}")
    print(f"NDCG@10: {top10.ndcg_at_k:.4f}")
    print(f"per-query results: {output_path}")


if __name__ == "__main__":
    main()
