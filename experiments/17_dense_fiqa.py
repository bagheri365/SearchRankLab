"""Run the dense retrieval baseline on FiQA."""

from time import perf_counter

from searchranklab.datasets import validate_dataset
from searchranklab.datasets.fiqa import load_fiqa
from searchranklab.evaluation import (
    evaluate_queries,
    evaluate_run,
    write_query_evaluations_jsonl,
)
from searchranklab.retrieval import DenseRetriever


MODEL_NAME = "sentence-transformers/msmarco-MiniLM-L6-cos-v5"
OUTPUT_PATH = "results/fiqa/dense_per_query.jsonl"


def main() -> None:
    dataset = load_fiqa()
    validate_dataset(dataset)

    started = perf_counter()
    retriever = DenseRetriever(
        dataset.corpus,
        model_name=MODEL_NAME,
    )
    embedding_seconds = perf_counter() - started

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

    print("Dense / FiQA")
    print(f"model: {MODEL_NAME}")
    print(f"documents: {len(dataset.corpus):,}")
    print(f"queries: {len(dataset.queries):,}")
    print(f"embedding/index time: {embedding_seconds:.3f}s")
    print(f"search time: {search_seconds:.3f}s")
    print(
        f"mean batched query latency: "
        f"{(search_seconds / len(dataset.queries)) * 1000:.2f} ms"
    )
    print(f"Recall@100: {recall100.recall_at_k:.4f}")
    print(f"MRR@10: {top10.mrr_at_k:.4f}")
    print(f"NDCG@10: {top10.ndcg_at_k:.4f}")
    print(f"per-query results: {output_path}")


if __name__ == "__main__":
    main()
