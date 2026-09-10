"""Run the first BM25 baseline on SciFact."""

from time import perf_counter

from searchranklab.datasets import load_scifact, validate_dataset
from searchranklab.evaluation import evaluate_run
from searchranklab.retrieval import BM25Retriever


def main() -> None:
    dataset = load_scifact()
    validate_dataset(dataset)

    started = perf_counter()
    retriever = BM25Retriever(dataset.corpus)
    index_seconds = perf_counter() - started

    started = perf_counter()
    run = retriever.batch_search(dataset.queries, k=100)
    search_seconds = perf_counter() - started

    recall100 = evaluate_run(run, dataset.qrels, k=100)
    top10 = evaluate_run(run, dataset.qrels, k=10)

    print("BM25 / SciFact")
    print(f"documents: {len(dataset.corpus):,}")
    print(f"queries: {len(dataset.queries):,}")
    print(f"index time: {index_seconds:.3f}s")
    print(f"search time: {search_seconds:.3f}s")
    print(f"mean query latency: {(search_seconds / len(dataset.queries)) * 1000:.2f} ms")
    print(f"Recall@100: {recall100.recall_at_k:.4f}")
    print(f"MRR@10: {top10.mrr_at_k:.4f}")
    print(f"NDCG@10: {top10.ndcg_at_k:.4f}")


if __name__ == "__main__":
    main()
