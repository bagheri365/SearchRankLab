"""Fuse FiQA BM25 and dense runs with reciprocal-rank fusion."""

from dataclasses import dataclass
from time import perf_counter

from searchranklab.analysis import load_run_records
from searchranklab.datasets import validate_dataset
from searchranklab.datasets.fiqa import load_fiqa
from searchranklab.evaluation import (
    evaluate_queries,
    evaluate_run,
    write_query_evaluations_jsonl,
)
from searchranklab.retrieval import fuse_runs


BM25_PATH = "results/fiqa/bm25_per_query.jsonl"
DENSE_PATH = "results/fiqa/dense_per_query.jsonl"
OUTPUT_PATH = "results/fiqa/hybrid_rrf_per_query.jsonl"
RRF_K = 60


def _record_doc_ids(record):
    """Return the stored ranked document IDs across supported record schemas."""

    for attribute in (
        "retrieved_doc_ids",
        "doc_ids",
        "ranking",
        "ranked_doc_ids",
    ):
        value = getattr(record, attribute, None)
        if value is not None:
            return list(value)

    available = ", ".join(sorted(vars(record)))
    raise AttributeError(
        "RunRecord does not expose a recognized ranked-document field; "
        f"available fields: {available}"
    )


@dataclass(frozen=True)
class _RankedDoc:
    """Minimal result object required by reciprocal-rank fusion."""

    doc_id: str


def _records_to_run(records):
    return {
        query_id: [_RankedDoc(doc_id) for doc_id in _record_doc_ids(record)]
        for query_id, record in records.items()
    }


def main() -> None:
    dataset = load_fiqa()
    validate_dataset(dataset)

    bm25_records = load_run_records(BM25_PATH)
    dense_records = load_run_records(DENSE_PATH)

    bm25_run = _records_to_run(bm25_records)
    dense_run = _records_to_run(dense_records)

    started = perf_counter()
    hybrid_run = fuse_runs(
        bm25_run,
        dense_run,
        k=100,
    )
    fusion_seconds = perf_counter() - started

    recall100 = evaluate_run(hybrid_run, dataset.qrels, k=100)
    top10 = evaluate_run(hybrid_run, dataset.qrels, k=10)

    records = evaluate_queries(
        queries=dataset.queries,
        qrels=dataset.qrels,
        run=hybrid_run,
    )
    output_path = write_query_evaluations_jsonl(records, OUTPUT_PATH)

    print("Hybrid RRF / FiQA")
    print(f"RRF k: {RRF_K}")
    print(f"documents: {len(dataset.corpus):,}")
    print(f"queries: {len(dataset.queries):,}")
    print(f"fusion time: {fusion_seconds:.3f}s")
    print(
        f"mean fusion latency: "
        f"{(fusion_seconds / len(dataset.queries)) * 1000:.2f} ms"
    )
    print(f"Recall@100: {recall100.recall_at_k:.4f}")
    print(f"MRR@10: {top10.mrr_at_k:.4f}")
    print(f"NDCG@10: {top10.ndcg_at_k:.4f}")
    print(f"per-query results: {output_path}")


if __name__ == "__main__":
    main()
