"""Cross-encoder reranking of SciFact hybrid candidates."""

from dataclasses import dataclass
from time import perf_counter

from sentence_transformers import CrossEncoder

from searchranklab.analysis import load_run_records
from searchranklab.datasets import load_scifact, validate_dataset
from searchranklab.evaluation import (
    evaluate_queries,
    evaluate_run,
    write_query_evaluations_jsonl,
)


HYBRID_PATH = "results/scifact/hybrid_rrf_per_query.jsonl"
OUTPUT_PATH = "results/scifact/hybrid_cross_encoder_per_query.jsonl"
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATES = 100


@dataclass(frozen=True)
class RankedResult:
    doc_id: str
    score: float


def _record_doc_ids(record):
    """Return stored ranked document IDs across supported record schemas."""

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


def _document_text(document):
    title = document.get("title", "").strip()
    text = document.get("text", "").strip()
    return f"{title} {text}".strip()


def main() -> None:
    dataset = load_scifact()
    validate_dataset(dataset)
    hybrid_records = load_run_records(HYBRID_PATH)

    model = CrossEncoder(MODEL_NAME)

    reranked_run = {}
    started = perf_counter()

    for query_id, record in hybrid_records.items():
        query = record.query
        candidate_ids = _record_doc_ids(record)[:CANDIDATES]

        pairs = [
            (query, _document_text(dataset.corpus[doc_id]))
            for doc_id in candidate_ids
        ]
        scores = model.predict(
            pairs,
            show_progress_bar=False,
        )

        ranked = sorted(
            (
                RankedResult(doc_id=doc_id, score=float(score))
                for doc_id, score in zip(candidate_ids, scores, strict=True)
            ),
            key=lambda result: result.score,
            reverse=True,
        )
        reranked_run[query_id] = ranked

    rerank_seconds = perf_counter() - started

    recall100 = evaluate_run(reranked_run, dataset.qrels, k=100)
    top10 = evaluate_run(reranked_run, dataset.qrels, k=10)

    records = evaluate_queries(
        queries=dataset.queries,
        qrels=dataset.qrels,
        run=reranked_run,
    )
    output_path = write_query_evaluations_jsonl(records, OUTPUT_PATH)

    print("Cross-encoder reranking / SciFact")
    print(f"model: {MODEL_NAME}")
    print(f"candidate source: hybrid RRF")
    print(f"candidates per query: {CANDIDATES}")
    print(f"queries: {len(reranked_run):,}")
    print(f"rerank time: {rerank_seconds:.3f}s")
    print(
        f"mean rerank latency: "
        f"{(rerank_seconds / len(reranked_run)) * 1000:.2f} ms"
    )
    print(f"Recall@100: {recall100.recall_at_k:.4f}")
    print(f"MRR@10: {top10.mrr_at_k:.4f}")
    print(f"NDCG@10: {top10.ndcg_at_k:.4f}")
    print(f"per-query results: {output_path}")


if __name__ == "__main__":
    main()
