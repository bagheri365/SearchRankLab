"""Attribute errors to candidate retrieval versus ranking quality."""

from collections import Counter

from searchranklab.analysis import load_run_records
from searchranklab.datasets import load_scifact
from searchranklab.datasets.fiqa import load_fiqa


DATASETS = {
    "SciFact": {
        "dataset": load_scifact,
        "hybrid": "results/scifact/hybrid_rrf_per_query.jsonl",
        "reranked": "results/scifact/hybrid_cross_encoder_per_query.jsonl",
    },
    "FiQA": {
        "dataset": load_fiqa,
        "hybrid": "results/fiqa/hybrid_rrf_per_query.jsonl",
        "reranked": "results/fiqa/hybrid_cross_encoder_per_query.jsonl",
    },
}


def _record_doc_ids(record):
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


def _relevant_ids(qrels_for_query):
    return {
        doc_id
        for doc_id, relevance in qrels_for_query.items()
        if relevance > 0
    }


def _best_rank(doc_ids, relevant_ids):
    for rank, doc_id in enumerate(doc_ids, start=1):
        if doc_id in relevant_ids:
            return rank
    return None


def _classify(hybrid_ids, reranked_ids, relevant_ids):
    hybrid_rank = _best_rank(hybrid_ids, relevant_ids)
    reranked_rank = _best_rank(reranked_ids, relevant_ids)

    if hybrid_rank is None:
        return "retrieval failure"

    if reranked_rank is not None and reranked_rank <= 10:
        if hybrid_rank <= 10:
            return "already top-10"
        return "ranking fixed"

    return "ranking failure"


def analyze(name, config):
    dataset = config["dataset"]()
    hybrid = load_run_records(config["hybrid"])
    reranked = load_run_records(config["reranked"])

    query_ids = sorted(set(hybrid) & set(reranked) & set(dataset.qrels))
    counts = Counter()

    for query_id in query_ids:
        relevant_ids = _relevant_ids(dataset.qrels[query_id])
        if not relevant_ids:
            continue

        hybrid_ids = _record_doc_ids(hybrid[query_id])[:100]
        reranked_ids = _record_doc_ids(reranked[query_id])[:100]

        counts[
            _classify(
                hybrid_ids,
                reranked_ids,
                relevant_ids,
            )
        ] += 1

    total = sum(counts.values())

    print(f"{name}")
    print(f"queries analyzed: {total}")
    for label in (
        "retrieval failure",
        "ranking failure",
        "ranking fixed",
        "already top-10",
    ):
        count = counts[label]
        percentage = 100.0 * count / total if total else 0.0
        print(f"{label:<18} {count:>4}  ({percentage:>5.1f}%)")
    print()


def main() -> None:
    print("Retrieval-vs-ranking failure attribution")
    print()
    for name, config in DATASETS.items():
        analyze(name, config)


if __name__ == "__main__":
    main()
