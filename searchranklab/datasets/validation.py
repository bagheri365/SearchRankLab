"""Dataset summary and integrity checks."""

from __future__ import annotations

from dataclasses import dataclass

from .scifact import RetrievalDataset


@dataclass(frozen=True)
class DatasetSummary:
    documents: int
    queries: int
    qrels: int
    relevant_documents: int


def summarize_dataset(dataset: RetrievalDataset) -> DatasetSummary:
    """Return basic counts for a retrieval dataset."""

    relevant_documents = {
        doc_id
        for judgments in dataset.qrels.values()
        for doc_id, score in judgments.items()
        if score > 0
    }
    return DatasetSummary(
        documents=len(dataset.corpus),
        queries=len(dataset.queries),
        qrels=sum(len(judgments) for judgments in dataset.qrels.values()),
        relevant_documents=len(relevant_documents),
    )


def validate_dataset(dataset: RetrievalDataset) -> None:
    """Raise ValueError when qrels reference missing queries or documents."""

    missing_queries = sorted(set(dataset.qrels) - set(dataset.queries))
    missing_documents = sorted(
        {
            doc_id
            for judgments in dataset.qrels.values()
            for doc_id in judgments
            if doc_id not in dataset.corpus
        }
    )

    problems = []
    if missing_queries:
        problems.append(f"missing queries referenced by qrels: {missing_queries[:5]}")
    if missing_documents:
        problems.append(f"missing documents referenced by qrels: {missing_documents[:5]}")

    if problems:
        raise ValueError("; ".join(problems))
