import pytest

from searchranklab.datasets.scifact import RetrievalDataset
from searchranklab.datasets.validation import summarize_dataset, validate_dataset


def test_summarize_dataset_counts_unique_relevant_documents():
    dataset = RetrievalDataset(
        corpus={
            "d1": {"title": "", "text": "one"},
            "d2": {"title": "", "text": "two"},
        },
        queries={"q1": "one", "q2": "two"},
        qrels={
            "q1": {"d1": 1, "d2": 0},
            "q2": {"d1": 1},
        },
    )

    summary = summarize_dataset(dataset)

    assert summary.documents == 2
    assert summary.queries == 2
    assert summary.qrels == 3
    assert summary.relevant_documents == 1


def test_validate_dataset_accepts_consistent_dataset():
    dataset = RetrievalDataset(
        corpus={"d1": {"title": "", "text": "one"}},
        queries={"q1": "one"},
        qrels={"q1": {"d1": 1}},
    )

    validate_dataset(dataset)


def test_validate_dataset_rejects_missing_documents():
    dataset = RetrievalDataset(
        corpus={},
        queries={"q1": "one"},
        qrels={"q1": {"missing": 1}},
    )

    with pytest.raises(ValueError, match="missing documents"):
        validate_dataset(dataset)
