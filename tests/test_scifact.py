import json
from pathlib import Path

import pytest

from searchranklab.datasets.scifact import load_beir_dataset, load_scifact


def _write_fixture_dataset(root: Path) -> Path:
    dataset_dir = root / "scifact"
    qrels_dir = dataset_dir / "qrels"
    qrels_dir.mkdir(parents=True)

    corpus = [
        {"_id": "d1", "title": "Document One", "text": "alpha beta"},
        {"_id": "d2", "title": "", "text": "gamma delta"},
    ]
    queries = [
        {"_id": "q1", "text": "alpha"},
        {"_id": "q_unused", "text": "unused query"},
    ]

    (dataset_dir / "corpus.jsonl").write_text(
        "\n".join(json.dumps(row) for row in corpus) + "\n",
        encoding="utf-8",
    )
    (dataset_dir / "queries.jsonl").write_text(
        "\n".join(json.dumps(row) for row in queries) + "\n",
        encoding="utf-8",
    )
    (qrels_dir / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t1\n",
        encoding="utf-8",
    )
    return dataset_dir


def test_load_beir_dataset(tmp_path):
    dataset_dir = _write_fixture_dataset(tmp_path)

    dataset = load_beir_dataset(dataset_dir)

    assert dataset.corpus["d1"] == {
        "title": "Document One",
        "text": "alpha beta",
    }
    assert dataset.queries == {"q1": "alpha"}
    assert dataset.qrels == {"q1": {"d1": 1}}


def test_load_scifact_uses_data_root(tmp_path):
    _write_fixture_dataset(tmp_path)

    dataset = load_scifact(tmp_path)

    assert set(dataset.corpus) == {"d1", "d2"}
    assert set(dataset.queries) == {"q1"}


def test_missing_required_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_beir_dataset(tmp_path / "missing")
