import json
from pathlib import Path

from searchranklab.datasets.fiqa import load_fiqa


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_load_fiqa_reads_beir_test_split(tmp_path):
    dataset_dir = tmp_path / "fiqa"
    (dataset_dir / "qrels").mkdir(parents=True)

    _write_jsonl(
        dataset_dir / "corpus.jsonl",
        [
            {"_id": "d1", "title": "Finance", "text": "A document."},
            {"_id": "d2", "title": "", "text": "Another document."},
        ],
    )
    _write_jsonl(
        dataset_dir / "queries.jsonl",
        [{"_id": "q1", "text": "What is diversification?"}],
    )
    (dataset_dir / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td2\t1\n",
        encoding="utf-8",
    )

    dataset = load_fiqa(dataset_dir)

    assert len(dataset.corpus) == 2
    assert dataset.queries["q1"] == "What is diversification?"
    assert dataset.qrels["q1"]["d2"] == 1
