import json

from searchranklab.evaluation import evaluate_queries, write_query_evaluations_jsonl
from searchranklab.retrieval import SearchResult


def test_evaluate_queries_creates_per_query_record():
    records = evaluate_queries(
        queries={"q1": "alpha"},
        qrels={"q1": {"d1": 1, "d2": 1}},
        run={
            "q1": [
                SearchResult(doc_id="d1", score=2.0),
                SearchResult(doc_id="d3", score=1.0),
            ]
        },
    )

    assert len(records) == 1
    record = records[0]
    assert record.query_id == "q1"
    assert record.query == "alpha"
    assert record.recall_at_100 == 0.5
    assert record.mrr_at_10 == 1.0
    assert record.ndcg_at_10 > 0.0
    assert record.relevant_doc_ids == ["d1", "d2"]
    assert record.retrieved_doc_ids == ["d1", "d3"]


def test_write_query_evaluations_jsonl(tmp_path):
    records = evaluate_queries(
        queries={"q1": "alpha"},
        qrels={"q1": {"d1": 1}},
        run={"q1": [SearchResult(doc_id="d1", score=1.0)]},
    )

    path = write_query_evaluations_jsonl(records, tmp_path / "run.jsonl")

    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["query_id"] == "q1"
    assert rows[0]["recall_at_100"] == 1.0
    assert rows[0]["retrieved_doc_ids"] == ["d1"]
