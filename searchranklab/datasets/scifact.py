"""Utilities for downloading and loading BEIR-style SciFact data."""

from __future__ import annotations

import csv
import json
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


SCIFACT_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/"
    "datasets/scifact.zip"
)


@dataclass(frozen=True)
class RetrievalDataset:
    """In-memory representation of a BEIR retrieval split."""

    corpus: dict[str, dict[str, str]]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]


def load_beir_dataset(data_dir: str | Path, split: str = "test") -> RetrievalDataset:
    """Load a BEIR-style dataset directory using only the Python standard library."""

    data_dir = Path(data_dir)
    corpus_path = data_dir / "corpus.jsonl"
    queries_path = data_dir / "queries.jsonl"
    qrels_path = data_dir / "qrels" / f"{split}.tsv"

    for path in (corpus_path, queries_path, qrels_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required dataset file not found: {path}")

    corpus: dict[str, dict[str, str]] = {}
    with corpus_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            corpus[str(row["_id"])] = {
                "title": row.get("title") or "",
                "text": row.get("text") or "",
            }

    queries: dict[str, str] = {}
    with queries_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            queries[str(row["_id"])] = row["text"]

    qrels: dict[str, dict[str, int]] = {}
    with qrels_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"query-id", "corpus-id", "score"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Expected qrels columns {sorted(required)}, got {reader.fieldnames}"
            )

        for row in reader:
            query_id = str(row["query-id"])
            corpus_id = str(row["corpus-id"])
            qrels.setdefault(query_id, {})[corpus_id] = int(row["score"])

    queries = {query_id: queries[query_id] for query_id in qrels if query_id in queries}

    return RetrievalDataset(corpus=corpus, queries=queries, qrels=qrels)


def load_scifact(data_root: str | Path = "data", split: str = "test") -> RetrievalDataset:
    """Load an already-downloaded SciFact dataset."""

    return load_beir_dataset(Path(data_root) / "scifact", split=split)


def download_scifact(
    data_root: str | Path = "data",
    *,
    url: str = SCIFACT_URL,
    force: bool = False,
) -> Path:
    """Download and extract the BEIR SciFact archive."""

    data_root = Path(data_root)
    target_dir = data_root / "scifact"

    required_files = (
        target_dir / "corpus.jsonl",
        target_dir / "queries.jsonl",
        target_dir / "qrels" / "test.tsv",
    )
    if not force and all(path.is_file() for path in required_files):
        return target_dir

    data_root.mkdir(parents=True, exist_ok=True)
    archive_path = data_root / "scifact.zip"

    urllib.request.urlretrieve(url, archive_path)

    try:
        if target_dir.exists() and force:
            shutil.rmtree(target_dir)

        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(data_root)
    finally:
        archive_path.unlink(missing_ok=True)

    missing = [path for path in required_files if not path.is_file()]
    if missing:
        raise RuntimeError(
            "SciFact archive extracted but required files are missing: "
            + ", ".join(str(path) for path in missing)
        )

    return target_dir
