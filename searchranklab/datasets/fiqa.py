"""FiQA dataset convenience wrappers."""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

from .scifact import RetrievalDataset, load_beir_dataset


FIQA_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip"


def load_fiqa(path: str | Path = "data/fiqa") -> RetrievalDataset:
    """Load the FiQA BEIR test split."""

    return load_beir_dataset(path, split="test")


def download_fiqa(path: str | Path = "data/fiqa") -> Path:
    """Download and extract FiQA if it is not already present."""

    path = Path(path)
    corpus_path = path / "corpus.jsonl"
    queries_path = path / "queries.jsonl"
    qrels_path = path / "qrels" / "test.tsv"

    if corpus_path.exists() and queries_path.exists() and qrels_path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    archive_path = path.parent / "fiqa.zip"

    try:
        urllib.request.urlretrieve(FIQA_URL, archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(path.parent)
    finally:
        if archive_path.exists():
            archive_path.unlink()

    if not corpus_path.exists() or not queries_path.exists() or not qrels_path.exists():
        raise FileNotFoundError(
            "FiQA download did not produce the expected BEIR files"
        )

    return path
