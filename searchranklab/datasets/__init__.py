"""Dataset loading utilities."""

from .scifact import (
    RetrievalDataset,
    download_scifact,
    load_beir_dataset,
    load_scifact,
)

__all__ = [
    "RetrievalDataset",
    "download_scifact",
    "load_beir_dataset",
    "load_scifact",
]
