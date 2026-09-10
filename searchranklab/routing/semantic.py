"""Semantic query features for pre-retrieval routing."""

from __future__ import annotations

from typing import Protocol

import numpy as np


DEFAULT_ROUTER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class QueryEncoder(Protocol):
    def encode(
        self,
        sentences,
        *,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        ...


def load_query_encoder(
    model_name: str = DEFAULT_ROUTER_MODEL,
    *,
    device: str = "cpu",
) -> QueryEncoder:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


def semantic_query_features(
    queries: list[str],
    encoder: QueryEncoder,
) -> np.ndarray:
    """Encode queries before retrieval for use as router inputs."""

    if not queries:
        return np.empty((0, 0), dtype=np.float32)

    embeddings = np.asarray(
        encoder.encode(
            queries,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )

    if embeddings.ndim != 2 or embeddings.shape[0] != len(queries):
        raise ValueError("encoder returned an invalid query embedding matrix")

    return embeddings
