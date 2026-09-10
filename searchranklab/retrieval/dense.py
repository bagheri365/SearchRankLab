"""Exact dense retrieval for small and medium development corpora."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from .bm25 import SearchResult


DEFAULT_MODEL = "sentence-transformers/msmarco-MiniLM-L6-cos-v5"


class TextEncoder(Protocol):
    """Minimal encoder interface used by DenseRetriever."""

    def encode(
        self,
        sentences: Sequence[str] | str,
        *,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        ...


def _load_sentence_transformer(model_name: str, device: str) -> TextEncoder:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


class DenseRetriever:
    """Exact cosine-similarity retriever using normalized dense embeddings.

    The implementation intentionally uses exact NumPy search for the first
    SciFact baseline. ANN indexing is a later experiment, after retrieval
    quality is established.
    """

    def __init__(
        self,
        corpus: dict[str, dict[str, str]],
        *,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        encoder: TextEncoder | None = None,
        batch_size: int = 64,
    ) -> None:
        if not corpus:
            raise ValueError("corpus must not be empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.doc_ids = list(corpus)
        self.encoder = encoder or _load_sentence_transformer(model_name, device)
        self.batch_size = batch_size

        documents = [
            f"{corpus[doc_id].get('title', '')} {corpus[doc_id].get('text', '')}".strip()
            for doc_id in self.doc_ids
        ]
        self.document_embeddings = np.asarray(
            self.encoder.encode(
                documents,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            ),
            dtype=np.float32,
        )

        if self.document_embeddings.ndim != 2:
            raise ValueError("encoder must return a 2D document embedding matrix")
        if self.document_embeddings.shape[0] != len(self.doc_ids):
            raise ValueError("encoder returned the wrong number of document embeddings")

    def search(self, query: str, *, k: int = 10) -> list[SearchResult]:
        if k <= 0:
            raise ValueError("k must be positive")

        query_embedding = np.asarray(
            self.encoder.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        if query_embedding.ndim != 2 or query_embedding.shape[0] != 1:
            raise ValueError("encoder must return one query embedding")

        scores = self.document_embeddings @ query_embedding[0]
        top_k = min(k, len(self.doc_ids))
        top_indices = np.argsort(-scores, kind="stable")[:top_k]

        return [
            SearchResult(doc_id=self.doc_ids[index], score=float(scores[index]))
            for index in top_indices
        ]

    def batch_search(
        self,
        queries: dict[str, str],
        *,
        k: int = 100,
    ) -> dict[str, list[SearchResult]]:
        if k <= 0:
            raise ValueError("k must be positive")
        if not queries:
            return {}

        query_ids = list(queries)
        query_texts = [queries[query_id] for query_id in query_ids]
        query_embeddings = np.asarray(
            self.encoder.encode(
                query_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

        if query_embeddings.ndim != 2 or query_embeddings.shape[0] != len(query_ids):
            raise ValueError("encoder returned the wrong number of query embeddings")

        score_matrix = query_embeddings @ self.document_embeddings.T
        top_k = min(k, len(self.doc_ids))
        run: dict[str, list[SearchResult]] = {}

        for row_index, query_id in enumerate(query_ids):
            scores = score_matrix[row_index]
            top_indices = np.argsort(-scores, kind="stable")[:top_k]
            run[query_id] = [
                SearchResult(doc_id=self.doc_ids[index], score=float(scores[index]))
                for index in top_indices
            ]

        return run
