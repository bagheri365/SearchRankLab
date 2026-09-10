"""BM25 lexical retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi


TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase word-tokenization used consistently for corpus and queries."""

    return TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class SearchResult:
    doc_id: str
    score: float


class BM25Retriever:
    """Small in-memory BM25 retriever suitable for BEIR-sized development sets."""

    def __init__(self, corpus: dict[str, dict[str, str]], *, k1: float = 1.5, b: float = 0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")

        self.doc_ids = list(corpus)
        tokenized_corpus = []
        for doc_id in self.doc_ids:
            doc = corpus[doc_id]
            text = f"{doc.get('title', '')} {doc.get('text', '')}".strip()
            tokenized_corpus.append(tokenize(text))

        self._index = BM25Okapi(tokenized_corpus, k1=k1, b=b)

    def search(self, query: str, *, k: int = 10) -> list[SearchResult]:
        if k <= 0:
            raise ValueError("k must be positive")

        scores = self._index.get_scores(tokenize(query))
        ranked = sorted(
            zip(self.doc_ids, scores, strict=True),
            key=lambda pair: (-float(pair[1]), pair[0]),
        )
        return [
            SearchResult(doc_id=doc_id, score=float(score))
            for doc_id, score in ranked[:k]
        ]

    def batch_search(self, queries: dict[str, str], *, k: int = 100) -> dict[str, list[SearchResult]]:
        return {query_id: self.search(query, k=k) for query_id, query in queries.items()}
