"""Corpus-derived lexical specificity features for routing."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .features import TOKEN_RE


LEXICAL_FEATURE_NAMES = (
    "mean_idf",
    "max_idf",
    "min_idf",
    "idf_std",
    "mean_document_frequency_ratio",
    "min_document_frequency_ratio",
    "rare_token_ratio",
)


@dataclass(frozen=True)
class LexicalStatistics:
    document_count: int
    document_frequency: dict[str, int]


def build_lexical_statistics(
    corpus: dict[str, dict[str, str]],
) -> LexicalStatistics:
    """Build document-frequency statistics from the corpus only."""

    if not corpus:
        raise ValueError("corpus must not be empty")

    document_frequency: Counter[str] = Counter()

    for document in corpus.values():
        text = f"{document.get('title', '')} {document.get('text', '')}".lower()
        tokens = set(TOKEN_RE.findall(text))
        document_frequency.update(tokens)

    return LexicalStatistics(
        document_count=len(corpus),
        document_frequency=dict(document_frequency),
    )


def _idf(token: str, stats: LexicalStatistics) -> float:
    df = stats.document_frequency.get(token.lower(), 0)
    return math.log((stats.document_count + 1) / (df + 1)) + 1.0


def lexical_query_features(
    query: str,
    stats: LexicalStatistics,
) -> np.ndarray:
    """Return IDF/document-frequency features available before retrieval."""

    tokens = [token.lower() for token in TOKEN_RE.findall(query)]
    if not tokens:
        return np.zeros(len(LEXICAL_FEATURE_NAMES), dtype=np.float64)

    idf_values = np.asarray([_idf(token, stats) for token in tokens], dtype=np.float64)
    df_ratios = np.asarray(
        [
            stats.document_frequency.get(token, 0) / stats.document_count
            for token in tokens
        ],
        dtype=np.float64,
    )
    rare_ratio = float(np.mean(df_ratios <= 0.01))

    return np.asarray(
        [
            float(np.mean(idf_values)),
            float(np.max(idf_values)),
            float(np.min(idf_values)),
            float(np.std(idf_values)),
            float(np.mean(df_ratios)),
            float(np.min(df_ratios)),
            rare_ratio,
        ],
        dtype=np.float64,
    )


def featurize_lexical_queries(
    queries: list[str],
    stats: LexicalStatistics,
) -> np.ndarray:
    if not queries:
        return np.empty((0, len(LEXICAL_FEATURE_NAMES)), dtype=np.float64)
    return np.vstack([lexical_query_features(query, stats) for query in queries])
