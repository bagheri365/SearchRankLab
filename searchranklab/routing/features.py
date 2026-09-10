"""Cheap query-only features for pre-retrieval routing."""

from __future__ import annotations

import re

import numpy as np


TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)

FEATURE_NAMES = (
    "token_count",
    "character_count",
    "average_token_length",
    "unique_token_ratio",
    "digit_token_ratio",
    "uppercase_token_ratio",
    "question_mark",
)


def query_features(query: str) -> np.ndarray:
    """Return features available before any retriever is executed."""

    tokens = TOKEN_RE.findall(query)
    token_count = len(tokens)
    lengths = [len(token) for token in tokens]

    if token_count:
        unique_ratio = len({token.lower() for token in tokens}) / token_count
        digit_ratio = sum(token.isdigit() for token in tokens) / token_count
        uppercase_ratio = sum(
            token.isupper() and any(char.isalpha() for char in token)
            for token in tokens
        ) / token_count
        average_length = sum(lengths) / token_count
    else:
        unique_ratio = 0.0
        digit_ratio = 0.0
        uppercase_ratio = 0.0
        average_length = 0.0

    return np.asarray(
        [
            float(token_count),
            float(len(query)),
            float(average_length),
            float(unique_ratio),
            float(digit_ratio),
            float(uppercase_ratio),
            float("?" in query),
        ],
        dtype=np.float64,
    )


def featurize_queries(queries: list[str]) -> np.ndarray:
    """Convert query strings to a 2D feature matrix."""

    if not queries:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
    return np.vstack([query_features(query) for query in queries])
