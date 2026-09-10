"""Simple pre-retrieval routing baselines."""

from __future__ import annotations

import random

from .features import query_features


def always_strategy(queries: list[str], strategy: str) -> list[str]:
    """Route every query to the same strategy."""
    return [strategy] * len(queries)


def random_strategy(
    queries: list[str],
    *,
    strategies: tuple[str, ...] = ("bm25", "dense", "hybrid"),
    seed: int = 42,
) -> list[str]:
    """Uniform random router with deterministic seeding."""
    rng = random.Random(seed)
    return [rng.choice(strategies) for _ in queries]


def length_heuristic(
    queries: list[str],
    *,
    dense_threshold: int = 12,
    hybrid_threshold: int = 20,
) -> list[str]:
    """Tiny query-length heuristic.

    Short queries use BM25, medium queries use dense retrieval, and long
    queries use hybrid retrieval.
    """
    if dense_threshold <= 0:
        raise ValueError("dense_threshold must be positive")
    if hybrid_threshold <= dense_threshold:
        raise ValueError("hybrid_threshold must be greater than dense_threshold")

    predictions = []
    for query in queries:
        token_count = int(query_features(query)[0])
        if token_count >= hybrid_threshold:
            predictions.append("hybrid")
        elif token_count >= dense_threshold:
            predictions.append("dense")
        else:
            predictions.append("bm25")
    return predictions
