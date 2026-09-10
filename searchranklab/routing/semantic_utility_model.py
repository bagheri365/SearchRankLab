"""Utility regression using semantic query embeddings."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .semantic import QueryEncoder, semantic_query_features


class SemanticUtilityRouter:
    """Predict per-strategy utility from query embeddings only."""

    def __init__(
        self,
        *,
        encoder: QueryEncoder,
        strategies: tuple[str, ...] = ("bm25", "dense", "hybrid"),
        alpha: float = 10.0,
    ) -> None:
        if not strategies:
            raise ValueError("strategies must not be empty")
        if alpha < 0:
            raise ValueError("alpha must be nonnegative")

        self.encoder = encoder
        self.strategies = strategies
        self.models = {
            strategy: Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("regressor", Ridge(alpha=alpha)),
                ]
            )
            for strategy in strategies
        }

    def fit(self, queries: list[str], utilities: dict[str, list[float]]):
        if not queries:
            raise ValueError("training data must not be empty")
        if set(utilities) != set(self.strategies):
            raise ValueError("utilities must define every strategy exactly once")
        if any(len(values) != len(queries) for values in utilities.values()):
            raise ValueError("utility targets must match the number of queries")

        features = semantic_query_features(queries, self.encoder)
        for strategy in self.strategies:
            self.models[strategy].fit(features, utilities[strategy])
        return self

    def predict(self, queries: list[str]) -> list[str]:
        if not queries:
            return []

        features = semantic_query_features(queries, self.encoder)
        matrix = np.column_stack(
            [self.models[strategy].predict(features) for strategy in self.strategies]
        )
        best_indices = np.argmax(matrix, axis=1)
        return [self.strategies[index] for index in best_indices]
