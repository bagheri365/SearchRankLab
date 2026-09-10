"""Utility-regression router for cost-aware pre-retrieval selection."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import featurize_queries
from .lexical import LexicalStatistics, featurize_lexical_queries


class UtilityRegressionRouter:
    """Predict per-strategy utility and choose the highest predicted action."""

    def __init__(
        self,
        *,
        strategies: tuple[str, ...] = ("bm25", "dense", "hybrid"),
        lexical_stats: LexicalStatistics | None = None,
        alpha: float = 1.0,
    ) -> None:
        if not strategies:
            raise ValueError("strategies must not be empty")
        if alpha < 0:
            raise ValueError("alpha must be nonnegative")

        self.strategies = strategies
        self.lexical_stats = lexical_stats
        self.models = {
            strategy: Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("regressor", Ridge(alpha=alpha)),
                ]
            )
            for strategy in strategies
        }

    def _features(self, queries: list[str]) -> np.ndarray:
        surface = featurize_queries(queries)
        if self.lexical_stats is None:
            return surface

        lexical = featurize_lexical_queries(queries, self.lexical_stats)
        return np.hstack([surface, lexical])

    def fit(
        self,
        queries: list[str],
        utilities: dict[str, list[float]],
    ) -> "UtilityRegressionRouter":
        if not queries:
            raise ValueError("training data must not be empty")
        if set(utilities) != set(self.strategies):
            raise ValueError("utilities must define every strategy exactly once")
        if any(len(values) != len(queries) for values in utilities.values()):
            raise ValueError("utility targets must match the number of queries")

        features = self._features(queries)
        for strategy in self.strategies:
            self.models[strategy].fit(features, utilities[strategy])

        return self

    def predict_utilities(self, queries: list[str]) -> dict[str, np.ndarray]:
        if not queries:
            return {
                strategy: np.empty(0, dtype=np.float64)
                for strategy in self.strategies
            }

        features = self._features(queries)
        return {
            strategy: np.asarray(self.models[strategy].predict(features))
            for strategy in self.strategies
        }

    def predict(self, queries: list[str]) -> list[str]:
        predictions = self.predict_utilities(queries)
        if not queries:
            return []

        matrix = np.column_stack([predictions[strategy] for strategy in self.strategies])
        best_indices = np.argmax(matrix, axis=1)
        return [self.strategies[index] for index in best_indices]
