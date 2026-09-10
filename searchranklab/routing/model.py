"""Simple learned pre-retrieval router."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import featurize_queries
from .lexical import LexicalStatistics, featurize_lexical_queries


class PreRetrievalRouter:
    """Logistic-regression router over pre-retrieval features."""

    def __init__(
        self,
        *,
        class_weight: str | None = "balanced",
        lexical_stats: LexicalStatistics | None = None,
    ) -> None:
        self.class_weight = class_weight
        self.lexical_stats = lexical_stats
        self.pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight=class_weight,
                        max_iter=2000,
                        random_state=42,
                    ),
                ),
            ]
        )

    def _features(self, queries: list[str]) -> np.ndarray:
        surface = featurize_queries(queries)
        if self.lexical_stats is None:
            return surface

        lexical = featurize_lexical_queries(queries, self.lexical_stats)
        return np.hstack([surface, lexical])

    def fit(self, queries: list[str], labels: list[str]) -> "PreRetrievalRouter":
        if len(queries) != len(labels):
            raise ValueError("queries and labels must have the same length")
        if not queries:
            raise ValueError("training data must not be empty")

        self.pipeline.fit(self._features(queries), labels)
        return self

    def predict(self, queries: list[str]) -> list[str]:
        if not queries:
            return []
        return list(self.pipeline.predict(self._features(queries)))
