"""Simple learned pre-retrieval router."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import featurize_queries


class PreRetrievalRouter:
    """Multinomial logistic-regression router over query-only features."""

    def __init__(self) -> None:
        self.pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=42,
                    ),
                ),
            ]
        )

    def fit(self, queries: list[str], labels: list[str]) -> "PreRetrievalRouter":
        if len(queries) != len(labels):
            raise ValueError("queries and labels must have the same length")
        if not queries:
            raise ValueError("training data must not be empty")

        self.pipeline.fit(featurize_queries(queries), labels)
        return self

    def predict(self, queries: list[str]) -> list[str]:
        if not queries:
            return []
        return list(self.pipeline.predict(featurize_queries(queries)))
