"""Nested-CV tuning for semantic utility routing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class SemanticConfig:
    pca_components: int
    alpha: float


def _fit_models(
    features: np.ndarray,
    utilities: dict[str, np.ndarray],
    *,
    strategies: tuple[str, ...],
    config: SemanticConfig,
) -> dict[str, Pipeline]:
    models: dict[str, Pipeline] = {}
    for strategy in strategies:
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "pca",
                    PCA(
                        n_components=config.pca_components,
                        random_state=42,
                    ),
                ),
                ("ridge", Ridge(alpha=config.alpha)),
            ]
        )
        model.fit(features, utilities[strategy])
        models[strategy] = model
    return models


def select_semantic_config(
    features: np.ndarray,
    utilities: dict[str, np.ndarray],
    *,
    strategies: tuple[str, ...] = ("bm25", "dense", "hybrid"),
    pca_components: tuple[int, ...] = (8, 16, 32, 64),
    alphas: tuple[float, ...] = (1.0, 10.0, 100.0),
    cv_splits: int = 5,
    random_state: int = 42,
) -> SemanticConfig:
    """Select PCA/Ridge hyperparameters by mean held-out routing utility."""

    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("features must be a non-empty 2D matrix")
    if set(utilities) != set(strategies):
        raise ValueError("utilities must define every strategy exactly once")
    if any(len(values) != len(features) for values in utilities.values()):
        raise ValueError("utility targets must match feature rows")

    max_components = min(features.shape[0] - 1, features.shape[1])
    candidates = [
        SemanticConfig(components, alpha)
        for components in pca_components
        if 1 <= components <= max_components
        for alpha in alphas
    ]
    if not candidates:
        raise ValueError("no valid PCA configurations for the training data")

    splitter = KFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state,
    )
    utility_arrays = {
        strategy: np.asarray(values, dtype=np.float64)
        for strategy, values in utilities.items()
    }

    best_config = None
    best_score = -np.inf

    for config in candidates:
        fold_scores: list[float] = []
        for train_index, valid_index in splitter.split(features):
            models = _fit_models(
                features[train_index],
                {
                    strategy: values[train_index]
                    for strategy, values in utility_arrays.items()
                },
                strategies=strategies,
                config=config,
            )
            predicted = np.column_stack(
                [models[strategy].predict(features[valid_index]) for strategy in strategies]
            )
            chosen_indices = np.argmax(predicted, axis=1)
            realized = np.column_stack(
                [utility_arrays[strategy][valid_index] for strategy in strategies]
            )
            fold_scores.append(
                float(np.mean(realized[np.arange(len(valid_index)), chosen_indices]))
            )

        score = float(np.mean(fold_scores))
        if score > best_score:
            best_score = score
            best_config = config

    assert best_config is not None
    return best_config


class TunedSemanticUtilityRouter:
    """Semantic utility router with PCA/Ridge selected on training data only."""

    def __init__(
        self,
        *,
        encoder,
        strategies: tuple[str, ...] = ("bm25", "dense", "hybrid"),
        pca_components: tuple[int, ...] = (8, 16, 32, 64),
        alphas: tuple[float, ...] = (1.0, 10.0, 100.0),
        cv_splits: int = 5,
        random_state: int = 42,
    ) -> None:
        self.encoder = encoder
        self.strategies = strategies
        self.pca_components = pca_components
        self.alphas = alphas
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.config: SemanticConfig | None = None
        self.models: dict[str, Pipeline] = {}

    def _encode(self, queries: list[str]) -> np.ndarray:
        from .semantic import semantic_query_features
        return semantic_query_features(queries, self.encoder)

    def fit(self, queries: list[str], utilities: dict[str, list[float]]):
        if not queries:
            raise ValueError("training data must not be empty")

        features = self._encode(queries)
        utility_arrays = {
            strategy: np.asarray(values, dtype=np.float64)
            for strategy, values in utilities.items()
        }
        self.config = select_semantic_config(
            features,
            utility_arrays,
            strategies=self.strategies,
            pca_components=self.pca_components,
            alphas=self.alphas,
            cv_splits=self.cv_splits,
            random_state=self.random_state,
        )
        self.models = _fit_models(
            features,
            utility_arrays,
            strategies=self.strategies,
            config=self.config,
        )
        return self

    def predict(self, queries: list[str]) -> list[str]:
        if not queries:
            return []
        if self.config is None:
            raise ValueError("router must be fit before prediction")

        features = self._encode(queries)
        predicted = np.column_stack(
            [self.models[strategy].predict(features) for strategy in self.strategies]
        )
        best_indices = np.argmax(predicted, axis=1)
        return [self.strategies[index] for index in best_indices]
