"""Diagnostics for query-only routing features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import FEATURE_NAMES, featurize_queries


@dataclass(frozen=True)
class FeatureDiagnostic:
    label: str
    feature: str
    count: int
    mean: float
    std: float
    rest_mean: float
    standardized_effect: float


def _pooled_std(group: np.ndarray, rest: np.ndarray) -> float:
    if len(group) < 2 or len(rest) < 2:
        return 0.0

    group_var = float(np.var(group, ddof=1))
    rest_var = float(np.var(rest, ddof=1))
    numerator = (len(group) - 1) * group_var + (len(rest) - 1) * rest_var
    denominator = len(group) + len(rest) - 2

    if denominator <= 0:
        return 0.0

    pooled = float(np.sqrt(numerator / denominator))
    return pooled


def feature_diagnostics(
    queries: list[str],
    labels: list[str],
) -> list[FeatureDiagnostic]:
    """Compare each class against the remaining queries feature by feature."""

    if len(queries) != len(labels):
        raise ValueError("queries and labels must have the same length")
    if not queries:
        raise ValueError("diagnostic data must not be empty")

    matrix = featurize_queries(queries)
    label_array = np.asarray(labels, dtype=object)
    diagnostics: list[FeatureDiagnostic] = []

    for label in sorted(set(labels)):
        mask = label_array == label
        rest_mask = ~mask

        if not np.any(rest_mask):
            raise ValueError("diagnostics require at least two distinct labels")

        for feature_index, feature_name in enumerate(FEATURE_NAMES):
            group = matrix[mask, feature_index]
            rest = matrix[rest_mask, feature_index]

            group_mean = float(np.mean(group))
            rest_mean = float(np.mean(rest))
            pooled_std = _pooled_std(group, rest)
            effect = 0.0 if pooled_std == 0.0 else (group_mean - rest_mean) / pooled_std

            diagnostics.append(
                FeatureDiagnostic(
                    label=label,
                    feature=feature_name,
                    count=len(group),
                    mean=group_mean,
                    std=float(np.std(group, ddof=0)),
                    rest_mean=rest_mean,
                    standardized_effect=effect,
                )
            )

    return diagnostics


def strongest_effects(
    diagnostics: list[FeatureDiagnostic],
    *,
    n: int = 10,
) -> list[FeatureDiagnostic]:
    """Return the largest absolute standardized effects."""

    if n <= 0:
        raise ValueError("n must be positive")

    return sorted(
        diagnostics,
        key=lambda item: (
            -abs(item.standardized_effect),
            item.label,
            item.feature,
        ),
    )[:n]
