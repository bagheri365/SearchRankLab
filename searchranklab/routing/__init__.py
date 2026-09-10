"""Pre-retrieval routing utilities."""

from .baselines import always_strategy, length_heuristic, random_strategy
from .diagnostics import FeatureDiagnostic, feature_diagnostics, strongest_effects
from .evaluation import RoutedSummary, evaluate_routing
from .features import FEATURE_NAMES, featurize_queries, query_features
from .model import PreRetrievalRouter

__all__ = [
    "FEATURE_NAMES",
    "FeatureDiagnostic",
    "PreRetrievalRouter",
    "RoutedSummary",
    "always_strategy",
    "evaluate_routing",
    "feature_diagnostics",
    "featurize_queries",
    "length_heuristic",
    "query_features",
    "random_strategy",
    "strongest_effects",
]
