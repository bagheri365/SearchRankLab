"""Pre-retrieval routing utilities."""

from .baselines import always_strategy, length_heuristic, random_strategy
from .diagnostics import FeatureDiagnostic, feature_diagnostics, strongest_effects
from .evaluation import RoutedSummary, evaluate_routing
from .features import FEATURE_NAMES, featurize_queries, query_features
from .lexical import (
    LEXICAL_FEATURE_NAMES,
    LexicalStatistics,
    build_lexical_statistics,
    featurize_lexical_queries,
    lexical_query_features,
)
from .model import PreRetrievalRouter
from .robustness import RobustnessSummary, SeedResult, summarize_seed_results
from .utility_model import UtilityRegressionRouter

__all__ = [
    "FEATURE_NAMES",
    "LEXICAL_FEATURE_NAMES",
    "FeatureDiagnostic",
    "LexicalStatistics",
    "PreRetrievalRouter",
    "RobustnessSummary",
    "RoutedSummary",
    "SeedResult",
    "UtilityRegressionRouter",
    "always_strategy",
    "build_lexical_statistics",
    "evaluate_routing",
    "feature_diagnostics",
    "featurize_lexical_queries",
    "featurize_queries",
    "length_heuristic",
    "lexical_query_features",
    "query_features",
    "random_strategy",
    "strongest_effects",
    "summarize_seed_results",
]
