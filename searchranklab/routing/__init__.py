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
from .semantic import DEFAULT_ROUTER_MODEL, load_query_encoder, semantic_query_features
from .semantic_tuned import (
    SemanticConfig,
    TunedSemanticUtilityRouter,
    select_semantic_config,
)
from .semantic_utility_model import SemanticUtilityRouter
from .utility_model import UtilityRegressionRouter

__all__ = [
    "DEFAULT_ROUTER_MODEL",
    "FEATURE_NAMES",
    "LEXICAL_FEATURE_NAMES",
    "FeatureDiagnostic",
    "LexicalStatistics",
    "PreRetrievalRouter",
    "RobustnessSummary",
    "RoutedSummary",
    "SeedResult",
    "SemanticConfig",
    "SemanticUtilityRouter",
    "TunedSemanticUtilityRouter",
    "UtilityRegressionRouter",
    "always_strategy",
    "build_lexical_statistics",
    "evaluate_routing",
    "feature_diagnostics",
    "featurize_lexical_queries",
    "featurize_queries",
    "length_heuristic",
    "lexical_query_features",
    "load_query_encoder",
    "query_features",
    "random_strategy",
    "select_semantic_config",
    "semantic_query_features",
    "strongest_effects",
    "summarize_seed_results",
]
