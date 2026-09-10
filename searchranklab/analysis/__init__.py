"""Analysis utilities."""

from .cost_oracle import (
    CostAwareDecision,
    CostAwareSummary,
    compute_cost_aware_oracle,
    summarize_cost_aware_oracle,
)
from .disagreement import (
    DisagreementSummary,
    QueryComparison,
    RunRecord,
    compare_runs,
    largest_bm25_wins,
    largest_dense_wins,
    load_run_records,
    summarize_disagreements,
)
from .oracle import (
    OracleDecision,
    OracleSummary,
    compute_oracle_decisions,
    summarize_oracle,
)

__all__ = [
    "CostAwareDecision",
    "CostAwareSummary",
    "DisagreementSummary",
    "OracleDecision",
    "OracleSummary",
    "QueryComparison",
    "RunRecord",
    "compare_runs",
    "compute_cost_aware_oracle",
    "compute_oracle_decisions",
    "largest_bm25_wins",
    "largest_dense_wins",
    "load_run_records",
    "summarize_cost_aware_oracle",
    "summarize_disagreements",
    "summarize_oracle",
]
