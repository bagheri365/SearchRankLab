"""Analysis utilities."""

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
    "DisagreementSummary",
    "OracleDecision",
    "OracleSummary",
    "QueryComparison",
    "RunRecord",
    "compare_runs",
    "compute_oracle_decisions",
    "largest_bm25_wins",
    "largest_dense_wins",
    "load_run_records",
    "summarize_disagreements",
    "summarize_oracle",
]
