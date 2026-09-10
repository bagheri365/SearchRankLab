"""Inspect query-only feature separation for SciFact routing labels."""

from collections import Counter

from searchranklab.analysis import compute_cost_aware_oracle, load_run_records
from searchranklab.routing import feature_diagnostics, strongest_effects


BM25_PATH = "results/scifact/bm25_per_query.jsonl"
DENSE_PATH = "results/scifact/dense_per_query.jsonl"
HYBRID_PATH = "results/scifact/hybrid_rrf_per_query.jsonl"

COSTS = {"bm25": 1.0, "dense": 1.0, "hybrid": 2.0}
LAMBDA_VALUE = 0.10


def main() -> None:
    bm25 = load_run_records(BM25_PATH)
    dense = load_run_records(DENSE_PATH)
    hybrid = load_run_records(HYBRID_PATH)

    oracle = compute_cost_aware_oracle(
        bm25=bm25,
        dense=dense,
        hybrid=hybrid,
        costs=COSTS,
        lambda_value=LAMBDA_VALUE,
    )

    queries = [item.query for item in oracle]
    labels = [item.strategy for item in oracle]

    diagnostics = feature_diagnostics(queries, labels)

    print("Query-only feature diagnostics / SciFact")
    print(f"lambda: {LAMBDA_VALUE}")
    print(f"queries: {len(queries)}")
    print(f"labels: {dict(Counter(labels))}")
    print()
    print("Largest absolute standardized effects")
    print("label    feature                  count    mean    rest_mean    effect")

    for item in strongest_effects(diagnostics, n=12):
        print(
            f"{item.label:<8} "
            f"{item.feature:<24} "
            f"{item.count:>5} "
            f"{item.mean:>8.4f} "
            f"{item.rest_mean:>11.4f} "
            f"{item.standardized_effect:>9.4f}"
        )

    print()
    print("Interpretation guide")
    print("|effect| < 0.2: very weak separation")
    print("|effect| ~ 0.2-0.5: small separation")
    print("|effect| ~ 0.5-0.8: moderate separation")
    print("|effect| > 0.8: large separation")


if __name__ == "__main__":
    main()
