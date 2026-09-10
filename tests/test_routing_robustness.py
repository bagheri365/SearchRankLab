import pytest

from searchranklab.routing import SeedResult, summarize_seed_results


def test_summarize_seed_results_aggregates_by_router():
    results = [
        SeedResult(1, "bm25", 0.6, 0.8, 1.0, 0.5),
        SeedResult(2, "bm25", 0.8, 1.0, 1.0, 0.7),
        SeedResult(1, "router", 0.7, 0.9, 1.1, 0.59),
        SeedResult(2, "router", 0.9, 0.9, 1.1, 0.79),
    ]

    summaries = {
        summary.router: summary
        for summary in summarize_seed_results(results)
    }

    assert summaries["bm25"].runs == 2
    assert summaries["bm25"].mean_ndcg_at_10 == pytest.approx(0.7)
    assert summaries["bm25"].std_ndcg_at_10 == pytest.approx(0.1)
    assert summaries["router"].mean_utility == pytest.approx(0.69)


def test_single_run_has_zero_population_std():
    summary = summarize_seed_results(
        [SeedResult(42, "router", 0.7, 0.8, 1.0, 0.6)]
    )[0]

    assert summary.std_ndcg_at_10 == 0.0
    assert summary.std_utility == 0.0


def test_summarize_seed_results_rejects_empty_input():
    with pytest.raises(ValueError, match="must not be empty"):
        summarize_seed_results([])
