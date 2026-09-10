# SearchRankLab

SearchRankLab studies **when BM25, dense, or hybrid retrieval should be used per query—and whether those choices generalize across domains**.

> **Research question:** How predictable are sparse-versus-dense retrieval gains before retrieval, and how much cost can be saved without materially degrading relevance?

Built on BEIR SciFact and FiQA, the project covers sparse retrieval, dense retrieval, hybrid fusion, pre-retrieval routing, frozen cross-domain transfer, cross-encoder reranking, and failure attribution.

**Full results:** [RESULTS.md](RESULTS.md) · **Reproduce headline tables:** `python experiments/24_results_summary.py`

## Headline results

**NDCG@10**

| Dataset | BM25 | Dense | Hybrid | Hybrid + reranker |
| --- | ---: | ---: | ---: | ---: |
| SciFact | 0.6519 | 0.4870 | 0.6197 | **0.6933** |
| FiQA | 0.2167 | 0.2317 | 0.2686 | **0.3512** |

- **Retrieval behavior shifts by domain:** BM25 is strongest on SciFact; dense/hybrid are stronger on FiQA.
- **Hybrid improves candidate recall:** Recall@100 reaches 0.9520 on SciFact and 0.5485 on FiQA.
- **Cross-encoder reranking gives the clearest quality gain:** NDCG@10 rises to 0.6933 on SciFact and 0.3512 on FiQA.
- **Adaptive routing has oracle headroom but does not generalize robustly:** learned routers fail to consistently beat strong fixed baselines, including under frozen SciFact -> FiQA transfer.

The negative routing result is part of the study: the experiments preserve failed hypotheses instead of tuning them away.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
```

Download the datasets:

```bash
python experiments/00_download_scifact.py
python experiments/15_download_fiqa.py
```

Raw datasets and generated per-query results are intentionally ignored by Git.

## Research design

The main routing objective is:

```text
utility = NDCG@10 - lambda * cost
```

with `lambda = 0.10` for the primary routing comparisons.

The implementation-independent retrieval cost proxy is:

```text
BM25   = 1
dense  = 1
hybrid = 2
```

Pre-retrieval routers are restricted to information available **before retrieval**. Retrieval scores, result overlap, returned candidates, and post-retrieval statistics are excluded from claims about retrieval-cost savings.

SciFact is the source domain for router development. FiQA is the target domain for frozen cross-domain evaluation with no target-label tuning.

## Project structure

```text
searchranklab/
  datasets/       BEIR-style dataset loading and validation
  evaluation/     retrieval metrics and per-query evaluation
  retrieval/      BM25, dense retrieval, and hybrid RRF
  routing/        pre-retrieval features, routers, diagnostics, robustness
  analysis/       run loading and comparison utilities

experiments/      reproducible experiment entry points
tests/            unit tests
RESULTS.md        full experimental narrative and results
```

## Experiment flow

```text
BM25 vs dense vs hybrid
        |
        v
oracle routing ceiling
        |
        v
pre-retrieval routing
        |
        v
multi-split robustness
        |
        v
frozen SciFact -> FiQA transfer
        |
        v
cross-encoder reranking
        |
        v
retrieval-vs-ranking failure attribution
```

Representative commands:

```bash
# SciFact retrieval
python experiments/01_bm25_scifact.py
python experiments/02_dense_scifact.py
python experiments/04_hybrid_scifact.py

# Routing and robustness
python experiments/06_cost_oracle_scifact.py
python experiments/12_router_robustness_scifact.py
python experiments/14_tuned_semantic_router_scifact.py

# FiQA retrieval and oracle
python experiments/16_bm25_fiqa.py
python experiments/17_dense_fiqa.py
python experiments/18_hybrid_rrf_fiqa.py
python experiments/19_oracle_fiqa.py

# Frozen cross-domain evaluation
python experiments/20_frozen_scifact_to_fiqa.py

# Cross-encoder reranking
python experiments/21_cross_encoder_rerank_scifact.py
python experiments/22_cross_encoder_rerank_fiqa.py

# Failure attribution and consolidated results
python experiments/23_failure_attribution.py
python experiments/24_results_summary.py
```

## Models

Dense retrieval:

```text
sentence-transformers/msmarco-MiniLM-L6-cos-v5
```

Cross-encoder reranking:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The dense baseline uses normalized embeddings with exact NumPy cosine search so core retrieval-quality comparisons are not confounded by approximate-nearest-neighbor indexing.

## Cost and latency interpretation

Local Mac timings are **relative implementation measurements**, not production latency claims.

- BM25 timing depends strongly on the current Python implementation and corpus size.
- Dense query latency is measured after corpus embeddings are built and is amortized over batched queries.
- Hybrid fusion timing measures fusion overhead only, not the cost of executing both retrievers.
- Cross-encoder reranking scores 100 query-document pairs per query and adds substantial inference cost.

The routing conclusions therefore rely primarily on the explicit strategy-cost proxy rather than machine-specific latency.

## Reproduce the headline results

After generating the required local per-query files:

```bash
python experiments/24_results_summary.py
```

For full results, oracle ceilings, routing robustness, frozen transfer, failure attribution, and limitations, see [RESULTS.md](RESULTS.md).

## Scope

The core study is intentionally small enough to run on a MacBook-class machine: two BEIR datasets, one compact dense retriever, one compact cross-encoder reranker, and lightweight routing models.

The emphasis is on **experimental design, robustness, negative results, cross-domain evaluation, failure analysis, and relevance-cost tradeoffs** rather than large-scale infrastructure.
