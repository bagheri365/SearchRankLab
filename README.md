# SearchRankLab

SearchRankLab is a research project on query-adaptive information retrieval.

The central research question is:

> How predictable are sparse-versus-dense retrieval gains before retrieval, and how much cost can be saved without materially degrading relevance?

The project will compare:

- BM25 sparse retrieval
- dense retrieval
- hybrid retrieval
- query-adaptive routing
- cross-encoder reranking

Initial datasets:

- SciFact
- FiQA

## SciFact setup

Download the BEIR-formatted SciFact dataset:

```bash
python experiments/00_download_scifact.py
```

The raw dataset is stored under `data/scifact/` and is intentionally ignored by Git.

## BM25 baseline

After downloading SciFact, run:

```bash
python experiments/01_bm25_scifact.py
```

The baseline reports Recall@100, MRR@10, NDCG@10, and local timing measurements.

The BM25 experiment also writes per-query rankings and metrics to:

```text
results/scifact/bm25_per_query.jsonl
```

These records are used later for sparse-vs-dense disagreement analysis.

## Dense retrieval baseline

Run the CPU-friendly exact dense baseline on SciFact:

```bash
python experiments/02_dense_scifact.py
```

The initial dense baseline uses `sentence-transformers/msmarco-MiniLM-L6-cos-v5`
with normalized embeddings and exact NumPy cosine search. Exact search is
intentional at this stage so retrieval quality is not confounded by ANN
approximation.

Per-query results are written to:

```text
results/scifact/dense_per_query.jsonl
```

## Sparse vs dense disagreement analysis

After generating both per-query result files, run:

```bash
python experiments/03_compare_bm25_dense.py
```

This reports query-level BM25 wins, dense wins, ties, Recall@100 recoveries,
and examples with the largest NDCG@10 differences.

## Hybrid retrieval baseline

After generating the sparse and dense baselines, run:

```bash
python experiments/04_hybrid_scifact.py
```

This fuses BM25 and dense top-100 rankings with Reciprocal Rank Fusion (RRF)
using `k=60`, then reports Recall@100, MRR@10, and NDCG@10.

Per-query results are written to:

```text
results/scifact/hybrid_rrf_per_query.jsonl
```

## Oracle routing ceiling

After generating BM25, dense, and hybrid per-query outputs, run:

```bash
python experiments/05_oracle_scifact.py
```

The oracle selects the highest-NDCG@10 strategy for each query and provides an
upper bound on the potential value of adaptive routing before any router is
trained.

## Cost-aware oracle frontier

After the relevance-only oracle, run:

```bash
python experiments/06_cost_oracle_scifact.py
```

This evaluates the oracle objective

```text
utility = NDCG@10 - lambda * cost
```

across several lambda values. The first cost proxy is intentionally simple:
BM25=1, dense=1, hybrid=2. It is an implementation-independent relative compute
proxy, not a claim about production latency.

## First pre-retrieval router

Run the first learned router with:

```bash
python experiments/07_pre_retrieval_router_scifact.py
```

The router uses only cheap query features available before retrieval: length,
token statistics, digit/uppercase ratios, and question-mark presence. It does
not use retrieval scores, overlap, result sets, or relevance judgments as
features.

For this first in-domain prototype, cost-aware oracle decisions at `lambda=0.10`
are used as labels and SciFact queries are split deterministically into a
stratified 70/30 train/test split. This is a development experiment, not the
final cross-domain evaluation.
