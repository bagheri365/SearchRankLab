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
