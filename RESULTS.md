# SearchRankLab Results

SearchRankLab studies a practical search-system question:

> How predictable are sparse-versus-dense retrieval gains before retrieval, and how much cost can be saved without materially degrading relevance?

The experiments compare BM25, dense retrieval, reciprocal-rank-fusion hybrid retrieval, query-adaptive routing, and cross-encoder reranking on two BEIR datasets: SciFact and FiQA.

## Experimental protocol

The core relevance metrics are NDCG@10, MRR@10, and Recall@100.

Routing experiments use the cost-aware objective

`utility = NDCG@10 - lambda * cost`

with `lambda = 0.10` for the main routing comparisons.

The implementation-independent retrieval cost proxy is:

- BM25: 1
- dense: 1
- hybrid: 2

Pre-retrieval routers are restricted to information available before any retrieval call. Retrieval scores, result overlap, candidate sets, and post-retrieval statistics are not used to justify retrieval-cost savings.

SciFact is the source domain for routing development. FiQA is used as the target domain for frozen cross-domain evaluation.

## Retrieval and reranking results

### SciFact

| System | NDCG@10 | MRR@10 | Recall@100 |
| --- | ---: | ---: | ---: |
| BM25 | 0.6519 | 0.6186 | 0.8731 |
| Dense | 0.4870 | 0.4488 | 0.8342 |
| Hybrid RRF | 0.6197 | 0.5870 | 0.9520 |
| Hybrid + cross-encoder | **0.6933** | **0.6620** | **0.9520** |

BM25 is the strongest single retriever on SciFact. Hybrid RRF substantially improves candidate recall but degrades top-10 ranking quality. Cross-encoder reranking recovers that ranking quality and surpasses BM25 while preserving the hybrid candidate recall.

Relative to hybrid RRF, cross-encoder reranking improves NDCG@10 by **+0.0736** and MRR@10 by **+0.0750**.

### FiQA

| System | NDCG@10 | MRR@10 | Recall@100 |
| --- | ---: | ---: | ---: |
| BM25 | 0.2167 | 0.2703 | 0.4737 |
| Dense | 0.2317 | 0.2784 | 0.4956 |
| Hybrid RRF | 0.2686 | 0.3295 | 0.5485 |
| Hybrid + cross-encoder | **0.3512** | **0.4226** | **0.5485** |

FiQA reverses the SciFact ordering: dense retrieval slightly outperforms BM25, and hybrid RRF improves all three relevance metrics.

Cross-encoder reranking again gives a large ranking gain while preserving Recall@100. Relative to hybrid RRF, reranking improves NDCG@10 by **+0.0827** and MRR@10 by **+0.0931**.

## Oracle routing headroom

The cost-aware oracle at `lambda = 0.10` selects the highest-utility strategy independently for each query.

| Dataset | Oracle NDCG@10 | Oracle Recall@100 | Mean cost | Utility |
| --- | ---: | ---: | ---: | ---: |
| SciFact | 0.7574 | 0.9153 | 1.1167 | 0.6458 |
| FiQA | 0.3399 | 0.5336 | 1.1235 | 0.2276 |

Oracle selections are heterogeneous in both domains:

- SciFact: 225 BM25, 40 dense, 35 hybrid
- FiQA: 420 BM25, 148 dense, 80 hybrid

This shows that meaningful per-query routing headroom exists even though the best global retriever differs by domain.

## Learned pre-retrieval routing

Several pre-retrieval routing approaches were evaluated on SciFact:

- surface query features
- lexical-specificity features derived from corpus statistics
- class-balanced and unweighted logistic-regression routers
- utility-regression routers
- raw semantic query embeddings
- PCA-reduced semantic query embeddings with nested cross-validation

A single SciFact split initially suggested that lexical utility regression could outperform always-BM25 at the same mean cost. However, a 10-seed robustness study did not reproduce the gain:

| Router | NDCG@10 mean ± std | Recall@100 mean ± std | Mean cost | Utility mean ± std |
| --- | ---: | ---: | ---: | ---: |
| Always BM25 | **0.6791 ± 0.0318** | 0.8707 ± 0.0239 | **1.0000** | **0.5791 ± 0.0318** |
| Utility lexical | 0.6770 ± 0.0378 | **0.8782 ± 0.0237** | 1.0322 | 0.5738 ± 0.0382 |
| Tuned semantic utility | 0.6768 ± 0.0327 | 0.8752 ± 0.0276 | 1.0211 | 0.5747 ± 0.0333 |

The lexical utility router beats always-BM25 on only 3 of 10 splits. The tuned semantic router also wins on only 3 of 10 splits.

The core in-domain conclusion is therefore negative: substantial oracle headroom exists, but the tested cheap pre-retrieval features do not exploit it robustly.

## Frozen SciFact -> FiQA routing

The routing models are then trained on all SciFact queries and evaluated on all FiQA queries with:

- no FiQA label tuning
- no FiQA threshold tuning
- fixed `lambda = 0.10`
- frozen SciFact lexical statistics for the lexical router

| Strategy | NDCG@10 | Recall@100 | Mean cost | Utility |
| --- | ---: | ---: | ---: | ---: |
| Always BM25 | 0.2167 | 0.4737 | 1.0000 | 0.1167 |
| Always dense | **0.2317** | **0.4956** | **1.0000** | **0.1317** |
| Always hybrid | 0.2686 | 0.5485 | 2.0000 | 0.0686 |
| Frozen surface router | 0.2169 | 0.4751 | 1.0000 | 0.1169 |
| Frozen lexical router | 0.2191 | 0.4783 | 1.0509 | 0.1140 |
| Target oracle | 0.3399 | 0.5336 | 1.1235 | 0.2276 |

Neither frozen router beats the strongest cheap fixed FiQA baseline, always-dense.

The frozen surface router sends 639 of 648 FiQA queries to BM25, reflecting the source-domain preference learned from SciFact rather than transferable sparse-versus-dense query signals.

This is evidence that source-domain routing preferences can dominate apparently query-specific routing behavior under domain shift.

## Retrieval-versus-ranking failure attribution

Failures are classified using the hybrid top-100 candidate set and the cross-encoder reranking output.

A query is labeled:

- **retrieval failure** if no relevant document appears in the hybrid top 100
- **ranking failure** if a relevant document is retrieved but the reranker still fails to place one in the top 10
- **ranking fixed** if hybrid retrieves a relevant document below rank 10 and the reranker moves a relevant document into the top 10
- **already top-10** if hybrid already places a relevant document in the top 10

| Dataset | Retrieval failure | Ranking failure | Ranking fixed | Already top-10 |
| --- | ---: | ---: | ---: | ---: |
| SciFact | 4.3% | 12.0% | 12.0% | 71.7% |
| FiQA | 25.2% | 13.4% | 12.8% | 48.6% |

Among the remaining SciFact errors after hybrid retrieval, ranking failures are more common than candidate-generation failures. FiQA has a much larger candidate-generation bottleneck: one quarter of queries have no relevant document in the hybrid top 100.

## Main findings

1. **Sparse-versus-dense behavior is domain dependent.** BM25 is clearly stronger on SciFact, while dense retrieval is slightly stronger than BM25 on FiQA.

2. **Hybrid retrieval is valuable for candidate generation.** It raises Recall@100 substantially in both domains, although naive RRF can hurt top-rank quality.

3. **Cross-encoder reranking is consistently effective.** It preserves hybrid candidate recall while substantially improving NDCG@10 and MRR@10 on both datasets.

4. **Oracle routing headroom is real, but cheap learned routing is not robust.** The tested query-only routers fail to consistently beat strong fixed baselines.

5. **Routing does not transfer cleanly across domains.** A SciFact-trained router largely preserves SciFact's BM25 preference when evaluated frozen on FiQA, where dense retrieval is the stronger cheap fixed strategy.

6. **The dominant failure mode changes by domain.** SciFact failures are mostly ranking-related after hybrid retrieval, whereas FiQA retains a substantial retrieval bottleneck.

## Cost interpretation

Local latency measurements are useful for relative implementation comparisons but are not presented as production latency claims.

In particular:

- BM25 latency depends strongly on the current Python implementation and corpus size.
- dense query latency is measured after corpus embeddings are built and is amortized over batched queries.
- hybrid fusion timing measures fusion overhead only, not the cost of running both underlying retrievers.
- cross-encoder reranking adds substantial per-query inference cost because 100 query-document pairs are scored per query.

For scientific comparison, the routing study therefore reports both local timings and an implementation-independent strategy cost proxy.

## Limitations

- Only two BEIR datasets are used in the core study.
- The dense retriever uses one compact MS MARCO-trained sentence-transformer model.
- The cross-encoder uses one compact MS MARCO-trained reranker.
- Routing data is small, especially on SciFact, which limits the reliability of high-dimensional routing models.
- The cost proxy is deliberately simple and does not capture every deployment-specific systems cost.
- The hybrid method is standard reciprocal-rank fusion rather than a learned fusion model.
- No claim is made that the negative routing result generalizes to all feature sets, models, or search domains.

## Reproduction

The main consolidated summary can be reproduced with:

```bash
python experiments/24_results_summary.py
```

The script expects the local per-query result files generated by the earlier experiments under `results/scifact/` and `results/fiqa/`.
