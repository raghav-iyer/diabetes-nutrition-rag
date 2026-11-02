# Evaluation run `20261023-170501_ablation`

Questions: 52 (retrieval-labelled: 52). Relevance labels: 0 human (gold), 1085 LLM-drafted (silver). Golden questions expert-verified: 0/60. Median seconds/question: None.

> **Caveat:** all relevance labels are LLM drafts not yet checked by a human; treat retrieval numbers as provisional.

## Ablations (delta vs base, 95% CI)

| Variant | ndcg@10 | recall@10 | mrr |
|---|---|---|---|
| base | 0.668 | 0.569 | 0.866 |
| no_bm25 | -0.046 (-0.074, -0.019) | -0.079 (-0.132, -0.035) | -0.026 (-0.073, +0.017) |
| no_dense | -0.113 (-0.142, -0.083) | -0.107 (-0.146, -0.075) | -0.066 (-0.122, -0.011) |
| no_rewrite | -0.057 (-0.085, -0.030) | -0.059 (-0.092, -0.029) | -0.043 (-0.091, -0.006) |
| no_query_understanding | -0.039 (-0.078, +0.009) | -0.031 (-0.081, +0.028) | -0.023 (-0.078, +0.027) |
| concat_fusion | -0.056 (-0.091, -0.022) | -0.086 (-0.141, -0.040) | -0.032 (-0.071, -0.003) |
| no_filters | -0.000 (-0.013, +0.013) | -0.009 (-0.024, +0.003) | -0.001 (-0.001, +0.000) |
| no_rerank | -0.026 (-0.067, +0.015) | -0.068 (-0.116, -0.024) | -0.009 (-0.053, +0.030) |
| no_quality_weighting | +0.015 (-0.017, +0.051) | -0.037 (-0.089, +0.003) | -0.016 (-0.073, +0.046) |
| no_retry | +0.000 (+0.000, +0.000) | +0.000 (+0.000, +0.000) | +0.000 (+0.000, +0.000) |
| llamaindex_baseline | -0.066 (-0.122, -0.011) | -0.115 (-0.184, -0.043) | -0.028 (-0.077, +0.016) |

