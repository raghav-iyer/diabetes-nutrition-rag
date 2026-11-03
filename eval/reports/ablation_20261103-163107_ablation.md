# Evaluation run `20261103-163107_ablation`

Questions: 50 (retrieval-labelled: 50). Relevance labels: 0 human (gold), 1085 reviewed by GPT-6.1 Astra (UltraCode) (tier 2, reference), 0 Gemma draft only (tier 1). Golden questions expert-verified: 0/60. Median seconds/question: None.

Tier agreement, Gemma draft vs reviewer (n = 1085 pairs): exact 64%, within one grade 99%, Cohen's kappa 0.5161, quadratic-weighted kappa 0.8166; relevant/not (grade >= 2) agreement 86%, kappa 0.7074 (127 pairs became relevant, 28 became not relevant).

> **Labels are LLM-only (two tiers):** retrieval numbers measure agreement with GPT-6.1 Astra (UltraCode)'s grades, not with a human expert.

## Ablations (delta vs base, 95% CI)

| Variant | ndcg@10 | recall@10 | mrr |
|---|---|---|---|
| base | 0.719 | 0.552 | 0.859 |
| no_bm25 | -0.050 (-0.079, -0.022) | -0.053 (-0.089, -0.016) | +0.007 (-0.034, +0.048) |
| no_dense | -0.120 (-0.155, -0.085) | -0.100 (-0.144, -0.059) | -0.032 (-0.093, +0.031) |
| no_rewrite | -0.053 (-0.080, -0.026) | -0.038 (-0.067, -0.007) | -0.022 (-0.070, +0.021) |
| no_query_understanding | -0.037 (-0.081, +0.011) | -0.021 (-0.063, +0.031) | -0.001 (-0.055, +0.053) |
| concat_fusion | -0.061 (-0.091, -0.030) | -0.053 (-0.094, -0.013) | +0.007 (-0.034, +0.051) |
| no_filters | -0.010 (-0.025, +0.005) | -0.011 (-0.028, +0.003) | +0.010 (+0.000, +0.031) |
| no_rerank | -0.026 (-0.068, +0.016) | -0.061 (-0.104, -0.018) | +0.044 (-0.022, +0.109) |
| no_quality_weighting | +0.008 (-0.023, +0.038) | -0.012 (-0.037, +0.018) | +0.046 (-0.008, +0.104) |
| no_retry | +0.000 (+0.000, +0.000) | +0.000 (+0.000, +0.000) | +0.000 (+0.000, +0.000) |
| strategy=recursive | -0.044 (-0.063, -0.024) | -0.042 (-0.064, -0.020) | -0.003 (-0.031, +0.028) |
| strategy=sliding_window | -0.022 (-0.041, -0.004) | -0.017 (-0.034, +0.001) | -0.012 (-0.051, +0.020) |
| strategy=hierarchical | -0.094 (-0.123, -0.066) | -0.072 (-0.104, -0.038) | -0.048 (-0.104, +0.004) |
| llamaindex_baseline | -0.057 (-0.120, +0.006) | -0.108 (-0.181, -0.035) | +0.062 (+0.003, +0.124) |

