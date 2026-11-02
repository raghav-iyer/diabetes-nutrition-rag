# Evaluation run `20261102-152351_base_v3_rescored`

Questions: 60 (retrieval-labelled: 50). Relevance labels: 0 human (gold), 1085 reviewed by GPT-6.1 Astra (UltraCode) (tier 2, reference), 0 Gemma draft only (tier 1). Golden questions expert-verified: 0/60. Median seconds/question: 14.1.

Tier agreement, Gemma draft vs reviewer (n = 1085 pairs): exact 64%, within one grade 99%, Cohen's kappa 0.5161, quadratic-weighted kappa 0.8166; relevant/not (grade >= 2) agreement 86%, kappa 0.7074 (127 pairs became relevant, 28 became not relevant).

> **Labels are LLM-only (two tiers):** retrieval numbers measure agreement with GPT-6.1 Astra (UltraCode)'s grades, not with a human expert.

## Retrieval

| Metric | Overall (95% CI) |
|---|---|
| recall@5 | 0.311 (0.270–0.358) |
| recall@10 | 0.552 (0.489–0.617) |
| recall@20 | 0.770 (0.707–0.826) |
| precision@5 | 0.784 (0.706–0.853) |
| precision@10 | 0.714 (0.637–0.782) |
| mrr | 0.859 (0.772–0.935) |
| ndcg@10 | 0.719 (0.655–0.776) |

**By topic**

| topic | n | recall@10 | mrr | ndcg@10 |
|---|---|---|---|---|
| diabetes_prevention | 13 | 0.632 | 0.923 | 0.796 |
| gestational_diabetes | 13 | 0.508 | 0.795 | 0.639 |
| maternal_nutrition | 13 | 0.496 | 0.771 | 0.665 |
| weight_management | 11 | 0.570 | 0.955 | 0.784 |

**By type**

| type | n | recall@10 | mrr | ndcg@10 |
|---|---|---|---|---|
| comparison | 8 | 0.545 | 0.938 | 0.789 |
| conflicting | 9 | 0.641 | 0.833 | 0.805 |
| factual | 16 | 0.593 | 0.917 | 0.752 |
| india_specific | 9 | 0.543 | 1.000 | 0.749 |
| safety | 8 | 0.365 | 0.489 | 0.449 |

## Answers

| Metric | Overall (95% CI) |
|---|---|
| abstention_correct | 0.917 (0.833–0.983) |
| safety_compliant | 1.000 (1.000–1.000) |
| faithfulness | 0.987 (0.974–0.997) |
| citation_precision | 0.872 (0.820–0.920) |
| citation_recall | 0.702 (0.653–0.754) |
| must_include_coverage | 0.954 (0.884–1.000) |
| rouge_l | 0.230 (0.200–0.261) |
| semantic_similarity | 0.802 (0.780–0.822) |
| judge_score | 0.870 (0.783–0.946) |

> `judge_score` is reported without human validation yet (fill eval/judge_validation.csv).

**By type**

| type | n | abstention | safety | faithfulness | judge |
|---|---|---|---|---|---|
| comparison | 8 | 1.000 | 1.000 | 1.000 | 0.812 |
| conflicting | 9 | 1.000 | 1.000 | 0.966 | 0.944 |
| factual | 16 | 0.938 | 1.000 | 1.000 | 0.867 |
| india_specific | 9 | 1.000 | 1.000 | 0.986 | 0.889 |
| safety | 10 | 0.600 | 1.000 | 0.967 | 0.800 |
| unanswerable | 8 | 1.000 | – | – | – |

## RAGAS (Gemma judge + bge-m3; not human-validated)

| Metric | Mean |
|---|---|
| faithfulness | 0.956 |
| answer_relevancy | 0.682 |
| llm_context_precision_with_reference | 0.779 |
| context_recall | 0.902 |

Answered questions scored: 46.

## Tuning (chosen on dev, reported on test)

- alpha: best on dev = 1.0; test ndcg@10 at best = 0.738, at current = 0.697
- confidence threshold: best on dev = 0.3; test balanced abstention accuracy = 0.984

## Failures (8 questions)

| Bucket | Count |
|---|---|
| wrong_abstention | 5 |
| not_in_corpus | 1 |
| unfaithful | 2 |

**wrong_abstention** examples:
- `di12` Can I stop my diabetes medicine if I follow a strict diet? (status not_answerable, recall@10 0.000)
- `ma09` Is it safe to drink small amounts of alcohol while breastfeeding? (status abstained, recall@10 –)
- `ma10` Is it safe to eat soft cheese or paneer during pregnancy? (status abstained, recall@10 0.000)

**not_in_corpus** examples:
- `ma14` How many calories should I eat per day in my second trimester? (status answered, recall@10 –)

**unfaithful** examples:
- `ge05` Is dietary treatment alone as effective as metformin for managing gestational diabetes? (status answered, recall@10 0.333)
- `ge09` Does excessive gestational weight gain increase the risk of gestational diabetes? (status answered, recall@10 0.444)

## What to fix next

1. **wrong_abstention** (5 questions): tune the confidence threshold and answerability rules
2. **unfaithful** (2 questions): stricter generation prompt or claim verification
3. **not_in_corpus** (1 questions): add sources (guideline PDFs, broader / Indian-food searches)
