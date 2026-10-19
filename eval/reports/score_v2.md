# Quality score v2 and supersession

Extractions: `models-gemma-4-26B-A4B-it.jsonl`; 606/606 documents extracted (others keep v1).

- Spearman rho (v1 vs v2) over extracted docs: 0.917
- Documents whose score changed: 605
- Mean v1 0.616, mean v2 0.590

## By design (mean v1 -> v2)

| design | n | v1 | v2 |
|---|---|---|---|
| guideline | 4 | 0.921 | 0.908 |
| meta_analysis | 100 | 0.812 | 0.774 |
| prospective_cohort | 116 | 0.446 | 0.440 |
| rct | 319 | 0.577 | 0.547 |
| systematic_review | 67 | 0.783 | 0.756 |

## Flags

- industry_funded: 3
- superseded: 4

## Supersession (4 documents superseded)

Spot-check these pairs by hand before trusting the thresholds in `score_v2.supersession`.

| older | newer | title sim | pop overlap |
|---|---|---|---|
| PMID:29157030 (2019) One step versus two step approach for gestational diabetes screening:  | PMID:30173594 (2020) Screening for gestational diabetes mellitus: one step versus two step  | 0.714 | 1.0 |
| PMID:30231792 (2020) The impact of omega-3 fatty acid supplementation on glycemic control i | PMID:32942938 (2022) Association between omega-3 fatty acid supplementation and lower risk  | 0.5 | 0.5 |
| PMID:28927313 (2019) Efficacy of probiotic supplement for gestational diabetes mellitus: a  | PMID:36368685 (2023) Probiotic Supplement for the Prevention of Gestational Diabetes: A Met | 0.5 | 0.5 |
| PMID:28409688 (2018) Effect of exercise during pregnancy to prevent gestational diabetes me | PMID:42457535 (2026) Effects of supervised exercise during pregnancy on gestational diabete | 0.5 | 1.0 |
