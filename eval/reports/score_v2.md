# Quality score v2 and supersession

Extractions: `models-gemma-4-26B-A4B-it.jsonl`; 7956/7956 documents extracted (others keep v1).

- Spearman rho (v1 vs v2) over extracted docs: 0.912
- Documents whose score changed: 7910
- Mean v1 0.572, mean v2 0.547

## By design (mean v1 -> v2)

| design | n | v1 | v2 |
|---|---|---|---|
| case_control | 73 | 0.291 | 0.290 |
| cohort | 264 | 0.430 | 0.432 |
| controlled_trial | 59 | 0.437 | 0.419 |
| cross_sectional | 327 | 0.258 | 0.257 |
| guideline | 49 | 0.853 | 0.837 |
| meta_analysis | 1191 | 0.793 | 0.758 |
| prospective_cohort | 1265 | 0.434 | 0.433 |
| rct | 3946 | 0.553 | 0.521 |
| systematic_review | 782 | 0.747 | 0.714 |

## Flags

- india_relaxed: 776
- industry_funded: 24
- low_extraction_confidence: 2
- superseded: 82

## Supersession (82 documents superseded)

Spot-check these pairs by hand before trusting the thresholds in `score_v2.supersession`.

| older | newer | title sim | pop overlap |
|---|---|---|---|
| PMID:35472102 (2022) Association between Adherence to the Mediterranean Diet and Risk of Ty | PMID:41241045 (2025) Adherence to Mediterranean Diet and Risk of Type 2 Diabetes: An Update | 1.0 | 1.0 |
| PMID:39673331 (2024) Systematic review on maternal dietary patterns during pregnancy and of | PMID:41054961 (2025) Systematic review of maternal dietary patterns during pregnancy and of | 1.0 | 1.0 |
| PMID:34111140 (2021) Dietary diversity among pregnant women and associated factors in Ethio | PMID:41132055 (2025) Dietary diversity and associated factors among pregnant women in Ethio | 1.0 | 1.0 |
| PMID:23993665 (2013) Optimizing natural fertility: a committee opinion. | PMID:28228319 (2017) Optimizing natural fertility: a committee opinion. | 1.0 | 1.0 |
| PMID:30277579 (2018) Calcium supplementation during pregnancy for preventing hypertensive d | PMID:41330480 (2025) Calcium supplementation during pregnancy for preventing hypertensive d | 1.0 | 1.0 |
| PMID:26934541 (2016) Long-term effects of weight-reducing diets in people with hypertension | PMID:33555049 (2021) Long-term effects of weight-reducing diets in people with hypertension | 1.0 | 1.0 |
| PMID:31348529 (2019) Vitamin D supplementation for women during pregnancy. | PMID:39077939 (2024) Vitamin D supplementation for women during pregnancy. | 1.0 | 1.0 |
| PMID:25604891 (2015) Different strategies for diagnosing gestational diabetes to improve ma | PMID:28832911 (2017) Different strategies for diagnosing gestational diabetes to improve ma | 1.0 | 0.5 |
| PMID:25922862 (2015) Calcium supplementation (other than for preventing or treating hyperte | PMID:39560075 (2024) Calcium supplementation (other than for preventing or treating hyperte | 1.0 | 1.0 |
| PMID:25927101 (2015) Zinc supplementation for improving pregnancy and infant outcome. | PMID:33724446 (2021) Zinc supplementation for improving pregnancy and infant outcome. | 1.0 | 1.0 |
| PMID:24782359 (2014) Techniques of monitoring blood glucose during pregnancy for women with | PMID:28602020 (2017) Techniques of monitoring blood glucose during pregnancy for women with | 1.0 | 1.0 |
| PMID:24960615 (2014) Calcium supplementation during pregnancy for preventing hypertensive d | PMID:41330480 (2025) Calcium supplementation during pregnancy for preventing hypertensive d | 1.0 | 1.0 |
| PMID:25420475 (2014) Early additional food and fluids for healthy breastfed full-term infan | PMID:27574798 (2016) Early additional food and fluids for healthy breastfed full-term infan | 1.0 | 1.0 |
| PMID:23543574 (2013) Different types of dietary advice for women with gestational diabetes  | PMID:28236296 (2017) Different types of dietary advice for women with gestational diabetes  | 1.0 | 0.5 |
| PMID:24574258 (2014) Probiotics for preventing gestational diabetes. | PMID:33870484 (2021) Probiotics for preventing gestational diabetes. | 1.0 | 0.5 |
| PMID:23736989 (2013) Interconception care for women with a history of gestational diabetes  | PMID:28836274 (2017) Interconception care for women with a history of gestational diabetes  | 1.0 | 1.0 |
| PMID:31864397 (2019) Adherence of iron and folic acid supplementation and determinants amon | PMID:32131751 (2020) Adherence to iron-folic acid supplementation among pregnant women in E | 0.889 | 1.0 |
| PMID:25007189 (2014) Low carbohydrate versus isoenergetic balanced diets for reducing weigh | PMID:35088407 (2022) Low-carbohydrate versus balanced-carbohydrate diets for reducing weigh | 0.889 | 1.0 |
| PMID:29157030 (2019) One step versus two step approach for gestational diabetes screening:  | PMID:30173594 (2020) Screening for gestational diabetes mellitus: one step versus two step  | 0.875 | 1.0 |
| PMID:25864059 (2015) Diet and exercise interventions for preventing gestational diabetes me | PMID:29129039 (2017) Combined diet and exercise interventions for preventing gestational di | 0.875 | 1.0 |
| PMID:41098674 (2025) The role of the myo-inositol for the prevention of the gestational dia | PMID:42242341 (2026) The effect of myo-inositol supplementation on gestational diabetes mel | 0.857 | 0.5 |
| PMID:35460931 (2022) Myo-inositol supplementation for the prevention of gestational diabete | PMID:42242341 (2026) The effect of myo-inositol supplementation on gestational diabetes mel | 0.857 | 1.0 |
| PMID:28097361 (2017) Folic Acid Supplementation for the Prevention of Neural Tube Defects:  | PMID:37526714 (2023) Folic Acid Supplementation to Prevent Neural Tube Defects: Updated Evi | 0.857 | 1.0 |
| PMID:35603850 (2022) Maternal Body Mass Index, Diabetes, and Gestational Weight Gain and Ri | PMID:37049442 (2023) Maternal Body Mass Index, Gestational Weight Gain, and Risk of Cancer  | 0.833 | 1.0 |
| PMID:29446541 (2018) Group-Based Diet and Physical Activity Weight-Loss Interventions: A Sy | PMID:38017250 (2024) Dyad and group-based interventions in physical activity, diet, and wei | 0.8 | 0.5 |
