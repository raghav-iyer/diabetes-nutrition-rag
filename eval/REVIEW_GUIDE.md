# How to fill in `extraction_review.csv`

The spreadsheet checks what the local model (gpt-oss 20B, low reasoning) extracted from each
abstract. Your answers become the ground truth for `nrag extract-accuracy`.

## Layout

- One block of 16 rows per paper, one row per field. The title and abstract appear on the first row of each block.
- Column **A** is the model's answer. Column **B** is empty (only one model is being reviewed).
- The first 30 papers (480 rows) are the quick-screen set. If you're short of time, do those first:
  partial labels work, and unlabelled rows are skipped.

## What to write

| Column | What to enter |
|---|---|
| `verdict` | **A** if the answer is correct. **neither** if it is wrong. |
| `correct_value` | Only when the verdict is **neither**: the right value, written the same way as column A. |
| `notes` | Optional: anything odd ("abstract is a protocol", "off-topic"). |

Judge only against the **title and abstract shown**. If the abstract doesn't say something, the correct
answer is `null` / `not_reported` / `unclear`, even if you know the real study.

## Field rules

| Field | Correct when |
|---|---|
| study_design | Matches the design described (rct, meta_analysis, systematic_review, prospective_cohort, retrospective_cohort, case_control, cross_sectional, non_randomised_trial, narrative_review, guideline, other) |
| population.tags | Exactly the tags that apply, separated by `; `. Allowed: general_adults, prediabetes, type2_diabetes, overweight_obese, pregnant, lactating, gestational_diabetes, south_asian, indian, children, older_adults |
| population.description | A fair short description of who was studied |
| intervention / comparison | What was tested and what it was compared with (`null` if none) |
| outcomes | Main outcomes, each labelled `[hard]` (disease, birth outcomes, death) or `[surrogate]` (weight, HbA1c, glucose, lipids, biomarkers) |
| effect_direction | benefit / harm / no_effect / mixed / unclear for the main result |
| key_finding | Accurately states the main result (wording can differ) |
| sample_size | Participants analysed or enrolled (reviews: total participants). Within 5% counts as correct |
| duration_weeks | Intervention or follow-up length in weeks (months × 4.35). `null` if not stated |
| country_or_region | Where the study was done, or `null` |
| risk_of_bias_signals.* | yes / no / not_reported, as stated in the abstract. A "yes" the abstract doesn't support is wrong |
| funding_industry | yes / no / unclear. "no" needs an explicit statement of non-industry funding |

## When you're done

Save the file as CSV (keep the column names), then run:

```bash
.venv/bin/nrag extract-accuracy
```
