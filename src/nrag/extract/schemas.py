"""Pydantic schema for PICO and evidence-field extraction.

The JSON schema generated from `Extraction` is sent to the model as a
constrained-decoding format, and every response is validated against it.
Vocabularies here must match config.yaml (tests enforce this).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

YesNoNR = Literal["yes", "no", "not_reported"]

PopulationTag = Literal[
    "general_adults", "prediabetes", "type2_diabetes", "overweight_obese", "pregnant", "lactating",
    "gestational_diabetes", "south_asian", "indian", "children", "older_adults",
]

StudyDesign = Literal[
    "guideline", "meta_analysis", "systematic_review", "rct", "non_randomised_trial", "prospective_cohort",
    "retrospective_cohort", "case_control", "cross_sectional", "narrative_review", "other",
]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Population(_Strict):
    description: str = Field(description="Who was studied, in a few words, as stated in the text.")
    tags: list[PopulationTag] = Field(description="Every tag that applies. Use only the allowed values.")


class Outcome(_Strict):
    name: str = Field(description="Outcome as named in the text, e.g. 'incident type 2 diabetes', 'HbA1c'.")
    type: Literal["hard", "surrogate"] = Field(
        description="hard = disease incidence, birth outcomes, mortality, clinical events; "
                    "surrogate = weight, BMI, HbA1c, glucose, insulin, lipids, blood pressure, other biomarkers.")


class RiskOfBias(_Strict):
    randomisation_described: YesNoNR = Field(description="yes only if the method of randomisation is described.")
    blinding: YesNoNR = Field(description="yes if participants, staff or assessors were blinded; no if stated open-label.")
    low_dropout: YesNoNR = Field(description="yes if dropout/attrition is reported at or below the threshold, "
                                             "no if above it, not_reported if not given.")
    pre_registered: YesNoNR = Field(description="yes if a trial/review registration (e.g. NCT, CTRI, ISRCTN, "
                                                "PROSPERO) is mentioned.")


class Quotes(_Strict):
    sample_size: Optional[str] = Field(description="Verbatim phrase from the text stating the sample size, or null.")
    key_finding: Optional[str] = Field(description="Verbatim phrase from the text supporting the key finding, or null.")


class Extraction(_Strict):
    study_design: StudyDesign
    population: Population
    intervention: Optional[str] = Field(description="Intervention or exposure, or null if none.")
    comparison: Optional[str] = Field(description="Comparator, or null if none stated.")
    outcomes: list[Outcome]
    effect_direction: Literal["benefit", "harm", "no_effect", "mixed", "unclear"] = Field(
        description="Direction of the main result for the intervention/exposure on the primary outcome.")
    key_finding: str = Field(description="One sentence, in your own words, of the main result.")
    sample_size: Optional[int] = Field(description="Number of participants analysed or enrolled (for reviews: "
                                                   "total participants across included studies). null if not stated.")
    duration_weeks: Optional[float] = Field(description="Intervention or follow-up duration in weeks, or null.")
    country_or_region: Optional[str] = Field(description="Where the study was done, or null if not stated.")
    risk_of_bias_signals: RiskOfBias
    funding_industry: Literal["yes", "no", "unclear"] = Field(
        description="yes if funded by food, supplement or pharmaceutical industry; no if only public/charity funding stated.")
    quotes: Quotes
    confidence: Literal["high", "medium", "low"] = Field(description="Your confidence in this extraction overall.")


def extraction_json_schema() -> dict:
    return Extraction.model_json_schema()
