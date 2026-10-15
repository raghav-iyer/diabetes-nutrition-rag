from datetime import date

import pytest

from nrag.ingest.pubmed import (
    classify_design, design_text, exclusion_reason, extract_sample_size, recency_factor, reference_year, score,
    size_factor,
)


def _unstructured(text):
    return [["", text]]


# --- design classification -------------------------------------------------

@pytest.mark.parametrize("pub_types, mesh, text, expected", [
    (["Practice Guideline"], [], "", ("guideline", 1.0, "pubtype")),
    (["Meta-Analysis", "Review"], [], "", ("meta_analysis", 0.9, "pubtype")),
    (["Systematic Review"], [], "", ("systematic_review", 0.85, "pubtype")),
    (["Randomized Controlled Trial"], ["Prospective Studies"], "", ("rct", 0.75, "pubtype")),
    (["Clinical Trial"], [], "", ("controlled_trial", 0.6, "pubtype")),
    # Observational Study now defers to MeSH / text:
    (["Observational Study"], ["Prospective Studies"], "", ("prospective_cohort", 0.55, "mesh")),
    (["Observational Study"], ["Cross-Sectional Studies"], "", ("cross_sectional", 0.3, "mesh")),
    (["Observational Study"], [], "nothing informative", ("observational", 0.45, "pubtype")),
    # Multicenter Study is not a design:
    (["Multicenter Study"], ["Cohort Studies"], "", ("cohort", 0.55, "mesh")),
    (["Multicenter Study"], [], "nothing informative", ("unknown", 0.2, "none")),
    (["Journal Article"], ["Prospective Studies"], "", ("prospective_cohort", 0.55, "mesh")),
    (["Journal Article"], ["Retrospective Studies"], "", ("cohort", 0.55, "mesh")),
    (["Journal Article"], ["Case-Control Studies"], "", ("case_control", 0.4, "mesh")),
    (["Journal Article"], ["Case-Control Studies", "Retrospective Studies"], "", ("case_control", 0.4, "mesh")),
    (["Journal Article"], ["Cohort Studies", "Cross-Sectional Studies"], "", ("cross_sectional", 0.3, "mesh")),
    (["Journal Article"], ["Prospective Studies", "Case-Control Studies"], "", ("prospective_cohort", 0.55, "mesh")),
    (["Journal Article"], [], "a systematic review of trials", ("systematic_review", 0.85, "text")),
    (["Journal Article"], [], "participants were randomized in a parallel trial", ("rct", 0.75, "text")),
    (["Journal Article"], [], "in this prospective population-based cohort we", ("prospective_cohort", 0.55, "text")),
    (["Journal Article"], [], "a retrospective cohort of records", ("cohort", 0.55, "text")),
    (["Journal Article"], [], "a case-control analysis", ("case_control", 0.4, "text")),
    (["Journal Article"], [], "a cross-sectional survey", ("cross_sectional", 0.3, "text")),
    (["Review"], [], "an overview of iron needs", ("narrative_review", 0.15, "pubtype")),
    (["Review"], ["Cohort Studies"], "", ("cohort", 0.55, "mesh")),
    (["Review"], [], "this systematic review", ("systematic_review", 0.85, "text")),
    (["Editorial"], [], "", ("opinion", 0.05, "pubtype")),
    (["Journal Article"], [], "nothing informative", ("unknown", 0.2, "none")),
])
def test_classify_design(rules, pub_types, mesh, text, expected):
    assert classify_design(rules, pub_types, mesh, "", _unstructured(text)) == expected


def test_prior_work_sentences_ignored(rules):
    # Legacy classified this as a meta-analysis from the background sentence.
    text = "Previous meta-analyses suggested benefit. We surveyed 200 adults in a cross-sectional design."
    assert classify_design(rules, ["Journal Article"], [], "", _unstructured(text))[0] == "cross_sectional"


def test_background_section_ignored(rules):
    sections = [["BACKGROUND", "Several meta-analyses report benefits."],
                ["METHODS", "A cross-sectional survey of 300 women."]]
    assert classify_design(rules, ["Journal Article"], [], "", sections)[0] == "cross_sectional"


def test_title_always_counts(rules):
    label, _, method = classify_design(rules, ["Journal Article"], [], "Pulses and weight: a meta-analysis", [])
    assert (label, method) == ("meta_analysis", "text")


def test_design_text_keeps_own_study_sentences(rules):
    text = design_text(rules, "T", [["", "Earlier trials were small. We conducted a meta-analysis of 12 trials."]])
    assert "Earlier trials" not in text and "We conducted a meta-analysis" in text


# --- sample size --------------------------------------------------------------

@pytest.mark.parametrize("abstract, expected", [
    ("We enrolled n = 240 adults.", 240),
    ("N=1,234 participants took part.", 1234),
    ("Data from 12,000 women and 300 men.", 12000),
    ("Included 23 trials.", None),                     # trial counts are not participants
    ("n = 4 adults", None),                            # below minimum / too short
    ("There were 2,015 participants.", 2015),          # comma: not a year (legacy skipped this)
    ("We enrolled n = 1950 women.", 1950),             # n = ... is never a year
    ("A total of 2000 participants enrolled.", 2000),
    ("In 2015 women were invited to screening.", None),    # reads like a year
    ("Between 2012 and 2014 participants were recruited.", None),
    ("From 2010-2014 participants were recruited.", None),
    ("We included 1500 men and 2000 women.", 2000),
    ("No numbers here.", None),
])
def test_extract_sample_size(rules, abstract, expected):
    assert extract_sample_size(rules, abstract) == expected


# --- scoring factors ----------------------------------------------------------

@pytest.mark.parametrize("design, n, expected", [
    ("meta_analysis", 10, 1.0),
    ("guideline", None, 1.0),
    ("rct", None, 0.85),
    ("rct", 49, 0.60),
    ("rct", 50, 0.80),
    ("cohort", 199, 0.80),
    ("cohort", 200, 0.90),
    ("cohort", 999, 0.90),
    ("cohort", 1000, 1.00),
])
def test_size_factor(cfg, design, n, expected):
    assert size_factor(cfg["scoring"]["size"], design, n) == expected


def test_recency_factor():
    assert recency_factor(2026, 2026, 10, 0.6) == 1.0
    assert recency_factor(2016, 2026, 10, 0.6) == pytest.approx(0.8)
    assert recency_factor(1900, 2026, 10, 0.6) == pytest.approx(0.6, abs=1e-3)
    assert recency_factor(None, 2026, 10, 0.6) == 0.6
    assert recency_factor(2030, 2026, 10, 0.6) == 1.0     # future year clamps to age 0


def test_reference_year_defaults_to_today(rules):
    assert reference_year(rules, date(2031, 6, 1)) == 2031


def _rec(**kw):
    rec = {"design": "rct", "design_score": 0.75, "sample_size": 120, "year": 2016,
           "expression_of_concern": False, "medline_indexed": True, "mesh": ["Humans"]}
    rec.update(kw)
    return rec


def test_score_components_and_value(rules):
    r = score(rules, _rec(), 2026)
    assert r["score_components"] == {"design": 0.75, "size": 0.8, "recency": 0.8}
    assert r["quality_score"] == pytest.approx(0.75 * 0.8 * 0.8, abs=1e-4)
    assert r["flags"] == []


def test_score_flags_and_expression_of_concern(rules):
    r = score(rules, _rec(expression_of_concern=True, medline_indexed=False, mesh=[]), 2026)
    assert r["quality_score"] == pytest.approx(0.75 * 0.8 * 0.8 * 0.5, abs=1e-4)
    assert r["flags"] == ["expression_of_concern", "not_medline_indexed", "not_yet_mesh_indexed"]


# --- exclusion ----------------------------------------------------------------

def _ex(**kw):
    rec = {"retracted": False, "pub_types": ["Journal Article"], "abstract": "x" * 250,
           "mesh": ["Humans"], "languages": ["eng"], "design": "rct"}
    rec.update(kw)
    return rec


@pytest.mark.parametrize("kw, expected", [
    ({}, None),
    ({"retracted": True}, "retracted"),
    ({"pub_types": ["Published Erratum"]}, "notice_not_article"),
    ({"pub_types": ["Expression of Concern"]}, "notice_not_article"),
    ({"abstract": "x" * 199}, "no_or_short_abstract"),
    ({"abstract": ""}, "no_or_short_abstract"),
    ({"mesh": ["Animals", "Rats"]}, "not_human"),
    ({"mesh": []}, None),                              # not yet MeSH-indexed: kept
    ({"languages": ["ger"]}, "not_english"),
    ({"languages": ["eng", "ger"]}, None),
    ({"design": "opinion"}, "opinion_piece"),
])
def test_exclusion_reason(rules, kw, expected):
    assert exclusion_reason(rules, _ex(**kw)) == expected
