import pytest

from nrag.ingest.credibility import assess


def _rec(**kw):
    rec = {"pub_types": ["Journal Article", "Randomized Controlled Trial"], "year": 2020,
           "expression_of_concern": False, "medline_indexed": True, "design": "rct",
           "design_method": "pubtype", "sample_size": 120}
    rec.update(kw)
    return rec


@pytest.mark.parametrize("kw, expected", [
    ({}, ("included", None)),
    ({"design": "meta_analysis", "sample_size": None}, ("included", None)),     # no size rule for reviews
    ({"design": "guideline", "sample_size": None}, ("included", None)),
    ({"design": "prospective_cohort", "sample_size": 5000}, ("included", None)),
    ({"design": "prospective_cohort", "sample_size": 999}, ("excluded", "below_min_sample_size")),
    ({"design": "prospective_cohort", "sample_size": None}, ("provisional", "sample_size_unknown")),
    ({"sample_size": 49}, ("excluded", "below_min_sample_size")),
    ({"sample_size": 50}, ("included", None)),
    ({"sample_size": None}, ("provisional", "sample_size_unknown")),
    ({"design_method": "text"}, ("provisional", "design_from_text_only")),
    ({"design_method": "text", "sample_size": None}, ("provisional", "sample_size_unknown")),
    ({"design": "cohort"}, ("excluded", "design_not_credible")),               # retrospective / unspecified
    ({"design": "controlled_trial"}, ("excluded", "design_not_credible")),
    ({"design": "cross_sectional"}, ("excluded", "design_not_credible")),
    ({"design": "narrative_review"}, ("excluded", "design_not_credible")),
    ({"design": "unknown"}, ("excluded", "design_not_credible")),
    ({"year": 2012}, ("excluded", "before_min_year")),
    ({"year": 2013}, ("included", None)),
    ({"year": None}, ("included", None)),
    ({"medline_indexed": False}, ("excluded", "not_medline_journal")),
    ({"expression_of_concern": True}, ("excluded", "expression_of_concern")),
    ({"pub_types": ["Clinical Trial Protocol", "Randomized Controlled Trial"]}, ("excluded", "excluded_pub_type")),
    ({"pub_types": ["Preprint"]}, ("excluded", "excluded_pub_type")),
])
def test_assess(cfg, kw, expected):
    assert assess(cfg["credibility"], _rec(**kw)) == expected


def test_unknown_n_policy_is_configurable(cfg):
    cred = dict(cfg["credibility"])
    assert assess({**cred, "unknown_n": "exclude"}, _rec(sample_size=None)) == ("excluded", "sample_size_unknown")
    assert assess({**cred, "unknown_n": "include"}, _rec(sample_size=None)) == ("included", None)


def test_dropping_cohorts_from_policy(cfg):
    cred = {**cfg["credibility"], "include_designs": ["guideline", "meta_analysis", "systematic_review", "rct"]}
    assert assess(cred, _rec(design="prospective_cohort", sample_size=5000)) == ("excluded", "design_not_credible")
