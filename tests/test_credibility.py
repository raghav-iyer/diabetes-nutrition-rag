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



from nrag.ingest.credibility import assess_with_india, india_focused


def _india(**kw):
    return _rec(title="Diet in Indian adults", abstract="A study in Chennai, India.", mesh=["Humans"], **kw)


@pytest.mark.parametrize("kw, expected", [
    ({"design": "cross_sectional", "sample_size": 500}, ("included", None, True)),
    ({"design": "cross_sectional", "sample_size": 120}, ("excluded", "design_not_credible", False)),   # below relaxed min
    ({"design": "cohort", "sample_size": 400}, ("included", None, True)),
    ({"design": "case_control", "sample_size": None}, ("provisional", "sample_size_unknown", True)),
    ({"sample_size": 35}, ("included", None, True)),                                                   # small RCT
    ({"sample_size": 20}, ("excluded", "below_min_sample_size", False)),
    ({"design": "narrative_review"}, ("excluded", "design_not_credible", False)),                     # never relaxed
    ({"sample_size": 120}, ("included", None, False)),                                                 # passes strictly
    ({"medline_indexed": False, "design": "cross_sectional", "sample_size": 900},
     ("excluded", "not_medline_journal", False)),                                                      # other rules stay strict
])
def test_india_relaxed_rules(cfg, kw, expected):
    assert assess_with_india(cfg["credibility"], _india(**kw)) == expected


def test_non_india_paper_is_not_relaxed(cfg):
    rec = _rec(title="Diet in Danish adults", abstract="A study in Aarhus.", mesh=["Humans", "Denmark"],
               design="cross_sectional", sample_size=900)
    assert assess_with_india(cfg["credibility"], rec) == ("excluded", "design_not_credible", False)


def test_india_focus_detection(cfg):
    c = cfg["credibility"]
    assert india_focused(c, {"title": "x", "abstract": "", "mesh": ["Bangladesh"]})
    assert india_focused(c, {"title": "Diet among South Asians in the UK", "abstract": "", "mesh": []})
    assert india_focused(c, {"title": "Type 2 diabetes in Asian Indians", "abstract": "", "mesh": []})
    assert not india_focused(c, {"title": "Diet in American Indians", "abstract": "Native American adults.", "mesh": []})
    assert not india_focused(c, {"title": "Diet in Danish adults", "abstract": "Aarhus, Denmark.", "mesh": ["Denmark"]})


def test_relaxed_rules_can_be_disabled(cfg):
    c = {**cfg["credibility"], "india_relaxed": {**cfg["credibility"]["india_relaxed"], "enabled": False}}
    assert assess_with_india(c, _india(design="cross_sectional", sample_size=900))[2] is False
