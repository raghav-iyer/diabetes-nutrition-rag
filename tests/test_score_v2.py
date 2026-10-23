import pytest

from nrag.score import v2
from test_extract import good_extraction


def _doc(doc_id, design="rct", year=2020, title="Low glycaemic index diet and diabetes", topics=("diabetes_prevention",),
         q=0.5, n=None):
    return {"doc_id": doc_id, "design": design, "year": year, "title": title, "topics": list(topics),
            "quality_score": q, "sample_size": n, "flags": [], "retracted": False}


def test_unextracted_doc_keeps_v1(cfg):
    r = v2.score_doc(cfg, _doc("PMID:1"), None)
    assert r["quality_score_v2"] == 0.5 and not r["extracted"]


def test_rob_rewards_and_penalises_rct(cfg):
    good = good_extraction(risk_of_bias_signals={"randomisation_described": "yes", "blinding": "yes",
                                                 "low_dropout": "yes", "pre_registered": "yes"})
    bad = good_extraction(risk_of_bias_signals={"randomisation_described": "no", "blinding": "no",
                                                "low_dropout": "no", "pre_registered": "not_reported"})
    hi = v2.score_doc(cfg, _doc("a", n=300), good, )["quality_score_v2"]
    lo = v2.score_doc(cfg, _doc("b", n=300), bad)["quality_score_v2"]
    assert hi > 0.5 > lo
    lo_b, hi_b = cfg["score_v2"]["rob_clip"]
    assert 0.5 * lo_b <= lo and hi <= 0.5 * hi_b + 1e-9


def test_rob_not_applied_to_cohort(cfg):
    r = v2.score_doc(cfg, _doc("a", design="prospective_cohort"), good_extraction())
    assert "rob" not in r["components"]


def test_industry_funding_flag_and_penalty(cfg):
    r = v2.score_doc(cfg, _doc("a", design="prospective_cohort"), good_extraction(funding_industry="yes"))
    assert r["quality_score_v2"] < 0.5 and "industry_funded" in r["flags"]


def test_extracted_sample_size_fills_unknown(cfg):
    r = v2.score_doc(cfg, _doc("a", design="prospective_cohort", n=None), good_extraction(sample_size=5000, funding_industry="no"))
    assert r["quality_score_v2"] > 0.5            # unknown_n 0.85 -> large 1.0
    r2 = v2.score_doc(cfg, _doc("a", design="prospective_cohort", n=300), good_extraction(sample_size=5000, funding_industry="no"))
    assert r2["quality_score_v2"] == 0.5          # known n is never overridden


def test_supersession_only_for_new_versions_of_the_same_review(cfg):
    docs = [_doc("cochrane_old", "systematic_review", 2015, "Vitamin D supplementation for women during pregnancy"),
            _doc("cochrane_new", "systematic_review", 2024, "Vitamin D supplementation for women during pregnancy"),
            _doc("lgi_old", "meta_analysis", 2015, "Low glycaemic index diet and type 2 diabetes: a meta-analysis"),
            _doc("lgi_other", "meta_analysis", 2023, "Glycaemic index of foods and incident diabetes in adults"),
            _doc("probiotic_old", "meta_analysis", 2020, "Effect of probiotic supplementation in pregnant women: a meta-analysis"),
            _doc("probiotic_gdm", "meta_analysis", 2023, "Probiotic supplement for the prevention of gestational diabetes: a meta-analysis"),
            _doc("rct", "rct", 2025, "Vitamin D supplementation for women during pregnancy")]
    exts = {d["doc_id"]: good_extraction() for d in docs}
    sup = v2.find_supersession(cfg, docs, exts)
    assert sup["cochrane_old"]["superseded_by"] == "cochrane_new"       # same review, newer version
    assert "lgi_old" not in sup                                          # similar topic, different review, no "update"
    assert "probiotic_old" not in sup                                    # same supplement, different question
    assert "cochrane_new" not in sup and "rct" not in sup


def test_explicit_update_lowers_the_title_bar(cfg):
    old = _doc("old", "meta_analysis", 2015, "Low glycaemic index diet and type 2 diabetes: a meta-analysis")
    upd = _doc("new", "meta_analysis", 2023, "Low glycaemic index diets and diabetes incidence: an updated meta-analysis")
    not_upd = _doc("new", "meta_analysis", 2023, "Low glycaemic index diets and diabetes incidence: a meta-analysis")
    exts = {d["doc_id"]: good_extraction() for d in (old, upd)}
    assert v2.find_supersession(cfg, [old, upd], exts)["old"]["superseded_by"] == "new"    # similarity 0.5 + "updated"
    assert v2.find_supersession(cfg, [old, not_upd], exts) == {}                          # same titles, no "update"


def test_guideline_not_superseded_by_review(cfg):
    docs = [_doc("g", "guideline", 2015, "Diet and diabetes prevention"),
            _doc("m", "meta_analysis", 2023, "Diet and diabetes prevention")]
    assert v2.find_supersession(cfg, docs, {}) == {}


def test_spearman():
    assert v2.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert v2.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
