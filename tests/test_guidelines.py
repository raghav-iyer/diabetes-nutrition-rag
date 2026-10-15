import re

import pymupdf
import pytest
import yaml

from nrag.ingest.guidelines import GuidelineEntry, find_recommendations, load_registry, parse_entry, parse_guideline_pdf

BODY = ("Lifestyle behaviour change programmes reduce progression from prediabetes to type 2 diabetes. "
        "This section summarises the recommendations for adults at risk.")


def make_pdf(path):
    """Synthetic guideline PDF: a large heading, body text, ADA-style graded
    recommendations sharing one paragraph, a bold numbered sub-heading and a ruled table."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 80), "3. Prevention or Delay of Diabetes", fontsize=16, fontname="hebo")
    page.insert_textbox(pymupdf.Rect(72, 95, 520, 150), BODY, fontsize=10, fontname="helv")
    recs = ("3.1 Monitor for the development of type 2 diabetes in people with prediabetes at least annually. B "
            "3.2 Refer adults with overweight and prediabetes to an intensive lifestyle behaviour change "
            "program to achieve and maintain weight loss of at least 7%. A")
    page.insert_textbox(pymupdf.Rect(72, 160, 520, 240), recs, fontsize=10, fontname="helv")
    page.insert_text((72, 270), "3.3 Pharmacological interventions", fontsize=10, fontname="hebo")
    page.insert_textbox(pymupdf.Rect(72, 280, 520, 330),
                        "Metformin may be considered for selected high-risk adults according to the evidence.",
                        fontsize=10, fontname="helv")
    # 3x3 ruled table
    x0, y0, w, h = 72, 360, 150, 22
    for r in range(4):
        page.draw_line((x0, y0 + r * h), (x0 + 3 * w, y0 + r * h))
    for c in range(4):
        page.draw_line((x0 + c * w, y0), (x0 + c * w, y0 + 3 * h))
    cells = [["Outcome", "Lifestyle", "Control"], ["Incidence", "4.8", "11.0"], ["Weight change", "-5.6", "-0.1"]]
    for r, row in enumerate(cells):
        for c, txt in enumerate(row):
            page.insert_text((x0 + c * w + 4, y0 + r * h + 15), txt, fontsize=9, fontname="helv")
    doc.save(path)


@pytest.fixture()
def pdf(tmp_path):
    p = tmp_path / "g.pdf"
    make_pdf(p)
    return p


def test_sections_and_headings(cfg, pdf):
    parsed = parse_guideline_pdf(pdf, cfg["guidelines"])
    titles = [s["title"] for s in parsed["sections"]]
    assert titles == ["3. Prevention or Delay of Diabetes", "3.3 Pharmacological interventions"]
    assert parsed["sections"][0]["text"].startswith("Lifestyle behaviour change programmes")
    assert parsed["body_font_size"] == 10.0


def test_tables_kept_whole_and_removed_from_text(cfg, pdf):
    parsed = parse_guideline_pdf(pdf, cfg["guidelines"])
    assert len(parsed["tables"]) == 1
    assert parsed["tables"][0]["rows"] == [["Outcome", "Lifestyle", "Control"], ["Incidence", "4.8", "11.0"],
                                           ["Weight change", "-5.6", "-0.1"]]
    assert all("Incidence" not in s["text"] for s in parsed["sections"])


def test_ada_recommendations_with_grades(cfg, pdf, tmp_path):
    entry = GuidelineEntry(id="test_ada", title="T", issuing_body="ADA", year=2026, topics=["diabetes_prevention"],
                           population=["prediabetes"], file=str(pdf), grade_scheme="ada")
    parsed = parse_entry(entry, cfg)
    recs = parsed["recommendations"]
    assert [(r["grade"], r["text"][:3]) for r in recs] == [("B", "3.1"), ("A", "3.2")]
    assert recs[1]["text"].endswith("at least 7%.")
    assert recs[0]["id"] == "test_ada:r1" and recs[0]["section"] == "3. Prevention or Delay of Diabetes"
    assert parsed["doc_id"] == "GUIDELINE:test_ada" and "paragraphs" not in parsed["sections"][0]


def test_missing_file_returns_none(cfg, tmp_path):
    entry = GuidelineEntry(id="x", title="T", issuing_body="B", year=2020, topics=[], population=[],
                           file=str(tmp_path / "nope.pdf"))
    assert parse_entry(entry, cfg) is None


@pytest.mark.parametrize("scheme, para, grade", [
    ("who", "Counselling about healthy eating and keeping physically active during pregnancy is recommended. Recommended", "Recommended"),
    ("who", "Vitamin A supplementation is only recommended for pregnant women in areas where deficiency is a severe public health problem. Context-specific recommendation", "Context-specific recommendation"),
    ("grade_system", "We suggest low glycaemic index diets for women with gestational diabetes (conditional recommendation, low certainty evidence).", "conditional recommendation, low certainty evidence"),
])
def test_other_grade_schemes(cfg, scheme, para, grade):
    pattern = re.compile(cfg["guidelines"]["grade_patterns"][scheme])
    recs = find_recommendations([{"title": "S", "page_start": 1, "paragraphs": [para]}], pattern, scheme, "g")
    assert len(recs) == 1 and recs[0]["grade"] == grade


def test_ungraded_paragraph_is_not_a_recommendation(cfg):
    pattern = re.compile(cfg["guidelines"]["grade_patterns"]["ada"])
    assert find_recommendations([{"title": "S", "page_start": 1, "paragraphs": [BODY]}], pattern, "ada", "g") == []


def test_seeded_registry_is_valid(cfg):
    entries = load_registry(cfg)
    assert {e.id for e in entries} == {"icmr_nin_dgi_2024", "ada_soc_2026", "who_anc_2016", "iom_gwg_2009",
                                       "acog_co548_gwg", "rssdi_cpr_t2dm_2022"}
    assert next(e for e in entries if e.id == "iom_gwg_2009").year == 2009


@pytest.mark.parametrize("bad, msg", [
    ({"topics": ["astrology"]}, "unknown topics"),
    ({"population": ["martians"]}, "unknown population"),
    ({"supersedes": "missing_id"}, "supersedes unknown"),
    ({"grade_scheme": "nope"}, "unknown grade_scheme"),
    ({"id": "Bad-ID"}, "lowercase"),
])
def test_registry_validation(cfg, tmp_path, bad, msg):
    entry = {"id": "ok_id", "title": "T", "issuing_body": "B", "year": 2020, "topics": ["weight_management"],
             "population": ["general_adults"], **bad}
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump({"guidelines": [entry]}))
    with pytest.raises(ValueError, match=msg):
        load_registry(cfg, path)
