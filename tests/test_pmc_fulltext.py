import xml.etree.ElementTree as ET

import pytest
from conftest import PMC_ARTICLES

from nrag.ingest.pmc_fulltext import detect_licence, licence_from_url, parse_jats, section_name, table_to_markdown


def articles(cfg):
    return {a["pmcid"]: a for a in parse_jats(PMC_ARTICLES.read_bytes(), cfg["pmc"])}


@pytest.mark.parametrize("url, code", [
    ("https://creativecommons.org/licenses/by/4.0/", "CC-BY"),
    ("http://creativecommons.org/licenses/by-sa/3.0/igo/", "CC-BY-SA"),
    ("https://creativecommons.org/licenses/by-nc/4.0/", "CC-BY-NC"),
    ("https://creativecommons.org/licenses/by-nc-nd/4.0/", "CC-BY-NC-ND"),
    ("https://creativecommons.org/publicdomain/zero/1.0/", "CC0"),
    ("https://example.com/terms", None),
])
def test_licence_from_url(url, code):
    assert licence_from_url(url) == code


def _lic(inner, ns='xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:ali="http://www.niso.org/schemas/ali/1.0/"'):
    return ET.fromstring(f"<article {ns}><front><permissions><license{inner}</license></permissions></front></article>")


def test_detect_licence_variants():
    assert detect_licence(_lic(' xlink:href="https://creativecommons.org/licenses/by-nc/4.0/"><license-p>x</license-p>'))[0] == "CC-BY-NC"
    assert detect_licence(_lic('><license-p>Available for text mining under fair use.</license-p>'))[0] == "pmc-text-mining-only"
    assert detect_licence(_lic('><license-p>Granted for the COVID-19 pandemic.</license-p>'))[0] == "pmc-covid-temporary"
    assert detect_licence(_lic('><license-p>All rights reserved.</license-p>'))[0] == "other"
    assert detect_licence(ET.fromstring("<article><front/></article>")) == ("none", None)


def test_cc_by_article_parsed(cfg):
    a = articles(cfg)["PMC9000001"]
    assert a["allowed"] and a["licence"] == "CC-BY" and a["pmid"] == "90000001"
    assert a["licence_url"] == "https://creativecommons.org/licenses/by/4.0/"
    names = [s["name"] for s in a["sections"]]
    assert names == ["introduction", "methods", "results", "discussion", "conclusion"]   # funding + COI skipped


def test_section_text_excludes_tables_figures_and_titles(cfg):
    secs = {s["name"]: s for s in articles(cfg)["PMC9000001"]["sections"]}
    methods = secs["methods"]["text"]
    assert "Participants\nWe recruited 240 adults" in methods     # nested sub-section titles kept inline
    assert "Baseline characteristics" not in methods and "45.2" not in methods
    assert not methods.startswith("Materials and methods")
    assert "Flow diagram caption" not in secs["results"]["text"]
    assert "[1]" in secs["introduction"]["text"]                  # citation markers survive
    assert "Abstract text" not in " ".join(s["text"] for s in secs.values())


def test_tables_kept_whole_including_floats_group(cfg):
    tables = articles(cfg)["PMC9000001"]["tables"]
    assert [t["label"] for t in tables] == ["Table 1", "Table 2"]
    t1 = tables[0]
    assert t1["caption"] == "Baseline characteristics"
    assert t1["rows"] == [["Characteristic", "Intervention", "Control"], ["Age, years", "45.2", "44.8"],
                          ["BMI, kg/m2", "26.1", "25.9"]]
    assert t1["footer"] == "Values are means."
    md = table_to_markdown(t1)
    assert md.splitlines()[0] == "| Characteristic | Intervention | Control |" and md.count("\n") == 3


def test_non_open_licence_not_stored(cfg):
    a = articles(cfg)["PMC9000002"]
    assert a["licence"] == "pmc-covid-temporary" and not a["allowed"]
    assert a["sections"] == [] and a["tables"] == []


def test_section_name_mapping(cfg):
    m = cfg["pmc"]["section_map"]
    assert section_name("Background", None, m) == "introduction"
    assert section_name("Study design and participants", None, m) == "methods"
    assert section_name("Main findings", None, m) == "results"
    assert section_name("Strengths and limitations", None, m) == "other"
    assert section_name("Untitled", "methods", m) == "methods"
