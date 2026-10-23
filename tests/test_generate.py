import json

import pytest

from nrag.generate import safety
from nrag.generate.answer import Answerer, pack_sources, render
from nrag.llm import LLMError, LLMResponse
from nrag.retrieve.pipeline import RetrievalResult

# --- safety: pre-generation -----------------------------------------------------------------

@pytest.mark.parametrize("q, mode, cats", [
    ("Does a Mediterranean diet help weight loss?", "normal", []),
    ("Is it safe to take iron supplements while pregnant?", "restricted", ["pregnancy_supplements"]),
    ("How much iron do I need in pregnancy?", "restricted", ["pregnancy_supplements"]),
    ("Is vitamin B12 important when breastfeeding?", "restricted", ["pregnancy_supplements"]),
    ("Can I drink a glass of wine when breastfeeding?", "restricted", ["pregnancy_alcohol"]),
    ("Is tuna safe in pregnancy because of mercury?", "restricted", ["pregnancy_high_mercury_fish"]),
    ("Can I eat paneer made from unpasteurised milk when pregnant?", "restricted", ["pregnancy_food_safety"]),
    ("What diet controls gestational diabetes?", "restricted", ["pregnancy_gdm_management"]),
    ("Should I stop my metformin if I eat less rice?", "restricted", ["individual_advice"]),
    ("Can I stop my diabetes medicine if I follow a strict diet?", "restricted", ["individual_advice"]),
    ("Are very low calorie diets safe and effective for adults with obesity?", "normal", []),
    ("I want to eat 800 calories a day to lose weight fast", "supportive_only", ["disordered_eating"]),
    ("How do I lose 10 kg in two weeks eating 600 calories?", "supportive_only", ["disordered_eating"]),
    ("I make myself vomit after meals, what should I eat?", "supportive_only", ["disordered_eating"]),
    ("I'm pregnant and I have a fear of weight gain, can I stop eating dinner?", "supportive_only", ["disordered_eating"]),
])
def test_assess_question(cfg, q, mode, cats):
    a = safety.assess_question(q, cfg["safety"])
    assert a.mode == mode
    assert set(cats) <= set(a.categories)
    if not cats:
        assert a.categories == []


def test_population_from_query_understanding_triggers_pregnancy_rules(cfg):
    a = safety.assess_question("Are fish oil supplements helpful?", cfg["safety"], {"population_tags": ["pregnant"]})
    assert a.mode == "restricted" and "pregnancy_supplements" in a.categories
    assert cfg["safety"]["clinician_note"] in a.notes


def test_supportive_reply_has_resources_and_no_diet_advice(cfg):
    a = safety.assess_question("how to starve myself to lose weight", cfg["safety"])
    text = safety.supportive_reply(a, "disclaimer")
    assert "14416" in text and "diet plan" in text and "disclaimer" in text


@pytest.mark.parametrize("text, expected_n", [
    ("Take 60 mg daily.", 1),
    ("WHO recommends 30–60 mg of iron and 400 µg of folic acid.", 2),
    ("Aim for 1800 kcal/day.", 1),
    ("A dose of 0.4 mg folate.", 1),
    ("120 mg elemental iron and 2800 µg folic acid weekly.", 2),
    ("Take 1,000 IU daily.", 1),
    ("The trial enrolled 240 women for 12 weeks.", 0),
    ("together with 400 micrograms of folic acid", 1),
])
def test_redact_doses(cfg, text, expected_n):
    out, n = safety.redact_doses(text, cfg["safety"]["dose_pattern"], cfg["safety"]["dose_redaction"])
    assert n == expected_n
    if n:
        assert "mg" not in out and "µg" not in out and "kcal" not in out and cfg["safety"]["dose_redaction"] in out


def test_check_answer_redacts_every_field_and_keeps_guideline_position(cfg):
    a = safety.assess_question("iron supplements in pregnancy?", cfg["safety"])
    ans = {"direct_answer": "WHO recommends daily iron (30-60 mg) for all pregnant women.",
           "claims": [{"text": "Use 30 mg of iron.", "citations": ["X"]}, {"text": "Iron reduces anaemia.", "citations": ["X"]}],
           "conflicts": "An intermittent regimen of 120 mg iron weekly is an option.", "missing": "",
           "evidence_strength": {"label": "strong", "explanation": "WHO: 60 mg."}}
    out = safety.check_answer(ans, a, cfg["safety"])
    assert out["direct_answer"].startswith("WHO recommends daily iron") and "mg" not in out["direct_answer"]
    assert "mg" not in out["conflicts"] and "mg" not in out["evidence_strength"]["explanation"]
    assert len(out["claims"]) == 2 and "mg" not in out["claims"][0]["text"]
    assert any("conflicts" in f for f in out["safety_flags"])


def test_clean_claim_removes_reference_markers():
    from nrag.generate.answer import clean_claim
    assert clean_claim("Iron prevents anaemia [16].") == "Iron prevents anaemia."
    assert clean_claim("Reduces risk [31,32, 33] and [2-4] here [PMID:123].") == "Reduces risk and here."


def test_check_answer_leaves_normal_answers_alone(cfg):
    a = safety.assess_question("Does walking help weight loss?", cfg["safety"])
    ans = {"direct_answer": "Participants ate 1800 kcal/day in the trial.", "claims": []}
    assert safety.check_answer(ans, a, cfg["safety"])["direct_answer"].endswith("kcal/day in the trial.")


# --- source packing --------------------------------------------------------------------------

def _res(doc, text, **kw):
    r = {"doc_id": doc, "title": f"Title {doc}", "text": text, "parent_text": None, "year": 2020, "design": "rct",
         "source_type": "primary", "population_tags": ["pregnant"], "quality_score": 0.8, "url": None}
    r.update(kw)
    return r


def test_pack_sources_groups_by_doc_and_caps():
    text, sources = pack_sources([_res("A", "a1"), _res("A", "a2"), _res("B", "b1"), _res("C", "c1")], 2, 1000)
    assert list(sources) == ["A", "B"] and "[A] Title A" in text and "a1\n...\na2" in text and "[C]" not in text
    _, s2 = pack_sources([_res("A", "short", parent_text="full parent section")], 5, 1000)
    assert s2["A"]["body"] == "full parent section"


# --- answerer ----------------------------------------------------------------------------------

class FakeRetriever:
    def __init__(self, status="ok", results=None, analysis=None):
        self.status, self.results = status, results
        self.analysis = analysis or {"population_tags": [], "answerable": True, "answerable_reason": ""}
        self.calls = 0

    def retrieve(self, q):
        self.calls += 1
        return RetrievalResult(q, self.status, self.analysis, self.results or [], False, {}, {})


class FakeLLM:
    def __init__(self, contents):
        self.contents = list(contents)
        self.messages = []

    def chat(self, endpoint, model, messages, **kw):
        self.messages.append(messages)
        c = self.contents.pop(0)
        if isinstance(c, Exception):
            raise c
        return LLMResponse(content=c)


RESULTS = [_res("PMID:1", "Low GI diets reduced GDM.", design="meta_analysis"),
           _res("PMID:2", "No effect of low GI diet on birth weight.", design="rct", population_tags=["pregnant", "indian"]),
           _res("PMID:3", "A cross-sectional survey of diets.", design="cross_sectional")]


def _answer(**kw):
    a = {"direct_answer": "Low-GI diets may lower GDM risk.",
         "claims": [{"text": "Low GI diets reduced GDM.", "citations": ["PMID:1"]},
                    {"text": "Birth weight was unchanged.", "citations": ["PMID:2", "PMID:99"]},
                    {"text": "Survey says diets vary.", "citations": ["PMID:3"]}],
         "evidence_strength": {"label": "moderate", "explanation": "One meta-analysis and an RCT."},
         "conflicts": "", "abstain": False, "missing": ""}
    a.update(kw)
    return json.dumps(a)


def _verify(*levels):
    return json.dumps({"checks": [{"claim": i, "supported": s, "reason": "r"} for i, s in enumerate(levels, 1)]})


def _answerer(cfg, tmp_path, llm_contents, retriever=None):
    return Answerer(cfg, retriever or FakeRetriever(results=RESULTS), FakeLLM(llm_contents), "http://x", "m", tmp_path)


def test_answer_with_citation_checks(cfg, tmp_path):
    ans = _answerer(cfg, tmp_path, [_answer(), _verify("yes", "partial", "no")])
    out = ans.answer("Does a low GI diet prevent gestational diabetes?")
    a = out["answer"]
    assert out["status"] == "answered"
    assert [c["citations"] for c in a["claims"]] == [["PMID:1"], ["PMID:2"]]      # PMID:99 removed, claim 3 dropped
    assert "unknown_citation:PMID:99" in a["claims"][1]["flags"]
    assert out["verification"][2]["supported"] == "no"
    md = out["markdown"]
    assert md.startswith("Low-GI diets may lower GDM risk.") and "[PMID:1]" in md and "_(partly supported)_" in md
    assert md.count(cfg["generation"]["disclaimer"]) == 1
    assert "india_note" not in a                                                   # PMID:2 is Indian
    assert cfg["safety"]["clinician_note"] in md                                   # GDM -> pregnancy safety note


def test_strength_cap_and_india_note(cfg, tmp_path):
    only_weak = _answer(claims=[{"text": "Survey says diets vary.", "citations": ["PMID:3"]}],
                        evidence_strength={"label": "strong", "explanation": "x"})
    out = _answerer(cfg, tmp_path, [only_weak, _verify("yes")]).answer("What do adults eat?")
    a = out["answer"]
    assert a["evidence_strength"]["label"] == "moderate" and "strength_capped" in a["notes"]
    assert a["india_note"] == cfg["generation"]["india_note"]


def test_invalid_json_is_retried(cfg, tmp_path):
    out = _answerer(cfg, tmp_path, ["not json", _answer(), _verify("yes", "yes", "yes")]).answer("Weight loss diets?")
    assert out["status"] == "answered" and len(out["errors"]) == 1


def test_no_supported_claims_becomes_abstention(cfg, tmp_path):
    out = _answerer(cfg, tmp_path, [_answer(), _verify("no", "no", "no")]).answer("Weight loss diets?")
    assert out["status"] == "abstained" and "no_supported_claims" in out["answer"]["notes"]


def test_supportive_only_skips_retrieval_and_llm(cfg, tmp_path):
    r = FakeRetriever(results=RESULTS)
    out = _answerer(cfg, tmp_path, [], retriever=r).answer("How can I eat 500 calories a day to lose weight fast?")
    assert out["status"] == "supportive_only" and r.calls == 0 and "14416" in out["markdown"]


@pytest.mark.parametrize("status, expected", [("not_answerable", "can't answer this from research"),
                                              ("insufficient_evidence", "couldn't find research")])
def test_retrieval_statuses_are_explained(cfg, tmp_path, status, expected):
    r = FakeRetriever(status=status, analysis={"population_tags": [], "answerable": status != "not_answerable",
                                               "answerable_reason": "individual dosing"})
    out = _answerer(cfg, tmp_path, [], retriever=r).answer("q?")
    assert out["status"] == status and expected in out["markdown"]


def test_llm_failure_is_reported(cfg, tmp_path):
    out = _answerer(cfg, tmp_path, [LLMError("down")]).answer("Weight loss diets?")
    assert out["status"] == "generation_failed"


def test_answers_are_cached(cfg, tmp_path):
    _answerer(cfg, tmp_path, [_answer(), _verify("yes", "yes", "yes")]).answer("Weight loss diets?")
    out = _answerer(cfg, tmp_path, []).answer("Weight loss diets?")                 # no LLM contents left
    assert out["status"] == "answered"



def test_verifier_sees_the_same_source_text_as_the_generator(cfg, tmp_path):
    long_text = "Background. " * 110 + "RESULT: GDM fell by 30%."          # result at ~1,320 chars: past the old 1,200 cut
    res = [_res("PMID:1", long_text, design="meta_analysis")]
    llm = FakeLLM([_answer(claims=[{"text": "GDM fell by 30%.", "citations": ["PMID:1"]}]), _verify("yes")])
    Answerer(cfg, FakeRetriever(results=res), llm, "http://x", "m", tmp_path).answer("Low GI and GDM?")
    gen_prompt, ver_prompt = llm.messages[0][1]["content"], llm.messages[1][1]["content"]
    assert "RESULT: GDM fell by 30%." in gen_prompt and "RESULT: GDM fell by 30%." in ver_prompt



def test_redaction_collapses_repeats_and_adds_one_note(cfg):
    sc = cfg["safety"]
    out, _ = safety.redact_doses("daily 30-60 mg of iron and 0.4 mg (400 µg) of folate", sc["dose_pattern"], sc["dose_redaction"])
    assert out == "daily [amount withheld] of iron and [amount withheld] of folate"
    a = safety.assess_question("iron in pregnancy?", sc)
    safety.check_answer({"direct_answer": "Take 60 mg.", "claims": [{"text": "Use 30 mg.", "citations": []}]}, a, sc)
    assert a.notes.count(sc["dose_note"]) == 1



def test_supportive_reply_for_pregnancy_mentions_midwife_and_no_research_disclaimer(cfg, tmp_path):
    out = _answerer(cfg, tmp_path, []).answer("I am pregnant and scared of gaining weight, can I eat 800 calories a day?")
    assert out["status"] == "supportive_only" and "midwife" in out["markdown"]
    assert cfg["safety"]["support_disclaimer"] in out["markdown"] and "summarises published research" not in out["markdown"]
