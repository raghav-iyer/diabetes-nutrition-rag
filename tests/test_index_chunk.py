import numpy as np
import pytest

from nrag.index import chunk as ch
from nrag.index.embed import EmbeddingCache, Embedder

TOK = ch.WhitespaceTokenizer()
C = {"target_tokens": 40, "max_tokens": 50, "min_tokens": 8, "overlap_tokens": 5, "overlap_sentences": 1,
     "hierarchical": {"child_tokens": 15, "child_overlap": 3}, "semantic": {"breakpoint_percentile": 20}}


def para(n_sent, words=10, tag="w"):
    """n sentences of `words` tokens; sentences start with a capital, like real text."""
    return " ".join(" ".join(f"{tag.capitalize() if j == 0 else tag}{i}_{j}" for j in range(words - 1)) + f" end{i}."
                    for i in range(n_sent))


def test_sentences_handles_abbreviations_and_decimals():
    s = ch.sentences("Intake was high, e.g. 3 servings. Smith et al. found benefit. HR 0.58 (95% CI 0.4-0.8). Next one.")
    assert s == ["Intake was high, e.g. 3 servings.", "Smith et al. found benefit.", "HR 0.58 (95% CI 0.4-0.8).",
                 "Next one."]


@pytest.mark.parametrize("strategy", ["token_aware", "recursive", "sliding_window", "hierarchical"])
def test_strategies_respect_token_budget_and_cover_text(strategy):
    text = para(20)                                       # 200 words
    pieces = ch.split_text(strategy, text, TOK, C)
    limit = C["max_tokens"] if strategy != "hierarchical" else C["hierarchical"]["child_tokens"] + C["hierarchical"]["child_overlap"]
    assert pieces and all(ch.n_tokens(TOK, p) <= limit + C["min_tokens"] for p in pieces)
    covered = set(" ".join(pieces).split())
    assert covered == set(text.split())                   # nothing dropped


def test_token_aware_keeps_sentences_whole_with_overlap():
    pieces = ch.split_token_aware(para(12), TOK, C)
    assert all(p.endswith(".") for p in pieces)
    last_sentence = ch.sentences(pieces[0])[-1]
    assert pieces[1].startswith(last_sentence)            # one sentence of overlap


def test_sliding_window_has_fixed_overlap():
    pieces = ch.split_sliding_window(para(20), TOK, C)
    a, b = pieces[0].split(), pieces[1].split()
    assert len(a) == C["target_tokens"] and a[-C["overlap_tokens"]:] == b[:C["overlap_tokens"]]


def test_short_text_is_one_chunk_and_tiny_tail_merges():
    assert ch.split_text("token_aware", "One short sentence here.", TOK, C) == ["One short sentence here."]
    pieces = ch.split_token_aware(para(4) + " tiny tail.", TOK, C)
    assert not pieces[-1].startswith("tiny")


def test_semantic_splits_at_topic_shift():
    text = para(4, tag="diet") + " " + para(4, tag="sleep")
    sents = ch.sentences(text)

    def embed(ss):                                        # two orthogonal "topics"
        return np.array([[1.0, 0.0] if s.lower().startswith("diet") else [0.0, 1.0] for s in ss])

    pieces = ch.split_semantic(text, TOK, {**C, "min_tokens": 5}, embed)
    assert len(pieces) == 2 and pieces[0].startswith("Diet") and pieces[1].startswith("Sleep")
    with pytest.raises(ValueError):
        ch.split_semantic(text, TOK, C, None)
    assert len(sents) == 8


def _cfg():
    return {"index": {"chunk": C, "tables": {"summary": "caption", "max_markdown_chars": 4000}}}


DOC = {"doc_id": "PMID:1", "title": "Millets and HbA1c", "abstract": "We randomised 240 adults. HbA1c fell.",
       "year": 2021, "design": "rct", "source_type": "primary", "topics": ["diabetes_prevention"], "status": "included",
       "pubmed_url": "https://pubmed.ncbi.nlm.nih.gov/1/"}
EXTRACTION = {"population": {"tags": ["prediabetes", "indian"]}, "outcomes": [{"name": "HbA1c", "type": "surrogate"}],
              "effect_direction": "benefit", "key_finding": "Millets lowered HbA1c."}
FULLTEXT = {"sections": [{"name": "methods", "title": "Methods", "text": para(10)},
                         {"name": "results", "title": "Results", "text": para(3)}],
            "tables": [{"label": "Table 1", "caption": "Baseline", "rows": [["Age", "45"], ["BMI", "26"]], "footer": ""}],
            "recommendations": [{"text": "Refer adults with prediabetes to lifestyle programs.", "grade": "A",
                                 "section": "Prevention"}]}


def test_build_doc_chunks_metadata_parents_and_kinds():
    chunks, parents = ch.build_doc_chunks(DOC, "token_aware", TOK, _cfg(), extraction=EXTRACTION,
                                          score={"quality_score_v2": 0.81, "superseded_by": None}, fulltext=FULLTEXT)
    kinds = [c.kind for c in chunks]
    assert kinds[0] == "abstract" and kinds.count("table") == 1 and kinds.count("recommendation") == 1
    assert kinds.count("section") >= 3
    ab = chunks[0]
    assert ab.embed_text.startswith("Design: rct | Year: 2021 | Population: indian, prediabetes | Key finding: Millets")
    assert ab.text.startswith("Millets and HbA1c") and "Design:" not in ab.text      # header only in embed_text
    m = ab.meta
    assert m["india_focused"] and m["quality_score"] == 0.81 and not m["superseded"] and m["outcome_types"] == ["surrogate"]
    pids = {p.parent_id for p in parents}
    assert all(c.parent_id in pids for c in chunks)
    sec = [c for c in chunks if c.kind == "section"]
    assert {c.parent_id for c in sec} == {"PMID:1#s0", "PMID:1#s1"}
    table = next(c for c in chunks if c.kind == "table")
    assert "| Age | 45 |" in table.text and table.embed_text.startswith("Millets and HbA1c | Table 1 Baseline")
    rec = next(c for c in chunks if c.kind == "recommendation")
    assert rec.text.endswith("(Grade: A)")
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_chunk_ids_differ_by_strategy_but_parents_do_not():
    a, pa = ch.build_doc_chunks(DOC, "token_aware", TOK, _cfg(), fulltext=FULLTEXT)
    b, pb = ch.build_doc_chunks(DOC, "recursive", TOK, _cfg(), fulltext=FULLTEXT)
    assert {c.chunk_id for c in a}.isdisjoint({c.chunk_id for c in b})
    assert [p.parent_id for p in pa] == [p.parent_id for p in pb]


def test_table_summary_used_for_embedding_when_given():
    chunks, _ = ch.build_doc_chunks(DOC, "token_aware", TOK, _cfg(), fulltext=FULLTEXT,
                                    table_summaries={"PMID:1#t0": "Baseline age and BMI were similar."})
    table = next(c for c in chunks if c.kind == "table")
    assert table.embed_text == "Millets and HbA1c | Baseline age and BMI were similar."


def test_embedding_cache(tmp_path):
    calls = []

    def fake(texts):
        calls.append(list(texts))
        return np.array([[len(t), 1.0] for t in texts], dtype=np.float32)

    e = Embedder("m", EmbeddingCache(tmp_path / "e.sqlite"), encode_fn=fake)
    v1 = e.encode(["aa", "b", "aa"])
    assert v1.shape == (3, 2) and calls == [["aa", "b"]]               # duplicates embedded once
    v2 = Embedder("m", EmbeddingCache(tmp_path / "e.sqlite"), encode_fn=fake).encode(["b", "ccc"])
    assert calls[-1] == ["ccc"] and np.allclose(v2[0], v1[1])
    other = Embedder("m2", EmbeddingCache(tmp_path / "e.sqlite"), encode_fn=fake)
    other.encode(["b"])
    assert calls[-1] == ["b"]                                          # cache is per model


def test_semantic_does_not_split_on_ties():
    text = para(6)
    pieces = ch.split_semantic(text, TOK, {**C, "min_tokens": 5}, lambda ss: np.ones((len(ss), 2)) / np.sqrt(2))
    assert len(pieces) == 2                              # only the token budget splits it (60 words, max 50)
