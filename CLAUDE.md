# nutrition-rag

Evidence-ranked RAG over nutrition research for diabetes prevention, weight
management and maternal nutrition, with Indian / South Asian users in mind.
The goal is ML discipline applied to RAG: a measured corpus, an eval set from
the start, ablations and failure analysis. Every component must justify itself
with numbers.

## Status

Built in phases; stop after each phase for review and wait for the go-ahead.
- Phase 0 (setup + port of the legacy ingester): done.
- Phase 1 (dataset ingestion): done. Live sample run: `--max-per-topic 200`.
- Phase 2 (LLM PICO extraction): done on the 606-doc sample with Gemma 4 (606 ok, prompt pico_v1 with the fill-in template output mode; now `extract.active_model`; gpt-oss dropped). The Gemma-vs-gpt-oss review CSV was not labelled, so no field-level accuracy numbers exist yet (`nrag extract-accuracy` after labelling). Two providers: `local` (Ollama on the Mac: gpt-oss 20B, Qwen3.8 27B GGUF)
  and `remote` (the user's OpenAI-compatible vLLM server running Gemma 4 26B-A4B; base URL in `.env` as
  `LLM_REMOTE_ENDPOINTS`, never committed). No laptop, no paid APIs.
- Phase 3 (scoring v2 + supersession): built, `nrag score` (config `score_v2`, `src/nrag/score/`). Writes `data/corpus/scores_v2.jsonl` and `eval/reports/score_v2.md`. Not yet validated against labels: multipliers are priors; supersession (4 pairs on the sample) needs a manual spot-check.
- Full corpus ingested 2026-10-22: 7,127 documents (4,774 included, 2,353 provisional), 2,207 open-licence full
  texts, India boost +9% (the full India literature is too small for +30% under the credibility rules).
  Gemma extraction of the full corpus runs off-hours (scheduled 19:00 IST, 2 concurrent).
- Phase 4 (chunking + indexing): built, `nrag index`. AlloyDB Omni in Docker via Colima
  (`colima start --cpu 4 --memory 6`; `docker compose up -d db`), bge-m3 on MPS (~14 chunks/s at ~290 tokens).
- Corpus 2026-10-22 (later): +500 India papers via `nrag add-india` and relaxed India credibility rules
  (`credibility.india_relaxed`, flagged `india_relaxed`, 776 docs) -> 7,956 documents; Indian-staples search
  source `india_foods` added (runs on the next full ingest). Table summaries by Gemma (`nrag summarize-tables`).
- Phase 5 (retrieval): built, `nrag search "q" [--trace] [--baseline]`. Query understanding (Gemma) -> BM25 +
  dense on original + rewritten -> RRF -> filters/population boost -> bge-reranker-v2-m3 (~11 s / 50 pairs on MPS)
  -> alpha*rerank + (1-alpha)*quality (+guideline bonus) -> per-doc cap -> confidence retry/abstain; stage logs in
  `data/runs/retrieval/`. LlamaIndex baseline (`nrag llamaindex-build`) on the same chunks/embeddings.
  Known: reranker probabilities saturate near 1.0 (quality then decides order) and the population boost can
  favour GDM-management papers for GDM-prevention questions; both for Phase 7 tuning.
- Phase 6 (answers + safety): built, `nrag ask "q" [--trace]`. Structured Gemma answer (direct answer, claims with
  citations, strength, conflicts, abstain) -> citation checks (ids must be retrieved; second LLM pass verifies
  support on the SAME source text the generator saw) -> deterministic caveats (strength capped without strong
  designs; non-Indian-evidence note from metadata) -> safety post-check. Safety (`config.safety`): disordered
  eating -> supportive reply only (Tele-MANAS), no retrieval/LLM; individual advice / pregnancy risk topics ->
  restricted: guideline position, amounts redacted to "[amount withheld]" in every field, clinician note.
- No GCP account: Phase 10 is a local deployment (Docker compose, CI, local monitoring); the embedding
  comparison uses local models only.

## Commands

```bash
make setup                                  # .venv + package with dev extras (add pdf: pip install -e ".[dev,pdf]")
make test                                   # offline pytest (-m "not network")
.venv/bin/nrag ingest --max-per-topic 200   # sample run (~2 min cold, seconds when cached)
.venv/bin/nrag ingest                       # full corpus
.venv/bin/nrag ingest --since 2026/10/01    # incremental refresh (entry date)
.venv/bin/nrag guidelines                   # registry status; parses PDFs present
```

Phase 4 (needs Colima + the db container running; password in `.env` as POSTGRES_PASSWORD):

```bash
colima start --cpu 4 --memory 6 && docker compose up -d db
.venv/bin/nrag index --only-extracted          # default strategy on docs with an LLM extraction
.venv/bin/nrag index --all-strategies          # all 5 chunking strategies (ablation), full corpus
.venv/bin/nrag index --index-type scann        # hnsw (default) | ivfflat | scann
```

Phase 2 (needs `ollama serve` running and the models pulled):

```bash
.venv/bin/nrag extract --model gpt-oss:20b --sample screen       # 30 stratified abstracts (cached per doc)
.venv/bin/nrag extract --model qwen3.8:27b --sample screen
.venv/bin/nrag extract-compare --system gpt-oss-20b__think-low --system qwen3-8-27b   # blinded A/B CSV
.venv/bin/nrag extract-accuracy                                 # after the CSV is filled in
```

`ingest` flags: `--topic NAME` (repeatable), `--design-filter/--no-design-filter`,
`--no-fulltext`, `--no-guidelines`, `--config PATH`.
NCBI_EMAIL / NCBI_API_KEY are optional (without a key: 3 req/s).

## Architecture (src/nrag)

- `config.yaml`: every tunable number: queries, design rules, credibility policy, licences, TTLs, grade patterns. Never hardcode these.
- `config.py`: `load_config()` (honours `NRAG_CONFIG`), `resolve_path()`, `env()` for secrets.
- `net.py`: `CachedHttp`, a rate-limited, retrying client with an on-disk cache (`data/cache/{ncbi,web}`). **All network calls go through it.** Identity/secret params are excluded from cache keys.
- `ingest/pubmed.py`: parsing, `classify_design` (pubtype → MeSH → scoped text), `extract_sample_size`, `score` (v1: design × size × recency), `PubMedClient.search` (year slices, recursive date split over the 10k cap, proportional sampling with `limit`), `fetch` (POSTed ID batches).
- `ingest/credibility.py`: `assess()` → included / provisional / excluded with a reason.
- `ingest/retractions.py`: Retraction Watch (Crossref's GitLab CSV) index by DOI and PMID; reinstatements reverse retractions.
- `ingest/journals.py`: NLM Catalog `currentlyindexed` by NLM ID (UIDs have no leading zeros).
- `ingest/pmc_fulltext.py`: JATS parse with a **licence gate** (efetch returns full text for non-open licences too). Normalised sections; tables kept whole.
- `ingest/guidelines.py`: registry (pydantic, validated against topics and the population vocabulary) + pymupdf/pdfplumber parser + graded recommendations.
- `ingest/pipeline.py`: orchestration. The corpus is **rebuilt from scratch every run** from the raw store (`data/interim/pubmed_records.jsonl`, every stored record refetched at most every `efetch_ttl_days`) + config + external checks. Docs that drop out are appended to `removals_log.jsonl`.
- `llm.py`: `LLMClient` (backends `ollama` native /api/chat, or `openai_compat` /v1/chat/completions), `EndpointPool`; `client_from_config(llm_cfg, provider)`. Ollama `done:false` empty replies (runner crash) are retried as errors.
- Measured lessons (in config comments): constrained decoding biased gpt-oss answers, so extraction uses prompted JSON (`extract.output_mode`); Qwen3.8 27B MLX build OOMs on 24 GB, GGUF build runs 90/10 GPU/CPU at ~3 min/abstract; a smaller num_ctx crashed the GPU; runs abort after `llm.max_consecutive_failures`.
- `extract/schemas.py`: strict Pydantic `Extraction` (vocabularies must match config; tested). `extract/prompts/pico_v1.md`: versioned prompt.
- `extract/pico.py`: per-document extraction cached by input hash (model, think, prompt, schema, text), retries with validation feedback, quote verification against the abstract.
- `extract/review.py`: stratified sample (screen 30 ⊂ review 100), blinded A/B review CSV + key, scoring vs gold (accuracy, kappa, tag P/R/F1, numeric ±5%, hallucinated-"yes" rate, bootstrap CI, McNemar), rules-only baseline, pre-registered decision rule.
- `extract/runner.py`: CLI glue; results in `data/interim/extractions/<system>.jsonl`, failures in `failures.jsonl`.
- `index/chunk.py`: abstract chunks (metadata header in `embed_text` only), 5 section strategies (token_aware,
  recursive, sliding_window, semantic, hierarchical), whole tables, guideline recommendations; every chunk has
  filter metadata and a `parent_id` (section) for parent-document retrieval. Token budgets use bge-m3's tokenizer.
- `index/embed.py`: `Embedder` with a SQLite cache keyed by (model, text); `index/bm25.py`: bm25s per strategy.
- `index/store.py`: pgvector schema (`documents`, `parents`, `chunks` with a generated `tsv`); one partial ANN
  index per strategy (HNSW / IVFFlat / ScaNN); `vector_search(..., exact=True)` for recall baselines.
- `index/build.py`: orchestration; writes `data/index/manifest.json` and `eval/reports/index_stats.md`.
  DB tests use a throwaway `nrag_test` database and skip when the container isn't running.
- `retrieve/`: `query_understanding.py` (Pydantic `QueryAnalysis`, cached), `hybrid.py` (BM25 / dense / metadata /
  parents; sets `hnsw.ef_search >= k`), `rrf.py` (RRF + concat ablation), `rerank.py` (CrossEncoder),
  `pipeline.py` (`Retriever`, every stage switchable via config `retrieval`), `llamaindex_baseline.py`.
- `generate/`: `safety.py` (pre/post rules), `answer.py` (`Answerer`, `pack_sources`, citation verification,
  `render`), `prompts/answer_v1.md`. LLM calls cached in `data/cache/llm/answers/`.
- Test hygiene: commit only when `make test` is green (a grep-and-commit chain once committed a red suite);
  fake embeddings use crc32, not salted `hash()`.
- `cli.py`: typer app; later-phase commands are stubs that exit with code 2.

Outputs (`data/corpus/`, gitignored): `documents.jsonl` (included + provisional, sorted by quality),
`excluded.jsonl` (reason per record), `fulltext/{PMCID}.json`, `fulltext/guideline_{id}.json`,
`summary.json`, `removals_log.jsonl`. Document IDs: `PMID:<n>`, `GUIDELINE:<id>`.

## Credibility policy ("extremely credible research post 2012")

Confirmed defaults, all in `config.yaml` `credibility`: guidelines, meta-analyses, systematic
reviews, RCTs (n ≥ 50) and prospective cohorts (n ≥ 1000); published 2013+; MEDLINE journals;
no expressions of concern; no protocols or preprints. Registry guidelines are exempt from the
date cutoff while current (IOM 2009). Unknown sample size or a text-only design match makes a
record **provisional**; Phase 2 extraction resolves these.

## India boost

`pubmed.india_boost`: extra per-topic searches (India/South Asia terms AND diet terms AND topic outcomes),
because India-specific evidence is ~3% of the main results. The user asked for ~30% more India-focused
research: in sample runs `fraction` = 0.45 of each topic's quota is fetched, and about half passes the
credibility rules, giving +31% documents (200/topic sample: 145 India-boost docs on 461 main). In full runs
all India hits are added; the full India literature is too small to reach +30% of the whole corpus.
Records carry `search_sources` (main / india_boost).

## Conventions

- Secrets only via env / `.env` (gitignored; see `.env.example`). Never print them.
- Cache every network and LLM call; responses are keyed by their exact inputs.
- Records are flagged or excluded with a reason, never silently dropped.
- Every module has pytest tests; a phase isn't done until `make test` passes.
- Don't invent data or metrics. If something can't run here, say so and give the command.
- Fixtures are synthetic (fake PMIDs 900000xx, PMC9000001/2); regenerate PubMed XML with `tests/fixtures/pubmed/make_fixtures.py`.
- Commit at the end of each phase.

## Changes from the legacy script (tests/test_parity.py pins the exact diffs)

1. Searches are split by year (and by date within a year over the 10k cap); totals come from one full-range count because papers can match two year slices.
2. Retraction/EoC: Retraction Watch + PubMed, rechecked for the whole store every run; retraction propagates to DOI duplicates.
3. MEDLINE status is journal-level (NLM Catalog) or citation-level, not citation status alone.
4. Design text rules ignore background sections and prior-work sentences.
5. Stored records are refetched (TTL), so MeSH/EoC/retraction updates are picked up.
6. `Multicenter Study` is not a design; `Observational Study` defers to MeSH; `prospective_cohort` is separate; case-control and cross-sectional MeSH win over generic cohort.
7. Sample sizes 1900–2100 are skipped only in a date-like context; excluded.jsonl is rewritten (not appended).

## Known limitations / open items

- Remote Gemma in prompt mode echoes the JSON schema (`$defs`) on ~1/3 of first attempts (fixed by the retry,
  but it adds load to a shared production server). Candidate fix: a filled-in JSON template instead of the raw
  schema. Fixed by `extract.output_mode: template` for Gemma (first-attempt validity 86% -> 98%). Remote runs only off-hours unless the user says otherwise.

- `humans[mh]` in the base query means not-yet-MeSH-indexed records are never returned.
- `--max-per-topic` samples the newest papers within each year, so samples under-represent early-year papers (2013 is missing from the 200/topic sample).
- DOAJ check not implemented (moot while MEDLINE is required). Crossref per-DOI API not used; the Retraction Watch CSV is Crossref's distribution.
- ACOG registry entry needs the user to choose which document; no guideline PDFs downloaded yet.
