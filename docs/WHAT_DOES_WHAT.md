# nutrition-rag: what does what

An evidence-ranked research assistant for **diabetes prevention, weight management, pregnancy
nutrition and gestational diabetes**, with Indian and South Asian users in mind. It answers only from
credible published research, cites every claim, says plainly how strong the evidence is, and says
when evidence is weak, mixed or missing. Every component is measured.

See [HOW_TO_RUN.md](HOW_TO_RUN.md) for the command behind each part.

---

## The whole flow

```
PubMed + PubMed Central + guideline PDFs
        │  1. INGEST    fetch, deduplicate, retraction + journal checks, credibility filter
        ▼
data/corpus/documents.jsonl        (papers that passed; full texts only under open licences)
        │  2. EXTRACT   Gemma reads each abstract: population, intervention, outcomes, effect, bias
        │  3. SCORE     quality score per paper + "replaced by a newer version" flag
        │  4. INDEX     chunks -> bge-m3 vectors -> AlloyDB (pgvector) + BM25 keyword index
        ▼
question ──► 5. RETRIEVE   understand -> hybrid search -> rerank -> quality-aware order -> confidence check
        │  6. ANSWER     Gemma writes a cited answer -> citation check -> safety check
        ▼
answer with citations, evidence strength and caveats
        │  7. EVALUATE   golden questions -> metrics, ablations, failure analysis, report
```

---

## 1. Ingest: building the corpus (`src/nrag/ingest/`)

| File | What it does |
|---|---|
| `pubmed.py` | Searches PubMed per topic, split by year (and by date inside a year) to get around PubMed's 10,000-result limit. Parses records; classifies study design (publication type → MeSH → abstract text, ignoring background sentences); extracts a rough sample size; computes the first quality score (design × size × recency) |
| `credibility.py` | The "extremely credible research" policy: guidelines, meta-analyses, systematic reviews, RCTs (50+), prospective cohorts (1,000+), from 2013, in MEDLINE journals, no expressions of concern, no protocols or preprints. Unknown sample size or a design found only in text makes a paper **provisional**. **India-focused papers** get relaxed rules (cohort, case-control and cross-sectional allowed; lower minimum sizes) and are flagged `india_relaxed` |
| `retractions.py` | Checks every paper against the Retraction Watch database (via Crossref) by DOI and PMID on every run; reinstatements are respected; retracted papers leave the corpus and are logged |
| `journals.py` | Checks whether the journal is currently indexed in MEDLINE (NLM Catalog), not just the paper's own indexing status |
| `pmc_fulltext.py` | Downloads full text from PubMed Central **only** for open licences (CC BY, CC BY-SA, CC0); splits it into normalised sections and keeps tables whole |
| `guidelines.py` | Registry of clinical guidelines (ICMR-NIN, ADA, WHO, IOM, ACOG, RSSDI) plus a PDF parser that splits by headings, keeps tables and extracts graded recommendations |
| `pipeline.py` | Runs all of the above. Rebuilds the corpus from scratch each run from stored records (deterministic). Handles the **India boost** (extra India / South Asia searches), the **Indian-staples searches** (millets, pulses, rice, ghee…) and `add-india` (add N India papers without re-ingesting) |

## 2. Extract: structured facts from each paper (`src/nrag/extract/`)

| File | What it does |
|---|---|
| `schemas.py` | The form the model fills in: design, population tags, intervention, comparison, outcomes (hard or surrogate), effect direction, key finding, sample size, duration, country, risk-of-bias signals, industry funding, quotes, confidence |
| `prompts/pico_v1.md` | The versioned extraction prompt |
| `pico.py` | Sends each abstract to the model with a fill-in JSON template; validates against the form; retries with the error message; checks that quoted text really appears in the abstract; caches by input hash |
| `review.py` | Stratified review sample, blinded A/B spreadsheet, and per-field accuracy scoring with confidence intervals and a rules-only baseline |
| `runner.py` | Command glue: which model, which papers, where results go |

## 3. Score: how much to trust each paper (`src/nrag/score/`)

| File | What it does |
|---|---|
| `v2.py` | Quality score v2: the first score adjusted by risk-of-bias signals, industry funding, extraction confidence and the extracted sample size. **Supersession**: marks an older review as replaced when a newer version has a near-identical title (or says it is an update) |
| `runner.py` | Writes the scores and a report with distributions and supersession pairs to spot-check |

## 4. Index: making the corpus searchable (`src/nrag/index/`)

| File | What it does |
|---|---|
| `chunk.py` | Splits documents into chunks: one per abstract (with a design/year/population/key-finding header used only for embedding); full-text sections with one of **five strategies**: token_aware, recursive, sliding_window, semantic, hierarchical; whole tables; one chunk per guideline recommendation. Every chunk carries filter metadata and its **parent section**, so search can return the whole section |
| `tables.py` | The model writes a one-paragraph summary of each table, used as the table's search text |
| `embed.py` | Turns text into vectors with **bge-m3**, locally on the Mac's GPU, with a resumable cache |
| `store.py` | AlloyDB Omni (Postgres + pgvector) schema: `documents`, `parents`, `chunks` with filter columns (topic, population, design, year, quality, superseded, India-focused); **HNSW / IVFFlat / ScaNN** indexes per strategy; exact search for recall checks |
| `bm25.py` | Keyword index (BM25) over the same chunks |
| `build.py` | Runs the whole index build and writes statistics |

## 5. Retrieve: finding the best evidence (`src/nrag/retrieve/`)

| File | What it does |
|---|---|
| `query_understanding.py` | The model identifies the population, topic and intent, decides whether research can answer the question at all, and rewrites it as a precise search query. The original question is always kept |
| `hybrid.py` | BM25 and vector search, run on both the original and the rewritten question |
| `rrf.py` | Merges the result lists (reciprocal rank fusion); plain concatenation is kept for comparison |
| `rerank.py` | **bge-reranker** cross-encoder re-scores the top 50 candidates, locally |
| `pipeline.py` | Drops superseded papers; boosts papers matching the question's population; final ranking = relevance × α + quality × (1 − α), plus a guideline bonus; at most 2 chunks per paper; if evidence is weak, one retry and then "insufficient evidence"; logs every stage for failure analysis. Every stage can be switched off |
| `llamaindex_baseline.py` | Standard LlamaIndex hybrid retriever over the same chunks and embeddings, for comparison |

## 6. Answer and safety (`src/nrag/generate/`)

| File | What it does |
|---|---|
| `prompts/answer_v1.md` | The versioned answer prompt: direct answer first, cite every claim, prefer guidelines, explain conflicts, no outside knowledge |
| `answer.py` | Packs the retrieved sources; the model writes a structured answer (direct answer, claims with citations, evidence strength, conflicts). Two citation checks: each id must be a retrieved source, and a second model pass confirms each source **supports** its claim (unsupported claims are dropped). Adds a note when the cited evidence is from outside India and South Asia; won't allow "strong" without strong study designs |
| `safety.py` | Before generation: **disordered-eating signals** get a supportive reply with the Tele-MANAS helpline and no diet advice; **pregnancy / breastfeeding risk topics** (supplements, medicines, alcohol, high-mercury fish, food safety, gestational diabetes) give the guideline position plus a doctor/midwife note; **personal dosing or medicine changes** are refused. After generation: individual amounts are replaced with "[amount withheld]". One disclaimer per answer |

## 7. Evaluate: proving it works (`src/nrag/eval/`)

| File | What it does |
|---|---|
| `golden.py` | The golden set (`eval/golden_set.jsonl`, 60 questions: factual, comparison, safety, conflicting, unanswerable, India-specific); a fixed dev/test split; candidate pooling from four systems; model-drafted relevance grades that humans correct |
| `retrieval_metrics.py` | Recall@k, precision@k, MRR, nDCG@10, by topic and type, with bootstrap confidence intervals |
| `answer_metrics.py` | Abstention accuracy, safety compliance, faithfulness, citation precision/recall, required phrases, ROUGE-L, similarity to the reference answer |
| `judge.py` | The model grades correctness against the reference answer with a fixed rubric; validation spreadsheet; agreement (accuracy, Cohen's kappa) |
| `ragas_eval.py` | RAGAS metrics (faithfulness, answer relevancy, context precision, context recall) using the same model and bge-m3 |
| `run.py` | Runs the evaluation, saves runs, compares runs, and tunes α and the confidence threshold on dev (reported on test) offline from the logs |
| `ablate.py` | Re-runs retrieval with each component switched off, with paired confidence intervals |
| `failures.py` | Sorts each failure into one of 9 causes: safety miss, wrong abstention, query misread, not in corpus, not retrieved, filtered out, ranked low, ignored by the model, unfaithful |
| `report.py` | The Markdown report: metrics, ablations, tuning, failures, "what to fix next" |
| `commands.py` | Shared models and connections plus the command glue |

---

## Infrastructure

| Part | What it does |
|---|---|
| `config.yaml` | Every setting: queries, rules, thresholds, models, weights. Nothing is hard-coded |
| `.env` | Secrets and the model server address. Gitignored |
| `src/nrag/config.py` | Loads the config and environment variables |
| `src/nrag/net.py` | Rate-limited, retrying, cached web requests (PubMed, Crossref, NLM) |
| `src/nrag/llm.py` | Talks to Ollama or any OpenAI-compatible server (the Gemma server); retries; spreads work across endpoints |
| `src/nrag/cli.py` | The `nrag` command-line tool |
| `docker-compose.yml`, `Dockerfile` | AlloyDB Omni database (pinned image) and the app container |
| `.github/workflows/ci.yml` | Runs the tests on every push |
| `scripts/` | One-shot chains for long runs |
| `tests/` | One test file per area; synthetic fixtures (fake PMIDs) |
| `legacy/ingest_pubmed.py` | The original ingestion script, kept for the parity tests |

## Which model does what

| Model | Where it runs | Used for |
|---|---|---|
| **Gemma 4 26B-A4B** | user's vLLM server (shared) | extraction, query understanding, answers, citation checks, table summaries, relevance drafts, correctness judge, RAGAS |
| **bge-m3** | Mac GPU | embeddings for search, semantic chunking, answer similarity, RAGAS |
| **bge-reranker-v2-m3** | Mac GPU | reranking the top 50 candidates |
| gpt-oss 20B (Ollama) | Mac | earlier extraction comparison only |

## Where the data lives

| Folder | Contents |
|---|---|
| `data/interim/` | Every fetched PubMed record (the store), extractions, table summaries |
| `data/corpus/` | `documents.jsonl`, `excluded.jsonl`, `scores_v2.jsonl`, full texts, summary, removals log |
| `data/index/` | BM25 indexes and the index manifest (vectors live in AlloyDB) |
| `data/cache/` | Cached HTTP responses, model outputs and embeddings |
| `data/runs/` | Retrieval stage logs and long-run logs |
| `eval/` | Golden set, label spreadsheets, runs, reports |
