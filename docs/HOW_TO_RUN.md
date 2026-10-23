# How to run nutrition-rag

Every command below runs from the project folder (`/Users/iyer/Documents/dataset rag`).
Most source files are not run directly: each `nrag` command runs one or more of them, and
the tables show which. Commands marked 🌐 call the **shared Gemma server**, so run them
off-hours and keep concurrency low (`--concurrency 2`).

See [WHAT_DOES_WHAT.md](WHAT_DOES_WHAT.md) for what each part does.

---

## 0. Setup

### One-time (fresh machine only)

```bash
python3 -m venv .venv
```

```bash
.venv/bin/python -m pip install -e ".[dev,pdf,search,baseline,ragas]"
```

```bash
brew install colima docker docker-compose
```

Copy `.env.example` to `.env` and fill in:

| Variable | What it is |
|---|---|
| `LLM_REMOTE_ENDPOINTS` | Base URL of the Gemma (vLLM) server, without `/v1` |
| `POSTGRES_PASSWORD` | Any strong password for the local database |
| `NCBI_EMAIL`, `NCBI_API_KEY` | Optional. A key raises PubMed's limit from 3 to 10 requests/second |

`.env` is gitignored. Never commit it.

### Every session

Activate the environment so you can type `nrag` instead of `.venv/bin/nrag`:

```bash
source .venv/bin/activate
```

Start the database runtime and the AlloyDB Omni container. They're needed after every reboot,
for `index`, `search`, `ask` and `eval`:

```bash
colima start --cpu 4 --memory 6
```

```bash
docker compose up -d db
```

Check that everything works:

```bash
make test
```

> Keep the Mac **plugged in with the lid open** during long jobs: macOS sleep pauses them.

---

## 1. Ingest: build the corpus

| File | Run via |
|---|---|
| `ingest/pubmed.py`, `credibility.py`, `retractions.py`, `journals.py`, `pmc_fulltext.py`, `guidelines.py`, `pipeline.py` | `nrag ingest` |
| `ingest/pipeline.py` (`add_india_papers`) | `nrag add-india` |
| `ingest/guidelines.py` on its own | `nrag guidelines` |

Full corpus (about 15–30 min; rebuilds `data/corpus/`):

```bash
nrag ingest
```

Small test run:

```bash
nrag ingest --max-per-topic 200
```

Weekly refresh (new PubMed entries since a date):

```bash
nrag ingest --since 2026/10/15
```

Add N India-focused papers without a full re-ingest:

```bash
nrag add-india --target 500
```

Guideline PDFs: put them in `data/guidelines/` (file names are in `data/guidelines/registry.yaml`), then check them:

```bash
nrag guidelines
```

Outputs: `data/corpus/documents.jsonl`, `excluded.jsonl`, `summary.json`, `removals_log.jsonl`, `fulltext/`.

---

## 2. Extract 🌐: structured facts from every paper

| File | Run via |
|---|---|
| `extract/pico.py`, `schemas.py`, `prompts/pico_v1.md`, `runner.py`, `llm.py` | `nrag extract` |
| `extract/review.py` | `nrag extract-compare`, `nrag review-xlsx`, `nrag extract-accuracy` |

Extract all papers with Gemma (cached; only new or changed papers are sent):

```bash
nrag extract --sample all --concurrency 2
```

Blinded review spreadsheet (Gemma vs gpt-oss), the colour-banded Excel copy, and accuracy after you label it:

```bash
nrag extract-compare --system models-gemma-4-26B-A4B-it --system gpt-oss-20b__think-low --sample review
```

```bash
nrag review-xlsx
```

```bash
nrag extract-accuracy
```

Outputs: `data/interim/extractions/<model>.jsonl`, `eval/extraction_review.csv/.xlsx`, `eval/reports/extraction_accuracy.md`.

---

## 3. Score: quality and supersession

| File | Run via |
|---|---|
| `score/v2.py`, `score/runner.py` | `nrag score` |

```bash
nrag score
```

Outputs: `data/corpus/scores_v2.jsonl`, `eval/reports/score_v2.md` (spot-check the supersession pairs).

---

## 4. Index: chunks, embeddings, pgvector, BM25

| File | Run via |
|---|---|
| `index/tables.py` | `nrag summarize-tables` 🌐 |
| `index/chunk.py`, `embed.py`, `store.py`, `bm25.py`, `build.py` | `nrag index` |
| `retrieve/llamaindex_baseline.py` | `nrag llamaindex-build` |

Gemma summaries of full-text tables (run after new full texts arrive):

```bash
nrag summarize-tables --concurrency 1
```

Default index (token_aware strategy, HNSW):

```bash
nrag index
```

All five chunking strategies (several hours):

```bash
nrag index --all-strategies
```

One strategy, or another vector index type:

```bash
nrag index --strategy semantic
```

```bash
nrag index --index-type scann
```

Only papers that have an extraction (quick test):

```bash
nrag index --only-extracted
```

LlamaIndex baseline over the same chunks:

```bash
nrag llamaindex-build
```

Outputs: AlloyDB tables `documents`, `parents`, `chunks`; `data/index/bm25/<strategy>/`; `eval/reports/index_stats.md`.

---

## 5. Retrieve: search only (no answer)

| File | Run via |
|---|---|
| `retrieve/query_understanding.py` 🌐, `hybrid.py`, `rrf.py`, `rerank.py`, `pipeline.py` | `nrag search` |

```bash
nrag search "Does a low glycaemic index diet reduce gestational diabetes risk?"
```

Every stage with scores:

```bash
nrag search "Which Indian diets help prevent type 2 diabetes?" --trace
```

The LlamaIndex baseline instead:

```bash
nrag search "Does breakfast help weight loss?" --baseline
```

Stage logs: `data/runs/retrieval/`.

---

## 6. Answer 🌐: cited answers with safety checks

| File | Run via |
|---|---|
| `generate/answer.py`, `safety.py`, `prompts/answer_v1.md` | `nrag ask` |

```bash
nrag ask "Is it safe to take iron supplements during pregnancy?"
```

With citation-check and safety details:

```bash
nrag ask "Which Indian diets help prevent type 2 diabetes in adults with prediabetes?" --trace
```

---

## 7. Evaluate 🌐

| File | Run via |
|---|---|
| `eval/golden.py` | `nrag eval-pool`, `nrag eval-import-labels` |
| `eval/run.py`, `retrieval_metrics.py`, `answer_metrics.py`, `judge.py`, `ragas_eval.py`, `failures.py`, `report.py`, `commands.py` | `nrag eval` |
| `eval/ablate.py` | `nrag ablate` |

1. Pool candidate papers for each golden question, and have Gemma draft 0–3 relevance grades into `eval/relevance_labels.csv`:

```bash
nrag eval-pool --workers 2
```

2. Fill in the `your_grade` column (0 = not relevant … 3 = directly answers it). Then load your labels; they override Gemma's drafts:

```bash
nrag eval-import-labels
```

3. Run the evaluation. The report goes to `eval/reports/run_<name>.md`, raw results to `eval/runs/<name>/`:

```bash
nrag eval --ragas --label my_run
```

Retrieval only (fast, no answers):

```bash
nrag eval --no-answers --label retrieval_only
```

Dev or test half only:

```bash
nrag eval --split test --label test_only
```

4. Ablations (each component switched off; deltas with 95% confidence intervals):

```bash
nrag ablate
```

5. Compare two runs (folder names under `eval/runs/`):

```bash
nrag compare 20261023-151354_base my_run
```

6. Failure buckets after your corrections in that run's `failures.csv`:

```bash
nrag failures 20261023-151354_base
```

Human checks the evaluation relies on:

| File | What to fill in |
|---|---|
| `eval/relevance_labels.csv` | `your_grade` 0–3 per question–paper pair |
| `eval/judge_validation.csv` | `your_score` 0–2 per answer (validates the LLM judge) |
| `eval/runs/<run>/failures.csv` | `your_bucket` if the automatic failure cause is wrong |
| `eval/golden_set.jsonl` | review questions and reference answers; set `expert_verified: true` |

---

## 8. One-shot scripts

The chain after extraction: score, table summaries, index, LlamaIndex, pooling, label import, evaluation:

```bash
scripts/phase7_live.sh
```

Ablations, plus indexing and comparing the other four chunking strategies:

```bash
scripts/phase7_ablations.sh
```

Logs: `data/runs/phase7_live.log`, `data/runs/phase7_ablations.log`.

---

## 9. Tests

Everything (offline; needs the database container for the DB tests, otherwise they're skipped):

```bash
make test
```

One part:

```bash
pytest tests/test_retrieval.py
```

| Area | Test files |
|---|---|
| Ingest | `test_pubmed_parse.py`, `test_pubmed_rules.py`, `test_search.py`, `test_credibility.py`, `test_retractions_journals.py`, `test_pmc_fulltext.py`, `test_guidelines.py`, `test_pipeline.py`, `test_parity.py` |
| Extract | `test_extract.py`, `test_review.py` |
| Score | `test_score_v2.py` |
| Index | `test_index_chunk.py`, `test_index_store.py` |
| Retrieve | `test_retrieval.py` |
| Answer + safety | `test_generate.py` |
| Evaluate | `test_eval_golden_metrics.py`, `test_eval_runs.py` |
| Infra | `test_net.py`, `test_cli_config.py` |

---

## Typical order from scratch

`ingest` → `extract` 🌐 → `score` → `summarize-tables` 🌐 → `index` → `llamaindex-build`
→ `search` / `ask` 🌐 → `eval-pool` 🌐 → *(you label)* → `eval-import-labels` → `eval --ragas` 🌐 → `ablate`

Everything is cached (HTTP responses, LLM outputs, embeddings), so re-running a command only redoes what changed.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `database unavailable` / connection refused | `colima start --cpu 4 --memory 6`, then `docker compose up -d db` |
| `no endpoints for LLM provider 'remote'` | set `LLM_REMOTE_ENDPOINTS` in `.env` |
| A run "aborted after repeated model-server errors" | the server is down or overloaded; rerun later (finished items are cached) |
| A long job made no progress overnight | the Mac slept; keep it plugged in with the lid open |
| `Note: NCBI_EMAIL not set` | harmless; PubMed works without it at 3 requests/second |
