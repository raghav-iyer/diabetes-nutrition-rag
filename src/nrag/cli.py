"""`nrag` command-line interface."""

from __future__ import annotations

import json
from typing import List, Optional

import typer

from nrag.config import env, load_config, resolve_path

app = typer.Typer(help="Evidence-ranked nutrition research RAG.", no_args_is_help=True,
                  add_completion=False)

ConfigOpt = typer.Option(None, "--config", help="Path to config.yaml (default: repo root).")


def _clients(cfg):
    from nrag.ingest.pipeline import Clients
    from nrag.ingest.pubmed import PubMedClient
    from nrag.net import CachedHttp

    email, api_key = env("NCBI_EMAIL"), env("NCBI_API_KEY")
    ncbi = cfg["ncbi"]
    cache = resolve_path(cfg["paths"]["cache_dir"])
    ncbi_http = CachedHttp(cache / "ncbi", delay_s=ncbi["delay_with_key_s"] if api_key else ncbi["delay_no_key_s"],
                           timeout_s=ncbi["timeout_s"], retries=ncbi["retries"])
    web_http = CachedHttp(cache / "web", delay_s=1.0, timeout_s=300, retries=ncbi["retries"])
    params = {"tool": ncbi["tool"], **({"email": email} if email else {}), **({"api_key": api_key} if api_key else {})}
    pubmed = PubMedClient(ncbi_http, ncbi, cfg["cache"], email, api_key, sort=cfg["pubmed"]["sort"])
    if not email:
        typer.echo("Note: NCBI_EMAIL not set; requests are sent with `tool` only (allowed, 3 req/s).", err=True)
    return Clients(pubmed=pubmed, ncbi=ncbi_http, web=web_http, ncbi_params=params)


@app.command()
def ingest(
    since: Optional[str] = typer.Option(None, help="Incremental: only records entered since YYYY/MM/DD."),
    max_per_topic: Optional[int] = typer.Option(None, help="Sample N records per topic, spread across years (testing)."),
    topics: Optional[List[str]] = typer.Option(None, "--topic", help="Search only these topics (repeatable)."),
    design_filter: Optional[bool] = typer.Option(None, "--design-filter/--no-design-filter",
                                                 help="Override config pubmed.design_filter."),
    fulltext: bool = typer.Option(True, help="Fetch PMC open-licence full text."),
    guidelines: bool = typer.Option(True, help="Parse guideline PDFs from the registry."),
    config: Optional[str] = ConfigOpt,
):
    """Build the corpus: PubMed search + refresh, retraction and journal checks,
    credibility filter, PMC full text, guidelines. Writes data/corpus/."""
    from nrag.ingest import pipeline

    cfg = load_config(config)
    unknown = set(topics or []) - set(cfg["pubmed"]["topics"])
    if unknown:
        raise typer.BadParameter(f"unknown topic(s): {', '.join(sorted(unknown))}")
    clients = _clients(cfg)
    summary = pipeline.run(cfg, clients, since=since, max_per_topic=max_per_topic, topics=topics,
                           design_filter=design_filter, fulltext=fulltext, guidelines=guidelines)
    pipeline.print_summary(summary)
    for name, http in (("NCBI", clients.ncbi), ("web", clients.web)):
        typer.echo(f"{name} cache: {http.stats['hits']} hits, {http.stats['misses']} misses")


@app.command("add-india")
def add_india(
    target: int = typer.Option(500, help="How many new India-focused documents to add."),
    design_filter: bool = typer.Option(False, "--design-filter/--no-design-filter",
                                       help="Restrict the India searches to strong designs."),
    config: Optional[str] = ConfigOpt,
):
    """Add N new India-focused papers (India searches only), then rebuild the corpus from the store."""
    from nrag.ingest import pipeline

    cfg = load_config(config)
    clients = _clients(cfg)
    added = pipeline.add_india_papers(cfg, clients, target, design_filter)
    typer.echo(f"Added: {added}")
    summary = pipeline.run(cfg, clients, search=False)
    pipeline.print_summary(summary)


@app.command("guidelines")
def guidelines_cmd(config: Optional[str] = ConfigOpt):
    """Show the guidelines registry and parse any PDFs present (no corpus write)."""
    from nrag.ingest import guidelines as gl

    cfg = load_config(config)
    for e in gl.load_registry(cfg):
        parsed = gl.parse_entry(e, cfg)
        if parsed is None:
            where = e.file_glob or e.file
            typer.echo(f"  [missing] {e.id:<22} {e.year}  put the PDF at {where}")
        else:
            typer.echo(f"  [parsed]  {e.id:<22} {e.year}  {len(parsed['sections'])} sections, "
                       f"{len(parsed['tables'])} tables, {len(parsed['recommendations'])} graded recommendations")


def _not_yet(phase: int):
    typer.echo(f"Not implemented yet (Phase {phase}).", err=True)
    raise typer.Exit(code=2)


@app.command()
def extract(
    model: Optional[str] = typer.Option(None, help="Model name (default: LLM_MODEL or extract.active_model)."),
    think: Optional[str] = typer.Option(None, help="Reasoning: low|medium|high (gpt-oss), true|false (qwen)."),
    sample: str = typer.Option("all", help="screen (30) | review (100) | all documents."),
    limit: Optional[int] = typer.Option(None, help="Only the first N documents of the selection."),
    cache: bool = typer.Option(True, help="Reuse cached extractions."),
    concurrency: Optional[int] = typer.Option(None, help="Concurrent requests per endpoint (overrides config)."),
    config: Optional[str] = ConfigOpt,
):
    """LLM PICO / evidence-field extraction (local Ollama or a remote OpenAI-compatible server)."""
    from nrag.extract import runner

    cfg = load_config(config)
    model = model or runner.active_model(cfg)
    if concurrency:
        cfg["llm"]["providers"][runner.provider_for(cfg, model)]["concurrency_per_endpoint"] = concurrency
    summary = runner.run(cfg, model, runner.parse_think(think), sample, limit, cache)
    typer.echo("\n=== Extraction summary ===")
    for k, v in summary.items():
        typer.echo(f"{k}: {v}")


@app.command("extract-compare")
def extract_compare(
    systems: List[str] = typer.Option(..., "--system", help="Two (or one) result names from data/interim/extractions."),
    sample: str = typer.Option("screen", help="screen (30) | review (100)."),
    force: bool = typer.Option(False, help="Overwrite a review CSV that already has verdicts."),
    config: Optional[str] = ConfigOpt,
):
    """Write the blinded A/B review spreadsheet (eval/extraction_review.csv)."""
    from nrag.extract import runner

    path = runner.compare(load_config(config), systems, sample, force)
    typer.echo(f"Wrote {path}. Fill `verdict` (A, B, both, neither) and `correct_value` when neither is right.")


@app.command("review-xlsx")
def review_xlsx(config: Optional[str] = ConfigOpt):
    """Number the papers in the review CSV and write a colour-banded .xlsx copy to label in."""
    from nrag.config import resolve_path
    from nrag.extract import review

    cfg = load_config(config)
    csv_path = resolve_path(cfg["extract"]["review_csv"])
    n = review.add_paper_numbers(csv_path)
    xlsx = csv_path.with_suffix(".xlsx")
    review.write_review_xlsx(csv_path, xlsx)
    typer.echo(f"{n} papers numbered. Label {xlsx}, then File > Save As > CSV over {csv_path}.")


@app.command("extract-accuracy")
def extract_accuracy(config: Optional[str] = ConfigOpt):
    """Field-level accuracy, model comparison and decision from the filled review CSV."""
    from nrag.extract import runner

    result, report = runner.accuracy(load_config(config))
    typer.echo(report.read_text())
    typer.echo(f"Saved {report} (+ .json)")


@app.command()
def score(model: Optional[str] = typer.Option(None, help="Extraction model (default: extract.active_model)."),
          config: Optional[str] = ConfigOpt):
    """Quality scoring v2 (extraction-aware) and supersession (Phase 3)."""
    from nrag.score import runner

    out, report = runner.run(load_config(config), model)
    typer.echo(report.read_text())
    typer.echo(f"Saved {out} and {report}")


@app.command("summarize-tables")
def summarize_tables(
    concurrency: int = typer.Option(1, help="Concurrent requests (the remote server is shared)."),
    limit: Optional[int] = typer.Option(None, help="Only the first N tables (testing)."),
    config: Optional[str] = ConfigOpt,
):
    """One-paragraph LLM summaries of every full-text table (embedding text for table chunks)."""
    from nrag.extract import runner
    from nrag.index import build, tables
    from nrag.llm import EndpointPool, client_from_config

    cfg = load_config(config)
    model = runner.active_model(cfg)
    provider = runner.provider_for(cfg, model)
    inputs = build.load_inputs(cfg)
    items = tables.table_inputs(inputs["docs"], inputs["fulltext"], cfg["index"]["tables"]["max_markdown_chars"])
    items = items[:limit] if limit else items
    typer.echo(f"Summarising {len(items)} tables with {model} via {provider} ({concurrency} concurrent)")
    stats = tables.summarize(client_from_config(cfg["llm"], provider),
                             EndpointPool(runner.endpoints(cfg, provider), concurrency), model, items,
                             tables.output_path(cfg, model))
    typer.echo(f"Done: {stats}")


@app.command()
def index(
    strategy: Optional[List[str]] = typer.Option(None, "--strategy", help="Chunking strategy (repeatable); default from config."),
    all_strategies: bool = typer.Option(False, "--all-strategies", help="Build every chunking strategy."),
    index_type: Optional[str] = typer.Option(None, help="hnsw | ivfflat | scann (default from config)."),
    only_extracted: bool = typer.Option(False, help="Index only documents with an LLM extraction (the sample)."),
    config: Optional[str] = ConfigOpt,
):
    """Chunk documents, embed them, and build pgvector (AlloyDB Omni) + BM25 indexes (Phase 4)."""
    from nrag.index import build

    cfg = load_config(config)
    strategies = list(cfg["index"]["strategies"]) if all_strategies else (strategy or None)
    report = build.run(cfg, strategies, index_type, only_extracted)
    typer.echo(build.render_report(report))


@app.command("llamaindex-build")
def llamaindex_build(config: Optional[str] = ConfigOpt):
    """Build the LlamaIndex baseline (PGVectorStore + BM25) from the indexed chunks."""
    from nrag.index import store
    from nrag.retrieve import llamaindex_baseline

    cfg = load_config(config)
    typer.echo(f"Built: {llamaindex_baseline.build(cfg, store.connect(cfg))}")


@app.command()
def search(
    question: str = typer.Argument(..., help="Question to retrieve evidence for."),
    trace: bool = typer.Option(False, help="Show every pipeline stage with scores."),
    baseline: bool = typer.Option(False, help="Use the LlamaIndex baseline retriever instead."),
    config: Optional[str] = ConfigOpt,
):
    """Retrieve ranked evidence for a question (Phase 5; no answer generation)."""
    from nrag.retrieve.pipeline import retriever_from_config

    if baseline:
        from nrag.retrieve.llamaindex_baseline import LlamaIndexBaseline
        for r in LlamaIndexBaseline(load_config(config)).retrieve(question):
            typer.echo(f"{r['rank']:>2}. [{r['score']:.4f}] {r['doc_id']}  {r['text'][:100]}")
        return

    res = retriever_from_config(load_config(config)).retrieve(question)
    a = res.analysis
    typer.echo(f"Status: {res.status}{' (after one retry)' if res.retried else ''}")
    typer.echo(f"Understood: population={a['population_tags']} topics={a['topics']} intent={a['intent']} "
               f"answerable={a['answerable']}")
    typer.echo(f"Rewritten: {a['rewritten_query']}\n")
    for r in res.results:
        typer.echo(f"{r['rank']:>2}. [{r['final_score']:.3f} | rerank {r['rerank_score']:.3f} | q {r['quality_score'] or 0:.2f}] "
                   f"{r['design']} {r['year']}  {r['title'][:90]}")
        typer.echo(f"    {r['doc_id']} | {r['kind']}/{r['section']} | {r.get('url') or ''}")
    if trace:
        for k, v in res.stages.items():
            shown = v if not isinstance(v, list) else v[:10]
            typer.echo(f"\n[{k}] {json.dumps(shown, default=str)[:1500]}")
        typer.echo(f"\nTimings (s): {res.timings}")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to answer."),
    trace: bool = typer.Option(False, help="Show retrieval, safety and citation-check details."),
    config: Optional[str] = ConfigOpt,
):
    """Answer a question with cited, evidence-labelled research (Phase 6)."""
    from nrag.generate.answer import answerer_from_config

    out = answerer_from_config(load_config(config)).answer(question)
    typer.echo(out["markdown"])
    if trace:
        typer.echo(f"\n--- status: {out['status']} | safety: {out['safety']['mode']} {out['safety']['categories']}")
        if out.get("retrieval"):
            typer.echo(f"retrieval: {out['retrieval']['status']} timings={out['retrieval']['timings']}")
        for v in out["verification"]:
            typer.echo(f"claim {v['claim']}: {v['supported']} ({v['reason']})")
        if out["answer"]:
            typer.echo(f"notes: {out['answer'].get('notes')} safety_flags: {out['answer'].get('safety_flags')}")
        if out["errors"]:
            typer.echo(f"errors: {out['errors']}")


@app.command("eval-pool")
def eval_pool(workers: int = typer.Option(2, help="Concurrent LLM grading requests (shared server)."),
              config: Optional[str] = ConfigOpt):
    """Pool candidate documents per golden question from several systems; Gemma drafts 0-3 grades."""
    from nrag.eval import commands

    cfg = load_config(config)
    typer.echo(json.dumps(commands.pool_and_grade(cfg, commands.Components(cfg), workers), indent=1))


@app.command("eval-import-labels")
def eval_import_labels(config: Optional[str] = ConfigOpt):
    """Import relevance labels (your_grade overrides the LLM draft) into the golden set."""
    from nrag.eval import commands

    typer.echo(json.dumps(commands.import_labels(load_config(config)), indent=1))


@app.command("eval")
def eval_(split: str = typer.Option("all", help="all | dev | test"),
          answers: bool = typer.Option(True, help="Also generate and score answers."),
          judge: bool = typer.Option(True, help="LLM correctness judge."),
          label: str = typer.Option("base", help="Name for this run."),
          config: Optional[str] = ConfigOpt):
    """Run the evaluation on the golden set; writes eval/runs/<run>/ and eval/reports/run_<run>.md."""
    from nrag.eval import commands

    cfg = load_config(config)
    run_dir, metrics = commands.evaluate(cfg, commands.Components(cfg), split, answers, label, judge)
    typer.echo((run_dir / "report.md").read_text())


@app.command()
def ablate(split: str = typer.Option("all", help="all | dev | test"), config: Optional[str] = ConfigOpt):
    """Re-run retrieval evaluation with each component off; deltas with bootstrap 95% CIs."""
    from nrag.eval import commands

    cfg = load_config(config)
    run_dir, _ = commands.ablations(cfg, commands.Components(cfg), split)
    typer.echo((run_dir / "report.md").read_text())


@app.command()
def failures(run: str = typer.Argument(..., help="Run directory name under eval/runs."),
             config: Optional[str] = ConfigOpt):
    """Failure buckets for a run (after your corrections in its failures.csv, if any)."""
    from nrag.eval import failures as fail_mod

    cfg = load_config(config)
    path = resolve_path(cfg["eval"]["runs_dir"]) / run / "failures.csv"
    typer.echo(json.dumps(fail_mod.corrected_counts(path), indent=1))


@app.command()
def compare(run_a: str, run_b: str, config: Optional[str] = ConfigOpt):
    """Paired metric deltas (A - B) with bootstrap 95% CIs between two eval runs."""
    from nrag.eval import run as run_mod

    cfg = load_config(config)
    d = resolve_path(cfg["eval"]["runs_dir"])
    a, _ = run_mod.load_run(d / run_a)
    b, _ = run_mod.load_run(d / run_b)
    typer.echo(json.dumps(run_mod.compare_runs(a, b, run_mod.RETRIEVAL_KEYS + run_mod.ANSWER_KEYS), indent=1))


if __name__ == "__main__":
    app()
