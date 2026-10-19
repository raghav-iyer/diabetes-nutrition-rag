"""`nrag` command-line interface."""

from __future__ import annotations

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


@app.command()
def index():
    """Chunk documents and build LanceDB + BM25 indexes (Phase 4)."""
    _not_yet(4)


@app.command()
def ask(question: str = typer.Argument(..., help="Question to answer.")):
    """Answer a question with cited, evidence-ranked sources (Phase 6/8)."""
    _not_yet(6)


@app.command("eval")
def eval_():
    """Run retrieval and answer evaluation on the golden set (Phase 7)."""
    _not_yet(7)


@app.command()
def ablate():
    """Run the eval with each pipeline component switched off (Phase 7)."""
    _not_yet(7)


@app.command()
def failures():
    """Classify failed eval questions into root-cause buckets (Phase 7)."""
    _not_yet(7)


@app.command()
def compare(run_a: str, run_b: str):
    """Compare two eval runs (Phase 7)."""
    _not_yet(7)


if __name__ == "__main__":
    app()
