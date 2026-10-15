"""`nrag` command-line interface."""

from __future__ import annotations

from typing import List, Optional

import typer

from nrag.config import env, load_config, resolve_path

app = typer.Typer(help="Evidence-ranked nutrition research RAG.", no_args_is_help=True,
                  add_completion=False)

ConfigOpt = typer.Option(None, "--config", help="Path to config.yaml (default: repo root).")


@app.command()
def ingest(
    since: Optional[str] = typer.Option(None, help="Incremental: only records entered since YYYY/MM/DD."),
    max_per_topic: Optional[int] = typer.Option(None, help="Cap records per topic (testing)."),
    topics: Optional[List[str]] = typer.Option(None, "--topic", help="Run only these topics (repeatable)."),
    strong_only: bool = typer.Option(False, help="Restrict to guidelines, reviews, RCTs and cohorts at search time."),
    config: Optional[str] = ConfigOpt,
):
    """Fetch PubMed records, filter, classify design and score."""
    from nrag.ingest.pubmed import PubMedClient, ingest as run_ingest, print_summary
    from nrag.net import CachedHttp

    cfg = load_config(config)
    unknown = set(topics or []) - set(cfg["pubmed"]["topics"])
    if unknown:
        raise typer.BadParameter(f"unknown topic(s): {', '.join(sorted(unknown))}")
    api_key = env("NCBI_API_KEY")
    ncbi = cfg["ncbi"]
    http = CachedHttp(
        resolve_path(cfg["paths"]["cache_dir"]) / "ncbi",
        delay_s=ncbi["delay_with_key_s"] if api_key else ncbi["delay_no_key_s"],
        timeout_s=ncbi["timeout_s"],
        retries=ncbi["retries"],
    )
    client = PubMedClient(http, ncbi, cfg["cache"], env("NCBI_EMAIL", required=True), api_key)
    records, excluded = run_ingest(
        cfg, client,
        resolve_path(cfg["paths"]["pubmed_corpus"]),
        resolve_path(cfg["paths"]["pubmed_excluded"]),
        since=since, max_per_topic=max_per_topic, topics=topics, strong_only=strong_only,
    )
    print_summary(records, excluded)
    typer.echo(f"\nHTTP cache: {http.stats['hits']} hits, {http.stats['misses']} misses")


def _not_yet(phase: int):
    typer.echo(f"Not implemented yet (Phase {phase}).", err=True)
    raise typer.Exit(code=2)


@app.command()
def extract():
    """LLM PICO / evidence-field extraction (Phase 2)."""
    _not_yet(2)


@app.command("extract-accuracy")
def extract_accuracy():
    """Field-level accuracy from eval/extraction_review.csv (Phase 2)."""
    _not_yet(2)


@app.command()
def score():
    """Quality scoring v2 and supersession (Phase 3)."""
    _not_yet(3)


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
