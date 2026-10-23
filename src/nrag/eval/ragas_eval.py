"""RAGAS metrics on an eval run, with local/self-hosted models only.

LLM: the active extraction model through its OpenAI-compatible endpoint (the user's vLLM server),
embeddings: bge-m3 locally. Metrics: faithfulness, response relevancy, context precision and
context recall (the last two use the golden reference answer). RAGAS numbers are reported next to
our own metrics, not instead of them: both rely on an LLM judge, and ours is validated against
human labels.
"""

from __future__ import annotations

from typing import Any


def build(cfg: dict[str, Any], base_url: str, model: str, max_workers: int = 2):
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    llm = LangchainLLMWrapper(ChatOpenAI(base_url=base_url.rstrip("/") + "/v1", api_key="not-needed", model=model,
                                         temperature=0, max_tokens=2048, timeout=600))
    emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name=cfg["index"]["embedding_model"], model_kwargs={"device": cfg["index"]["device"]},
        encode_kwargs={"normalize_embeddings": True}))
    return llm, emb


def samples_from_rows(rows: list[dict[str, Any]], references: dict[str, str]) -> tuple[list, list[str]]:
    from ragas import SingleTurnSample

    samples, ids = [], []
    for r in rows:
        if r.get("abstained") or not r.get("direct_answer") or not r.get("contexts"):
            continue
        samples.append(SingleTurnSample(user_input=r["question"], response=r["direct_answer"],
                                        retrieved_contexts=r["contexts"], reference=references.get(r["question_id"]) or None))
        ids.append(r["question_id"])
    return samples, ids


def evaluate_rows(cfg: dict[str, Any], rows: list[dict[str, Any]], references: dict[str, str], base_url: str,
                  model: str, max_workers: int = 2) -> dict[str, Any]:
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

    llm, emb = build(cfg, base_url, model, max_workers)
    samples, ids = samples_from_rows(rows, references)
    if not samples:
        return {"n": 0}
    metrics = [Faithfulness(llm=llm), ResponseRelevancy(llm=llm, embeddings=emb),
               LLMContextPrecisionWithReference(llm=llm), LLMContextRecall(llm=llm)]
    result = evaluate(EvaluationDataset(samples=samples), metrics=metrics,
                      run_config=RunConfig(max_workers=max_workers, timeout=900, max_retries=3), show_progress=False)
    df = result.to_pandas()
    per_q = {qid: {c: (None if df[c].iloc[i] != df[c].iloc[i] else round(float(df[c].iloc[i]), 4))
                   for c in df.columns if c not in ("user_input", "response", "retrieved_contexts", "reference")}
             for i, qid in enumerate(ids)}
    cols = [c for c in df.columns if c not in ("user_input", "response", "retrieved_contexts", "reference")]
    means = {c: round(float(df[c].mean(skipna=True)), 4) for c in cols}
    return {"n": len(samples), "means": means, "per_question": per_q}
