"""Cross-encoder reranking (bge-reranker-v2-m3). Scores are sigmoid probabilities in [0, 1],
comparable across queries, so the confidence threshold can be absolute."""

from __future__ import annotations

from typing import Any, Callable


class Reranker:
    def __init__(self, model_name: str, device: str = "cpu", max_length: int = 512, batch_size: int = 16,
                 score_fn: Callable[[str, list[str]], list[float]] | None = None):
        self.model_name, self.device, self.max_length, self.batch_size = model_name, device, max_length, batch_size
        self._score_fn = score_fn
        self._model = None

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        if self._score_fn is not None:
            return list(self._score_fn(query, passages))
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, device=self.device, max_length=self.max_length)
        return [float(s) for s in self._model.predict([(query, p) for p in passages], batch_size=self.batch_size)]


def reranker_from_config(rcfg: dict[str, Any]) -> Reranker:
    return Reranker(rcfg["model"], rcfg["device"], rcfg["max_length"], rcfg["batch_size"])
