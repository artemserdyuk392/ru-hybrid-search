"""Retrieval metrics: Recall@k and nDCG@k, kept dependency-free on purpose."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping


def _gain(relevant, doc_id) -> float:
    if isinstance(relevant, Mapping):
        return float(relevant.get(doc_id, 0))
    return 1.0 if doc_id in relevant else 0.0


def _ideal_gains(relevant) -> list[float]:
    if isinstance(relevant, Mapping):
        return sorted((float(v) for v in relevant.values()), reverse=True)
    return [1.0] * len(relevant)


def _dcg(gains: Iterable[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def recall_at_k(ranked: list, relevant, k: int) -> float:
    total = len(relevant)
    if total == 0:
        return 0.0
    hits = sum(1 for doc_id in ranked[:k] if _gain(relevant, doc_id) > 0)
    return hits / total


def ndcg_at_k(ranked: list, relevant, k: int) -> float:
    gains = [_gain(relevant, doc_id) for doc_id in ranked[:k]]
    # the ideal DCG is over the top-k ideal gains, not every relevant doc,
    # otherwise a perfect ranking scores below 1.0 whenever there are more
    # than k relevant documents.
    idcg = _dcg(_ideal_gains(relevant)[:k])
    if idcg == 0:
        return 0.0
    return _dcg(gains) / idcg
