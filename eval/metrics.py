"""Retrieval metrics: recall@k, MRR, nDCG@k, under binary relevance (a
document either is or is not a gold reference for a question).
"""

from __future__ import annotations

import math


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hit = sum(1 for r in ranked[:k] if r in relevant)
    return hit / len(relevant)


def mrr(ranked: list[str], relevant: set[str]) -> float:
    for i, r in enumerate(ranked, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 1) for i, r in enumerate(ranked[:k], start=1) if r in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0
