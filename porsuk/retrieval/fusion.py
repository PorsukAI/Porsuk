"""Reciprocal Rank Fusion for the optional hybrid search.

Rank-merge only: no score normalisation, so keyword and semantic scores on
different scales combine cleanly. score(d) = Σ 1/(k + rank_i(d)), k = 60.
"""

from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(rankings: list[list[str]], *, k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
