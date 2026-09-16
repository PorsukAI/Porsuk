from porsuk.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_merges_two_rankings():
    a = ["x", "y", "z"]
    b = ["y", "x", "w"]
    fused = reciprocal_rank_fusion([a, b], k=60)
    ids = [i for i, _ in fused]
    # y is rank 2 + rank 1, x is rank 1 + rank 2 -> tie; both above z and w
    assert set(ids[:2]) == {"x", "y"}
    assert ids[2:] == ["z", "w"] or ids[2:] == ["w", "z"]


def test_rrf_score_formula():
    fused = dict(reciprocal_rank_fusion([["a", "b"]], k=1))
    assert fused["a"] == 1 / (1 + 1)
    assert fused["b"] == 1 / (1 + 2)


def test_rrf_empty_rankings():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
