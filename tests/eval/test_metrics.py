import math

from eval.metrics import mrr, ndcg_at_k, recall_at_k


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"a", "d"}, k=3) == 0.5
    assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=1) == 0.5
    assert recall_at_k(["a"], set(), k=5) == 0.0


def test_mrr():
    assert mrr(["x", "a", "b"], {"a"}) == 0.5
    assert mrr(["a", "b"], {"a"}) == 1.0
    assert mrr(["x", "y"], {"a"}) == 0.0


def test_ndcg_perfect_and_reversed():
    # two relevant, both at top -> nDCG 1.0
    assert ndcg_at_k(["a", "b", "c"], {"a", "b"}, k=3) == 1.0
    # one relevant at position 3
    got = ndcg_at_k(["x", "y", "a"], {"a"}, k=3)
    assert math.isclose(got, (1 / math.log2(4)) / 1.0)


def test_ndcg_empty_relevant():
    assert ndcg_at_k(["a", "b"], set(), k=2) == 0.0
