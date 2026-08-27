import math

from ruhybrid.metrics import ndcg_at_k, recall_at_k


def test_recall_counts_relevant_in_topk():
    ranked = [5, 3, 1, 2, 4]
    assert recall_at_k(ranked, {1, 2}, 100) == 1.0
    assert recall_at_k(ranked, {3, 9}, 2) == 0.5
    assert recall_at_k(ranked, set(), 10) == 0.0


def test_ndcg_perfect_beats_shuffled_and_zero_without_relevant():
    assert math.isclose(ndcg_at_k([1, 2, 3], {1, 2, 3}, 10), 1.0)
    assert ndcg_at_k([1, 8, 9], {1}, 10) > ndcg_at_k([8, 9, 1], {1}, 10)
    assert ndcg_at_k([1, 2, 3], set(), 10) == 0.0
