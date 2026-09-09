import math

from ruhybrid.metrics import mrr_at_k, ndcg_at_k, recall_at_k


def test_recall_counts_relevant_in_topk():
    ranked = [5, 3, 1, 2, 4]
    assert recall_at_k(ranked, {1, 2}, 100) == 1.0
    assert recall_at_k(ranked, {1, 2}, 2) == 0.0
    assert recall_at_k(ranked, {3, 9}, 2) == 0.5


def test_recall_with_no_relevant_is_zero():
    assert recall_at_k([1, 2, 3], set(), 10) == 0.0


def test_ndcg_perfect_ranking_is_one():
    assert math.isclose(ndcg_at_k([1, 2, 3], {1, 2, 3}, 10), 1.0)


def test_ndcg_rewards_higher_rank():
    good = ndcg_at_k([1, 8, 9], {1}, 10)
    bad = ndcg_at_k([8, 9, 1], {1}, 10)
    assert good > bad
    assert math.isclose(good, 1.0)


def test_ndcg_with_no_relevant_is_zero():
    assert ndcg_at_k([1, 2, 3], set(), 10) == 0.0


def test_ndcg_idcg_truncated_to_k_when_relevant_exceeds_k():
    # perfect ranking must score 1.0 even with more relevant docs than k
    assert math.isclose(ndcg_at_k([1, 2, 3, 4, 5], {1, 2, 3, 4, 5}, 2), 1.0)


def test_mrr_uses_rank_of_first_relevant():
    assert mrr_at_k([9, 3, 1], {3}, 10) == 0.5
    assert mrr_at_k([1, 2, 3], {1}, 10) == 1.0
    assert mrr_at_k([9, 8, 7], {1}, 10) == 0.0
    assert mrr_at_k([9, 8, 1], {1}, 2) == 0.0
