import pytest

from core.retrieval import DEFAULT_RRF_K, reciprocal_rank_fusion


def test_hand_worked_example_with_k_equals_1():
    # ranking_1: A, B, C (best to worst) -> ranks A=1, B=2, C=3
    # ranking_2: B, C, A (best to worst) -> ranks B=1, C=2, A=3
    # with k=1: score(x) = sum of 1/(1+rank) across rankings containing x
    # A: 1/(1+1) + 1/(1+3) = 0.5     + 0.25     = 0.75
    # B: 1/(1+2) + 1/(1+1) = 0.3333… + 0.5       = 0.8333…
    # C: 1/(1+3) + 1/(1+2) = 0.25    + 0.3333…   = 0.5833…
    rankings = [["A", "B", "C"], ["B", "C", "A"]]

    results = reciprocal_rank_fusion(rankings, k=1)
    scores = {r.item_id: r.score for r in results}

    assert scores["A"] == pytest.approx(0.75)
    assert scores["B"] == pytest.approx(5 / 6)
    assert scores["C"] == pytest.approx(7 / 12)


def test_results_sorted_best_score_first():
    rankings = [["A", "B", "C"], ["B", "C", "A"]]

    results = reciprocal_rank_fusion(rankings, k=1)

    assert [r.item_id for r in results] == ["B", "A", "C"]


def test_item_missing_from_one_ranking_still_gets_scored():
    # ranking_1: X=rank1, Y=rank2. ranking_2: Y=rank1. X only appears in ranking_1.
    rankings = [["X", "Y"], ["Y"]]

    results = reciprocal_rank_fusion(rankings, k=DEFAULT_RRF_K)
    scores = {r.item_id: r.score for r in results}

    assert scores["X"] == pytest.approx(1 / (DEFAULT_RRF_K + 1))
    assert scores["Y"] == pytest.approx(1 / (DEFAULT_RRF_K + 2) + 1 / (DEFAULT_RRF_K + 1))
    assert scores["Y"] > scores["X"]


def test_single_ranking_preserves_original_order():
    rankings = [["first", "second", "third"]]

    results = reciprocal_rank_fusion(rankings)

    assert [r.item_id for r in results] == ["first", "second", "third"]


def test_empty_rankings_produces_empty_result():
    assert reciprocal_rank_fusion([]) == []


def test_deterministic_across_repeated_calls():
    rankings = [["A", "B", "C"], ["C", "A", "B"]]

    assert reciprocal_rank_fusion(rankings) == reciprocal_rank_fusion(rankings)
