from eval.metrics import context_recall


def test_context_recall_full_overlap():
    assert context_recall(["a", "b", "c"], ["a", "b"]) == 1.0


def test_context_recall_partial_overlap():
    assert context_recall(["a", "x", "y"], ["a", "b"]) == 0.5


def test_context_recall_no_overlap():
    assert context_recall(["x", "y"], ["a", "b"]) == 0.0


def test_context_recall_unanswerable_question_returns_one():
    # No ground-truth chunks means there's nothing to have missed.
    assert context_recall(["x", "y"], []) == 1.0


def test_context_recall_empty_retrieval_with_ground_truth_is_zero():
    assert context_recall([], ["a", "b"]) == 0.0
