import pytest

from rag_assistant.evaluation import overlap, percentile, recall_at_k, reciprocal_rank, refusal_accuracy


def test_recall():
    assert recall_at_k([3, 7, 9], [7], k=2) == 1.0
    assert recall_at_k([3, 7, 9], [9], k=2) == 0.0
    with pytest.raises(ValueError):
        recall_at_k([1], [], k=1)


def test_mrr():
    assert reciprocal_rank([5, 2, 8], [8]) == pytest.approx(1 / 3)
    assert reciprocal_rank([5], [1]) == 0.0


def test_overlap_percentile_refusal():
    assert overlap([1, 2, 3], [1, 2, 4]) == pytest.approx(2 / 3)
    assert percentile([1, 2, 3, 4, 100], 95) == 100
    assert refusal_accuracy([(True, True), (False, True)]) == 0.5
