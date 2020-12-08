import numpy as np  # type: ignore

from survival_evaluation.evaluations import l1


def test_hinge_perfect():
    event_times = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    event_indicators = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    predictions = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert l1(event_times, event_indicators, predictions, l1_type="hinge") == 0


def test_hinge_perfect_w_censoring():
    event_times = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    event_indicators = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
    predictions = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert l1(event_times, event_indicators, predictions, l1_type="hinge") == 0


def test_hinge_perfect_censoring():
    event_times = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    event_indicators = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    predictions = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert l1(event_times, event_indicators, predictions, l1_type="hinge") == 0


def test_margin_perfect():
    event_times = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    event_indicators = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    predictions = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    evaluation = l1(
        event_times,
        event_indicators,
        predictions,
        event_times,
        event_indicators,
        l1_type="margin",
    )
    assert evaluation == 0


def test_margin_perfect_tiny_censoring():
    event_times = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    event_indicators = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 1])
    predictions = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    evaluation = l1(
        event_times,
        event_indicators,
        predictions,
        event_times,
        event_indicators,
        l1_type="margin",
    )
    assert np.round(evaluation, 10) == np.round((1 / (9 + 0.8)) * 0.8 * (10 - 9), 10)
