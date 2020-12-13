import numpy as np  # type: ignore

from survival_evaluation.evaluations import d_calibration, l1


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


def test_d_calibration_perfect_values():
    event_indicators = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    predictions = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    pvalue, _ = d_calibration(event_indicators, predictions)
    assert round(pvalue, 3) == 1.000


def test_d_calibration_awful():
    event_indicators = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    predictions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pvalue, _ = d_calibration(event_indicators, predictions)
    assert round(pvalue, 3) == 0.000


# I'm so so sorry.
def test_d_calibration_20_values():
    event_indicators = np.array(
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    )
    predictions = np.array(
        [
            0.05,
            0.15,
            0.17,
            0.22,
            0.28,
            0.31,
            0.38,
            0.42,
            0.51,
            0.55,
            0.59,
            0.60,
            0.61,
            0.71,
            0.73,
            0.89,
            0.91,
            0.92,
            0.93,
            1.00,
        ]
    )
    pvalue, deaths = d_calibration(event_indicators, predictions)
    assert round(deaths[9], 3) == round((2 + 0.1 + 0.02173913) / 20, 3)
    assert round(deaths[8], 3) == round((0.1011235 + 0.1 + 0.10869) / 20, 3)
    assert round(deaths[7], 3) == round(
        (1 + 0.0140845 + 0.112359 + 0.1 + 0.10869) / 20, 3
    )
    assert round(deaths[6], 3) == round(
        (1 + 0.14084507 + 0.112359 + 0.1 + 0.10869) / 20, 3
    )
    assert round(deaths[5], 3) == round(
        (2 + 0.090909 + 0.166667 + 0.140845 + 0.112359 + 0.1 + 0.10869) / 20, 3
    )
    assert round(deaths[4], 3) == round(
        (0.047619 + 0.181818 + 0.166667 + 0.140845 + 0.112359 + 0.1 + 0.10869) / 20, 3
    )
    assert round(deaths[3], 3) == round(
        (
            1
            + 0.032258065
            + 0.2380952
            + 0.181818
            + 0.166667
            + 0.140845
            + 0.112359
            + 0.1
            + 0.10869
        )
        / 20,
        3,
    )
    assert round(deaths[2], 3) == round(
        (
            1
            + 0.09090909
            + 0.32258065
            + 0.2380952
            + 0.181818
            + 0.166667
            + 0.140845
            + 0.112359
            + 0.1
            + 0.10869
        )
        / 20,
        3,
    )
    assert round(deaths[1], 3) == round(
        (
            1
            + 0.333333
            + 0.4545454
            + 0.32258065
            + 0.2380952
            + 0.181818
            + 0.166667
            + 0.140845
            + 0.112359
            + 0.1
            + 0.10869
        )
        / 20,
        3,
    )
    assert round(deaths[0], 3) == round(
        (
            1
            + 0.6666667
            + 0.4545454
            + 0.32258065
            + 0.2380952
            + 0.181818
            + 0.166667
            + 0.140845
            + 0.112359
            + 0.1
            + 0.10869
        )
        / 20,
        3,
    )
    assert round(pvalue, 3) == 0.867
