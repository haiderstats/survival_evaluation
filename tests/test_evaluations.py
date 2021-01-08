import numpy as np  # type: ignore
import pytest  # type: ignore
from scipy.stats import chi2  # type: ignore

from survival_evaluation.evaluations import d_calibration, l1, one_calibration


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


def test_one_calibration_raises_error():
    event_times = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    event_indicators = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    predictions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    time = 2
    with pytest.raises(ValueError):
        one_calibration(event_times, event_indicators, predictions, time)


def test_one_calibration_failure():
    event_times = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    event_indicators = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    predictions = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])
    time = 2
    pvalue = one_calibration(event_times, event_indicators, predictions, time)[
        "p_value"
    ]
    assert round(pvalue, 3) == 0.000


def test_one_calibration():
    event_times = [
        455,
        210,
        1022,
        310,
        361,
        218,
        166,
        170,
        567,
        613,
        707,
        61,
        301,
        81,
        371,
        520,
        574,
        118,
        390,
        12,
        473,
        26,
        107,
        53,
        814,
        965,
        93,
        731,
        460,
        153,
        433,
        583,
        95,
        303,
        519,
        643,
        765,
        53,
        246,
        689,
        5,
        687,
        345,
        444,
        223,
        60,
        163,
        65,
        821,
        428,
        230,
        840,
        305,
        11,
        226,
        426,
        705,
        363,
        176,
        791,
        95,
        196,
        167,
        806,
        284,
        641,
        147,
        740,
        163,
        655,
        88,
        245,
        30,
        477,
        559,
        450,
        156,
        529,
        429,
        351,
        15,
        181,
        283,
        13,
        212,
        524,
        288,
        363,
        199,
        550,
        54,
        558,
        207,
        92,
        60,
        551,
        293,
        353,
        267,
        511,
        457,
        337,
        201,
        404,
        222,
        62,
        458,
        353,
        163,
        31,
        229,
        156,
        291,
        179,
        376,
        384,
        268,
        292,
        142,
        413,
        266,
        320,
        181,
        285,
        301,
        348,
        197,
        382,
        303,
        296,
        180,
        145,
        269,
        300,
        284,
        292,
        332,
        285,
        259,
        110,
        286,
        270,
        225,
        269,
        225,
        243,
        276,
        135,
        79,
        59,
        240,
        202,
        235,
        239,
        252,
        221,
        185,
        222,
        183,
        211,
        175,
        197,
        203,
        191,
        105,
        174,
        177,
    ]

    event_indicators = np.array(
        [
            1,
            1,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            1,
            1,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            1,
            0,
            1,
            1,
            1,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            1,
            1,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            1,
            1,
            1,
            0,
            1,
            1,
            1,
            0,
            1,
            1,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            1,
            0,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            1,
            1,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            1,
            1,
            1,
            0,
            1,
            0,
            0,
            0,
            1,
            1,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
    )

    predictions = [
        0.8019031,
        0.6408094,
        0.5546164,
        0.5773482,
        0.6147166,
        0.6532551,
        0.5428548,
        0.7073092,
        0.8072169,
        0.7424629,
        0.5262462,
        0.550347,
        0.6875078,
        0.8321219,
        0.8282183,
        0.7797371,
        0.7977442,
        0.449106,
        0.6147215,
        0.4276821,
        0.8026458,
        0.4750253,
        0.5312576,
        0.6406371,
        0.5282871,
        0.7774541,
        0.4125527,
        0.8303778,
        0.5959777,
        0.6306949,
        0.8724286,
        0.5865418,
        0.4854926,
        0.7348299,
        0.6350795,
        0.7977329,
        0.8521829,
        0.8137608,
        0.8066887,
        0.7162866,
        0.8332992,
        0.7836661,
        0.767761,
        0.6245639,
        0.810955,
        0.780627,
        0.5812366,
        0.4406558,
        0.8203158,
        0.7826016,
        0.7249483,
        0.7891305,
        0.8128987,
        0.6034382,
        0.7933256,
        0.8307611,
        0.8604378,
        0.7421858,
        0.7214845,
        0.7650237,
        0.7283151,
        0.7078839,
        0.7912299,
        0.6804085,
        0.7061995,
        0.7845782,
        0.8196074,
        0.82983,
        0.4888712,
        0.7946163,
        0.6755585,
        0.7008508,
        0.4707396,
        0.7309513,
        0.8904794,
        0.7930871,
        0.7064338,
        0.8264507,
        0.7220057,
        0.5411261,
        0.723002,
        0.7065871,
        0.7156136,
        0.4950752,
        0.509987,
        0.5535522,
        0.5156649,
        0.6655757,
        0.6907206,
        0.7287642,
        0.4311525,
        0.7937547,
        0.7028424,
        0.6284731,
        0.6876172,
        0.6608529,
        0.8312718,
        0.8152734,
        0.6888168,
        0.5464261,
        0.6906316,
        0.8530603,
        0.5851429,
        0.6488967,
        0.5679515,
        0.7813227,
        0.7507115,
        0.7662725,
        0.7241959,
        0.803509,
        0.5712843,
        0.4486743,
        0.5016122,
        0.5961632,
        0.8013915,
        0.8739488,
        0.8794363,
        0.6419602,
        0.6676567,
        0.6188532,
        0.8727374,
        0.8357409,
        0.6887415,
        0.7980068,
        0.7348448,
        0.8560611,
        0.669059,
        0.8698747,
        0.7054218,
        0.8183707,
        0.5261072,
        0.7876375,
        0.8717771,
        0.8288318,
        0.8003905,
        0.8548698,
        0.8886875,
        0.688836,
        0.7753272,
        0.6226393,
        0.8170398,
        0.6873293,
        0.7163855,
        0.7257527,
        0.8318468,
        0.8041266,
        0.8501849,
        0.6891629,
        0.8387537,
        0.6317905,
        0.8934105,
        0.8790651,
        0.8668981,
        0.5892356,
        0.8488244,
        0.6290668,
        0.6497142,
        0.6804241,
        0.5094187,
        0.4523887,
        0.8286029,
        0.7266575,
        0.7865056,
        0.8365328,
        0.6439156,
        0.7331439,
        0.8057345,
    ]
    pvalue = one_calibration(event_times, event_indicators, predictions, time=200.0)[
        "p_value"
    ]
    assert round(pvalue, 3) == 0.112


def test_d_calibration_perfect_values():
    event_indicators = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    predictions = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    pvalue = d_calibration(event_indicators, predictions)["p_value"]
    assert round(pvalue, 3) == 1.000


def test_d_calibration_fail():
    event_indicators = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    predictions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pvalue = d_calibration(event_indicators, predictions)["p_value"]
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
    dcal_pvalue = d_calibration(event_indicators, predictions)["p_value"]
    proportions = np.array(
        [
            round((2 + 0.1 + 0.0217391), 3),
            round((0.1011235 + 0.1 + 0.1086), 3),
            round((1 + 0.0140845 + 0.112359 + 0.1 + 0.1086), 3),
            round((1 + 0.14084507 + 0.112359 + 0.1 + 0.1086), 3),
            round((2 + 0.090909 + 0.166667 + 0.140845 + 0.112359 + 0.1 + 0.1086), 3),
            round(
                (0.047619 + 0.181818 + 0.166667 + 0.140845 + 0.112359 + 0.1 + 0.1086), 3
            ),
            round(
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
                ),
                3,
            ),
            round(
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
                ),
                3,
            ),
            round(
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
                ),
                3,
            ),
            round(
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
                ),
                3,
            ),
        ]
    )
    bins = 10
    chi2_statistic = np.sum(
        np.square(proportions - len(predictions) / bins) / (len(predictions) / bins)
    )
    pvalue = 1 - chi2.cdf(chi2_statistic, bins - 1)
    assert round(dcal_pvalue, 3) == round(pvalue, 3)
