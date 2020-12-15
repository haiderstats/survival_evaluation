from typing import Optional

import numpy as np  # type: ignore
from scipy.stats import chi2  # type: ignore

from survival_evaluation.types import NumericArrayLike
from survival_evaluation.utility import (
    KaplanMeier,
    KaplanMeierArea,
    to_array,
    validate_size,
)


# pylint: disable=too-many-arguments
def l1(
    event_times: NumericArrayLike,
    event_indicators: NumericArrayLike,
    predictions: NumericArrayLike,
    training_event_times: Optional[NumericArrayLike] = None,
    training_event_indicators: Optional[NumericArrayLike] = None,
    l1_type: str = "hinge",
) -> float:

    event_times = to_array(event_times)
    event_indicators = to_array(event_indicators, to_boolean=True)
    predictions = to_array(predictions)

    validate_size(event_times, event_indicators, predictions)
    if l1_type == "hinge":
        scores = event_times - predictions
        scores[~event_indicators] = np.maximum(scores[~event_indicators], 0)
        return np.mean(np.abs(scores))

    if l1_type == "margin":
        if training_event_times is None or training_event_indicators is None:
            raise ValueError(
                "If 'margin' is chosen, training set values must be included."
            )

        training_event_times = to_array(training_event_times)
        training_event_indicators = to_array(training_event_indicators, to_boolean=True)

        km_model = KaplanMeierArea(training_event_times, training_event_indicators)
        censor_times = event_times[~event_indicators]
        weights = 1 - km_model.predict(censor_times)
        best_guesses = km_model.best_guess(censor_times)

        scores = np.empty(predictions.size)
        scores[event_indicators] = (
            event_times[event_indicators] - predictions[event_indicators]
        )
        scores[~event_indicators] = weights * (
            best_guesses - predictions[~event_indicators]
        )
        weighted_multiplier = 1 / (np.sum(event_indicators) + np.sum(weights))
        return weighted_multiplier * np.sum(np.abs(scores))

    raise ValueError("L1 type must be either 'hinge' or 'margin'.")


# pylint: disable=too-many-arguments
def one_calibration(
    event_times: NumericArrayLike,
    event_indicators: NumericArrayLike,
    predictions: NumericArrayLike,
    time: float,
    bins: int = 10,
) -> float:

    event_times = to_array(event_times)
    event_indicators = to_array(event_indicators, to_boolean=True)
    predictions = 1 - to_array(predictions)

    prediction_order = np.argsort(-predictions)
    predictions = predictions[prediction_order]
    event_times = event_times[prediction_order]
    event_indicators = event_indicators[prediction_order]

    # Can't do np.mean since split array may be of different sizes.
    binned_event_times = np.array_split(event_times, bins)
    binned_event_indicators = np.array_split(event_indicators, bins)
    probability_means = [np.mean(x) for x in np.array_split(predictions, bins)]
    hosmer_lemeshow = 0
    for b in range(bins):
        prob = probability_means[b]
        if prob == 1.0:
            raise ValueError(
                "One-Calibration is not well defined: the risk"
                f"probability of the {b}th bin was {prob}."
            )
        km_model = KaplanMeier(binned_event_times[b], binned_event_indicators[b])
        event_probability = 1 - km_model.predict(time)
        bin_count = len(binned_event_times[b])
        hosmer_lemeshow += (bin_count * event_probability - bin_count * prob) ** 2 / (
            bin_count * prob * (1 - prob)
        )

    return 1 - chi2.cdf(hosmer_lemeshow, bins - 1)


def d_calibration(
    event_indicators: NumericArrayLike,
    predictions: NumericArrayLike,
    bins: int = 10,
) -> float:

    event_indicators = to_array(event_indicators, to_boolean=True)
    predictions = to_array(predictions)

    # include minimum to catch if probability = 1.
    bin_index = np.minimum(np.floor(predictions * bins), bins - 1).astype(int)
    censored_bin_indexes = bin_index[~event_indicators]
    uncensored_bin_indexes = bin_index[event_indicators]

    censored_predictions = predictions[~event_indicators]
    censored_contribution = 1 - (censored_bin_indexes / bins) * (
        1 / censored_predictions
    )
    censored_following_contribution = 1 / (bins * censored_predictions)

    contribution_pattern = np.tril(np.ones([bins, bins]), k=-1).astype(bool)

    following_contributions = np.matmul(
        censored_following_contribution, contribution_pattern[censored_bin_indexes]
    )
    single_contributions = np.matmul(
        censored_contribution, np.eye(bins)[censored_bin_indexes]
    )
    uncensored_contributions = np.sum(np.eye(bins)[uncensored_bin_indexes], axis=0)
    bin_count = (
        single_contributions + following_contributions + uncensored_contributions
    )
    chi2_statistic = np.sum(
        np.square(bin_count - len(predictions) / bins) / (len(predictions) / bins)
    )
    return 1 - chi2.cdf(chi2_statistic, bins - 1)
