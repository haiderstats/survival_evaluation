from typing import Optional

import numpy as np  # type: ignore
from scipy.stats import chi2  # type: ignore

from survival_evaluation.types import NumericArrayLike
from survival_evaluation.utility import (
    KaplanMeier,
    KaplanMeierArea,
    check_indicators,
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

    check_indicators(event_indicators)

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
    check_indicators(event_indicators)

    event_times = to_array(event_times)
    event_indicators = to_array(event_indicators, to_boolean=True)
    predictions = 1 - to_array(predictions)

    prediction_order = np.argsort(predictions)
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
        km_model = KaplanMeier(binned_event_times[b], binned_event_indicators[b])
        event_probability = 1 - km_model.predict(time)
        bin_count = len(binned_event_times[b])
        hosmer_lemeshow += (bin_count * event_probability - bin_count * prob) ** 2 / (
            bin_count * prob * (1 - prob)
        )

    return 1 - chi2.cdf(hosmer_lemeshow, bins - 1)


# import time

# if __name__ == "__main__":
#     start = time.time()
#     event_times = np.array(
#         [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5]
#     )
#     event_indicators = np.array(
#         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
#     )
#     predictions = np.array(
#         [
#             0.1,
#             0.05,
#             0.1,
#             0.2,
#             0.4,
#             0.4,
#             0.4,
#             0.5,
#             0.2,
#             0.6,
#             0.6,
#             0.65,
#             0.76,
#             0.2,
#             0.8,
#             0.87,
#             0.8,
#             0.9,
#             0.92,
#             0.99,
#             0.8,
#             0.78,
#             0.99,
#             0.86,
#         ]
#     )
#     print(one_calibration(event_times, event_indicators, predictions, 3, 10))
