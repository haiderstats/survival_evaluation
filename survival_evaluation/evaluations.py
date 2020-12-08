from typing import Optional

import numpy as np  # type: ignore

from survival_evaluation.types import NumericArrayLike
from survival_evaluation.utility import (
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
    training_event_times: Optional[NumericArrayLike],
    training_event_indicators: Optional[NumericArrayLike],
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

    elif l1_type == "margin":
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

    else:
        raise ValueError("L1 type must be either 'hinge' or 'margin'.")

    return np.mean(np.abs(scores))


# import time

# if __name__ == "__main__":
#     start = time.time()
#     event_times = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#     event_indicators = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0, 1])
#     kma = KaplanMeierArea(event_times, event_indicators)
#     print(kma.survival_times)
#     # kma = KaplanMeierArea(km)
#     print(kma.area)
#     print(kma.area_times)
#     print(kma.area_probabilities)
#     print(kma.best_guess(np.array([1.5, 2.5, 2.02, 1, 7, 9])))
#     print(time.time() - start)
