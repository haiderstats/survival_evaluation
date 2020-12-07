from dataclasses import InitVar, dataclass, field

import numpy as np  # type: ignore

from survival_evaluation.types import NumericArrayLike
from survival_evaluation.utility import check_indicators, to_array, validate_size


def l1(
    event_times: NumericArrayLike,
    event_indicators: NumericArrayLike,
    predictions: NumericArrayLike,
    l1_type: str = "margin",
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
    return -1.0


@dataclass
class KaplanMeier:
    event_times: InitVar[np.array]
    event_indicators: InitVar[np.array]
    survival_times: np.array = field(init=False)
    survival_probabilities: np.array = field(init=False)

    def __post_init__(self, event_times, event_indicators):
        index = np.lexsort((event_indicators, event_times))
        unique_times = np.unique(event_times[index], return_counts=True)
        self.survival_times = unique_times[0]
        population_count = np.flip(np.cumsum(np.flip(unique_times[1])))

        event_counter = np.append(0, np.cumsum(unique_times[1])[:-1])
        event_ind = list()
        for i in range(np.size(event_counter[:-1])):
            event_ind.append(event_counter[i])
            event_ind.append(event_counter[i + 1])
        event_ind.append(event_counter[-1])
        event_ind.append(len(event_indicators))
        events = np.add.reduceat(np.append(event_indicators[index], 0), event_ind)[::2]

        self.survival_probabilities = np.empty(population_count.size)
        survival_probability = 1
        counter = 0
        for population, event_num in zip(population_count, events):
            survival_probability *= 1 - event_num / population
            self.survival_probabilities[counter] = survival_probability
            counter += 1

    def predict(self, prediction_times: np.array):
        probability_index = np.digitize(prediction_times, self.survival_times)
        probability_index = np.where(
            probability_index == self.survival_times.size,
            probability_index - 1,
            probability_index,
        )
        probabilities = self.survival_probabilities[probability_index]
        probabilities = np.where(
            prediction_times < self.survival_times[0], 1, probabilities
        )

        return probabilities


# import time

# if __name__ == "__main__":
#     start = time.time()
#     event_times = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#     event_indicators = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
#     km = KaplanMeier(event_times, event_indicators)
#     print(km.predict(np.array([0, 0.5, 1, 1.1, 1.9, 2, 10, 10.5])))
#     print(time.time() - start)
