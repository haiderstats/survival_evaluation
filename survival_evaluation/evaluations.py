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
    survival_curve: np.array = field(init=False)

    def __post_init__(self, event_times, event_indicators):
        index = np.lexsort((event_indicators, event_times))
        unique_counts = np.unique(event_times[index], return_counts=True)[1]
        population_count = np.flip(np.cumsum(np.flip(unique_counts)))

        event_counter = np.append(0, np.cumsum(unique_counts)[:-1])
        event_ind = list()
        for i in range(np.size(event_counter[:-1])):
            event_ind.append(event_counter[i])
            event_ind.append(event_counter[i + 1])
        event_ind.append(event_counter[-1])
        event_ind.append(len(event_indicators))
        events = np.add.reduceat(np.append(event_indicators[index], 0), event_ind)[::2]

        self.survival_curve = np.empty(population_count.size)
        survival_probability = 1
        counter = 0
        for population, event_num in zip(population_count, events):
            survival_probability *= 1 - event_num / population
            self.survival_curve[counter] = survival_probability
            counter += 1

    def predict(self, time):
        pass

    # np.digitize


# import time

# if __name__ == "__main__":
#     start = time.time()
#     event_times = np.random.randint(1, 10000, size=100000)
#     event_indicators = np.random.randint(2, size=100000)
#     km = KaplanMeier(event_times, event_indicators)
#     print(time.time() - start)
#     print(km.survival_curve)
