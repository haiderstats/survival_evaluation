from dataclasses import InitVar, dataclass, field

import numpy as np  # type: ignore

from survival_evaluation.types import NumericArrayLike


def to_array(array_like: NumericArrayLike, to_boolean: bool = False) -> np.array:
    array = np.asarray(array_like)
    shape = np.shape(array)
    if len(shape) > 1:
        raise ValueError(
            f"Input should be a 1-d array. Got a shape of {shape} instead."
        )
    if np.any(array < 0):
        raise ValueError("All event times must be greater than or equal to zero.")
    if to_boolean:
        check_indicators(array)
        return array.astype(bool)
    return array


def check_indicators(indicators: np.array) -> None:
    if not all(np.logical_or(indicators == 0, indicators == 1)):
        raise ValueError(
            "Event indicators must be 0 or 1 where 0 indicates censorship and 1 is an event."
        )


def validate_size(
    event_times: NumericArrayLike,
    event_indicators: NumericArrayLike,
    predictions: NumericArrayLike,
):
    same_size = (
        np.shape(event_times) == np.shape(event_indicators) == np.shape(predictions)
    )
    if not same_size:
        raise ValueError("All three inputs must be of the same shape.")


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
        population_count = np.flip(np.flip(unique_times[1]).cumsum())

        event_counter = np.append(0, unique_times[1].cumsum()[:-1])
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
            probability_index == self.survival_times.size + 1,
            probability_index - 1,
            probability_index,
        )
        probabilities = np.append(1, self.survival_probabilities)[probability_index]

        return probabilities


@dataclass
class KaplanMeierArea(KaplanMeier):
    area_times: np.array = field(init=False)
    area_probabilities: np.array = field(init=False)
    area: np.array = field(init=False)

    def __post_init__(self, event_times, event_indicators):
        super().__post_init__(event_times, event_indicators)
        area_probabilities = np.append(1, self.survival_probabilities)
        area_times = np.append(0, self.survival_times)
        if self.survival_probabilities[-1] != 0:
            slope = (area_probabilities[-1] - 1) / area_times[-1]
            zero_survival = -1 / slope
            area_times = np.append(area_times, zero_survival)
            area_probabilities = np.append(area_probabilities, 0)

        area_diff = np.diff(area_times, 1)
        area = np.flip(np.flip(area_diff * area_probabilities[0:-1]).cumsum())

        self.area_times = np.append(area_times, np.inf)
        self.area_probabilities = area_probabilities
        self.area = np.append(area, 0)

    def best_guess(self, censor_times: np.array):
        surv_prob = self.predict(censor_times)
        censor_indexes = np.digitize(censor_times, self.area_times)
        censor_indexes = np.where(
            censor_indexes == self.area_times.size + 1,
            censor_indexes - 1,
            censor_indexes,
        )
        censor_area = (
            self.area_times[censor_indexes] - censor_times
        ) * self.area_probabilities[censor_indexes - 1]
        censor_area += self.area[censor_indexes]
        return censor_times + censor_area / surv_prob
