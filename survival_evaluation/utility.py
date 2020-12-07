import numpy as np  # type: ignore

from survival_evaluation.types import NumericArrayLike


def to_array(array_like: NumericArrayLike, to_boolean: bool = False) -> np.array:
    array = np.asarray(array_like)
    shape = np.shape(array)
    if len(shape) > 1:
        raise ValueError(
            f"Input should be a 1-d array. Got a shape of {shape} instead."
        )
    if to_boolean:
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
