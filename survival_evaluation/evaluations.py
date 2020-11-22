from typing import List, Union

import numpy as np  # type: ignore


def equal_counter(sorted_times: List[float]) -> np.array:
    equal_vec = [t == sorted_times[idx + 1] for idx, t in enumerate(sorted_times[:-1])]
    equal_count = list()
    count = 0

    for i in reversed(equal_vec):
        count = (count * i) + i
        equal_count.append(count)

    return np.array(list(reversed(equal_count)))


# pylint: disable=too-many-locals
def concordance(
    event_times: np.array,
    event_indicator: np.array,
    risk_scores: np.array,
) -> Union[float, ValueError]:
    time_ind = np.argsort(-event_times)
    sorted_times = event_times[time_ind]
    sorted_risks = risk_scores[time_ind]
    sorted_indicators = event_indicator[time_ind]

    score = 0.0
    comparables = 0
    equal_times = equal_counter(sorted_times)
    equal_risks = equal_counter(sorted_risks)
    risk_tie_counter = np.sum(
        sorted_indicators[:-1] * np.maximum(0, (equal_risks - equal_times))
    )
    for idx, risk in np.ndenumerate(sorted_risks[:-1]):
        i = idx[0]
        if sorted_indicators[i] == 0:
            continue

        equal_count = equal_times[i]
        comparative_risks = sorted_risks[i + 1 + equal_count :]
        comparables += len(comparative_risks)
        score += np.sum((risk >= comparative_risks))
    if comparables == 0:
        return ValueError("Error: There must be at least one comparable pair.")
    return (score - 0.5 * risk_tie_counter) / comparables


if __name__ == "__main__":
    # x = np.array([5, 4, 3, 2, 2, 2, 1])
    # print(equal_counter(x))
    # c = concordance(
    #     np.array([15, 14, 10, 8, 7, 6, 5]),
    #     np.array([1, 0, 1, 1, 1, 0, 1]),
    #     np.array([10, 9, 8, 5, 5, 5, 5]),
    # )
    # print(c)
    import time

    start = time.time()
    c = concordance(
        np.random.rand(50000),
        np.repeat(1, 50000),
        np.random.rand(50000),
    )
    print(c)
    print(time.time() - start)
