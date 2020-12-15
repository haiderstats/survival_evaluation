# Survival Evaluation

## What is this?

A python package implementing the survival functions found in the paper [Effective Ways To Build and Evaluate Individual Survival Distributions](https://www.jmlr.org/papers/volume21/18-772/18-772.pdf) by Haider et al. Currently the package only supports the L1-Hinge, L1-Margin, One-Calibration, and D-Calibration evaluation metrics. Future iterations will likely include Concordance and the Brier Score. Note that this package is only for evaluations, all models and predictions must be made _prior_ to utilizing the functions found here. Below is an outline of how to use each of these evaluation metrics, note that the input will differ between all evaluation metrics.

### L1-Hinge and L1-Margin

These evaluation functions exist but aren't necessarily recommended as reducing an entire survival distribution to a single point is throwing away a lot of information. For more discussion on these metrics please reference Haider et al.

Below we use a dataset from lifelines and build a cox proportional hazards model. Then we use the `predict_expectation` function to get the expected survival time to use in the L1-Hinge and L1-Margin calculations. Note that the L1-Margin function requires us to pass in the training set as well because we have to build a Kaplan-Meier function from data not derived from the evaluation dataset.

```python
from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi

import numpy as np
import random
from survival_evaluation import d_calibration, l1, one_calibration

rossi = load_rossi()

# Mix up the dataframe and split it into train and test
np.random.seed(42)
rossi = rossi.sample(frac=1.0)

train = rossi.iloc[:300,:]
test = rossi.iloc[300:,:]

cph = CoxPHFitter()
cph.fit(train, duration_col='week', event_col='arrest')

# Get the expected survival time (in weeks).
survival_predictions = cph.predict_expectation(test)
print(l1(test.week, test.arrest, survival_predictions, l1_type = 'hinge'))

# Margin requires learning the Kaplan-Meier curve from a training set so we must supply that data here.
print(l1(test.week, test.arrest, survival_predictions,train.week,train.arrest, l1_type = 'margin'))

```

### One Calibration

One Calibration requires the survival probability at a specific time point so we can utilize the `predict_survival_function` function from lifelines and specify a specific time. Note the p-value (0.095) here suggests there is _not_ enough evidence to support the model is _not_ one-calibrated (a p-value below 0.05 suggests the model is not one-calibrated).

```python
survival_probabilities = cph.predict_survival_function(test, times=25)
print(one_calibration(test.week, test.arrest, survival_probabilities.iloc[0,:], time= 25))
```

### D-Calibration

D-Calibration requires the survival probability at the event time (or censor time). To accomplish this we use the `predict_survival_function` function from lifelines and get a bit ugly. We want the survival probability of each survival curve at the time the observation either had their event or were censored. To do this we do a list comprehension over the test dataset and grab each prediction. There is probably a much better way to do this so I encourage you to figure that out and then put in a pull request! Note the p-value from D-Calibration gives 0.989 which suggests there is _not_ enough evidence to support the model is _not_ d-calibrated (a p-value below 0.05 suggests the model is not d-calibrated).


```python
survival_probabilities = [cph.predict_survival_function(row, times=row.week).to_numpy()[0][0] for _, row in test.iterrows()]
print(d_calibration(test.arrest, survival_probabilities))
```
