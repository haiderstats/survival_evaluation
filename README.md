# Survival Evaluation

## What is this?

A python package implementing the survival functions found in the paper [Effective Ways To Build and Evaluate Individual Survival Distributions](https://www.jmlr.org/papers/volume21/18-772/18-772.pdf) by Haider et al. Currently the package only supports the L1-Hinge, L1-Margin, One-Calibration, and D-Calibration evaluation metrics. Future iterations will likely include Concordance and the Brier Score. Note that this package is only for evaluations, all models and predictions must be made _prior_ to utilizing the functions found here. Below is an outline of how to use each of these evaluation metrics, note that the input will differ between all evaluation metrics.

### L1-Hinge and L1-Margin

```python
from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi

import numpy as np
import random
from survival_evaluation import l1

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

```python
survival_probabilities = cph.predict_survival_function(test, times=25)
one_calibration(test.week, test.arrest, survival_probabilities.iloc[0,:], time= 25)
```

### D-Calibration

```python
survival_probabilities = cph.predict_survival_function(test, times=test.week)
d_calibration(test.arrest, survival_probabilities.iloc[0,:])
```
