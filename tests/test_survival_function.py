import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.testing import all_survival_estimators


def all_survival_function_estimators():
    estimators = set()
    for cls in all_survival_estimators():
        if hasattr(cls, "predict_survival_function"):
            if issubclass(cls, CoxnetSurvivalAnalysis):
                est = cls(fit_baseline_model=True)
            else:
                est = cls()
            estimators.add(est)
    return estimators


@pytest.mark.parametrize("estimator", all_survival_function_estimators())
def test_survival_functions(estimator, make_whas500):
    data = make_whas500(to_numeric=True)

    estimator.fit(data.x[150:], data.y[150:])
    fns_cls = estimator.predict_survival_function(data.x[:150])
    fns_arr = estimator.predict_survival_function(data.x[:150], return_array=True)

    times = estimator.event_times_
    arr = np.row_stack([fn(times) for fn in fns_cls])

    assert_array_almost_equal(arr, fns_arr)
