import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.testing import all_survival_estimators
from sksurv.util import Surv


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

    times = estimator.unique_times_
    arr = np.vstack([fn(times) for fn in fns_cls])

    assert_array_almost_equal(arr, fns_arr)


@pytest.mark.parametrize("estimator", all_survival_function_estimators())
@pytest.mark.parametrize("y_time", [-1e-8, -1, np.finfo(float).min])
def test_fit_negative_survial_time_raises(estimator, y_time):
    X = np.random.randn(7, 3)
    y = Surv.from_arrays(event=np.ones(7, dtype=bool), time=[1, 9, 3, y_time, 1, 8, 1e10])

    with pytest.raises(ValueError, match="observed time contains values smaller zero"):
        estimator.fit(X, y)
