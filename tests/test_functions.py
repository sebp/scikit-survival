import numpy as np
from numpy.testing import assert_array_equal
import pytest

from sksurv.functions import StepFunction
from sksurv.testing import all_survival_estimators


@pytest.fixture()
def a_step_function():
    x = np.array([0, 1, 1.2, 1.75, 2, 2.1, 3, 3.94, 5.4, 9])
    y = np.array([11, 9, 9.12, 7.5, 7.25, 5.14, 3, 2.94, 2.4, 1.9])
    f = StepFunction(x, y)
    return f


@pytest.fixture()
def toy_data_exponential():
    rnd = np.random.RandomState(2)
    n_samples = 100
    x = rnd.randn(n_samples, 2)
    y = np.empty(n_samples, dtype=[("event", bool), ("time", float)])
    y["time"] = rnd.exponential(scale=np.exp(x[:, 0]), size=n_samples)
    y["event"] = rnd.binomial(1, 0.5, size=n_samples) == 1

    # ensure at least 2 uncensored events exist
    y["event"][:2] = True

    # mark entry with largest time as censored
    # see https://github.com/sebp/scikit-survival/issues/249
    idxmax = np.argmax(y["time"])
    y["event"][idxmax] = False
    return x, y


class TestStepFunction:
    @staticmethod
    def test_exact(a_step_function):
        actual = np.array([a_step_function(v) for v in a_step_function.x])
        assert_array_equal(actual, a_step_function.y)

    @staticmethod
    def test_not_exact(a_step_function):
        z = np.diff(a_step_function.x).min() / 2
        actual = np.array([a_step_function(v + z) for v in a_step_function.x[:-1]])
        assert_array_equal(actual, a_step_function.y[:-1])

    @staticmethod
    @pytest.mark.parametrize("value", [-100, 100, -np.finfo(float).eps * 8, np.finfo(float).eps * 8])
    def test_out_of_bounds(a_step_function, value):
        v = value
        if v > 0:
            v += a_step_function.domain[1]

        with pytest.raises(ValueError, match=r"x must be within \[0\.0+; 9\.0+\]"):
            a_step_function(v)

    @staticmethod
    def test_not_finite(a_step_function, non_finite_value):
        with pytest.raises(ValueError, match="x must be finite"):
            a_step_function(non_finite_value)

    @staticmethod
    def test_equal(a_step_function):
        x = np.array([0, 1, 1.2, 1.75, 2, 2.1, 3, 3.94, 5.4, 9])
        y = np.array([11, 9, 9.12, 7.5, 7.25, 5.14, 3, 2.94, 2.4, 1.9])
        other_step_function = StepFunction(x, y)

        assert a_step_function == other_step_function

        different_step_function = StepFunction(x + 1, y)
        assert a_step_function != different_step_function

        assert a_step_function != x


@pytest.mark.parametrize(
    "estimator_cls", [est for est in all_survival_estimators() if hasattr(est, "predict_cumulative_hazard_function")]
)
def test_predict_cumulative_hazard_function_range(estimator_cls, toy_data_exponential):
    x, y = toy_data_exponential

    estimator = estimator_cls()
    if "fit_baseline_model" in estimator.get_params():
        estimator.set_params(fit_baseline_model=True)
    estimator.fit(x, y)

    t_min = y["time"].min()
    t_max = y["time"].max()
    t_mid = (t_max - t_min) / 2.0

    for fn in estimator.predict_cumulative_hazard_function(x):
        v = fn(t_min)
        assert np.isfinite(v)
        assert v == 0

    for fn in estimator.predict_cumulative_hazard_function(x):
        v = fn(t_mid)
        assert np.isfinite(v)
        assert v >= 0

    t_smaller_min = t_min / 2
    for fn in estimator.predict_cumulative_hazard_function(x):
        v = fn(t_smaller_min)
        assert np.isfinite(v)
        assert v == 0

    for fn in estimator.predict_cumulative_hazard_function(x):
        v = fn(t_max)
        assert np.isfinite(v)
        assert v >= 0

    t_bigger_max = t_max + 1
    for fn in estimator.predict_cumulative_hazard_function(x):
        with pytest.raises(ValueError, match=r"x must be within \[[0-9.]+; [0-9.]+\]"):
            fn(t_bigger_max)


@pytest.mark.parametrize(
    "estimator_cls", [est for est in all_survival_estimators() if hasattr(est, "predict_survival_function")]
)
def test_predict_survival_function_range(estimator_cls, toy_data_exponential):
    x, y = toy_data_exponential

    estimator = estimator_cls()
    if "fit_baseline_model" in estimator.get_params():
        estimator.set_params(fit_baseline_model=True)
    estimator.fit(x, y)

    t_min = y["time"].min()
    t_max = y["time"].max()
    t_mid = (t_max - t_min) / 2.0

    for fn in estimator.predict_survival_function(x):
        v = fn(t_min)
        assert np.isfinite(v)
        assert v == 1

    for fn in estimator.predict_survival_function(x):
        v = fn(t_mid)
        assert np.isfinite(v)
        assert 0 <= v <= 1

    t_smaller_min = t_min / 2
    for fn in estimator.predict_survival_function(x):
        v = fn(t_smaller_min)
        assert np.isfinite(v)
        assert v == 1

    for fn in estimator.predict_survival_function(x):
        v = fn(t_max)
        assert np.isfinite(v)
        assert 0 <= v <= 1

    t_bigger_max = t_max + 1
    for fn in estimator.predict_survival_function(x):
        with pytest.raises(ValueError, match=r"x must be within \[[0-9.]+; [0-9.]+\]"):
            fn(t_bigger_max)
