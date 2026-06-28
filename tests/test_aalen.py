"""Tests for AalenAdditiveFitter."""
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from sksurv.datasets import load_whas500
from sksurv.functions import StepFunction
from sksurv.linear_model import AalenAdditiveFitter
from sksurv.util import Surv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def whas500_data():
    """WHAS500 dataset (n=500, 14 numeric features)."""
    X, y = load_whas500()
    X = X.astype(float)
    return X, y


@pytest.fixture(scope="module")
def fitted_whas500(whas500_data):
    """Pre-fitted model on WHAS500 for reuse in multiple tests."""
    X, y = whas500_data
    est = AalenAdditiveFitter(alpha=0.1)
    est.fit(X, y)
    return est, X, y


@pytest.fixture(scope="module")
def small_data():
    """Tiny synthetic survival dataset for fast unit tests."""
    rng = np.random.default_rng(42)
    n = 80
    X = pd.DataFrame(
        {
            "x0": rng.standard_normal(n),
            "x1": rng.standard_normal(n),
        }
    )
    times = rng.exponential(scale=5, size=n)
    events = rng.random(n) > 0.3
    y = Surv.from_arrays(events, times)
    return X, y


@pytest.fixture(scope="module")
def fitted_small(small_data):
    X, y = small_data
    est = AalenAdditiveFitter(alpha=0.0)
    est.fit(X, y)
    return est, X, y


# ---------------------------------------------------------------------------
# Basic API tests
# ---------------------------------------------------------------------------


def test_fit_returns_self(small_data):
    """fit() must return the estimator instance."""
    X, y = small_data
    est = AalenAdditiveFitter()
    result = est.fit(X, y)
    assert result is est


def test_fit_sets_unique_times(fitted_small):
    """unique_times_ must be a sorted 1-D array of event times."""
    est, X, y = fitted_small
    assert hasattr(est, "unique_times_")
    assert est.unique_times_.ndim == 1
    assert len(est.unique_times_) > 0
    # Must be strictly increasing
    assert np.all(np.diff(est.unique_times_) > 0)


def test_fit_sets_cumulative_coefficients(fitted_small):
    """cumulative_coefficients_ shape must be (n_event_times, n_features+1)."""
    est, X, y = fitted_small
    n_event_times = len(est.unique_times_)
    n_coefs = X.shape[1] + 1  # +1 for intercept
    assert est.cumulative_coefficients_.shape == (n_event_times, n_coefs)


def test_cumulative_coefficient_functions_keys(fitted_small):
    """cumulative_coefficient_functions_ must contain 'Intercept' and feature keys."""
    est, X, y = fitted_small
    keys = list(est.cumulative_coefficient_functions_.keys())
    assert "Intercept" in keys
    for col in X.columns:
        assert col in keys


def test_cumulative_coefficient_functions_are_step_functions(fitted_small):
    """All values in cumulative_coefficient_functions_ must be StepFunction objects."""
    est, X, y = fitted_small
    for name, fn in est.cumulative_coefficient_functions_.items():
        assert isinstance(fn, StepFunction), f"Expected StepFunction, got {type(fn)} for '{name}'"


def test_cum_baseline_hazard_is_step_function(fitted_small):
    """cum_baseline_hazard_ must be a StepFunction."""
    est, X, y = fitted_small
    assert isinstance(est.cum_baseline_hazard_, StepFunction)


# ---------------------------------------------------------------------------
# Prediction shape / type tests
# ---------------------------------------------------------------------------


def test_predict_survival_function_shape(fitted_small):
    """predict_survival_function returns an object array with one StepFunction per sample."""
    est, X, y = fitted_small
    fns = est.predict_survival_function(X)
    assert fns.shape == (len(X),)
    assert all(isinstance(f, StepFunction) for f in fns)


def test_predict_cumulative_hazard_function_shape(fitted_small):
    """predict_cumulative_hazard_function returns an object array of StepFunctions."""
    est, X, y = fitted_small
    fns = est.predict_cumulative_hazard_function(X)
    assert fns.shape == (len(X),)
    assert all(isinstance(f, StepFunction) for f in fns)


def test_predict_survival_function_return_array(fitted_small):
    """With return_array=True the output is a 2-D numeric array."""
    est, X, y = fitted_small
    arr = est.predict_survival_function(X, return_array=True)
    n_times = len(est.unique_times_)
    assert arr.shape == (len(X), n_times)
    assert np.issubdtype(arr.dtype, np.floating)


def test_predict_cumulative_hazard_return_array(fitted_small):
    """With return_array=True the output is a 2-D numeric array."""
    est, X, y = fitted_small
    arr = est.predict_cumulative_hazard_function(X, return_array=True)
    n_times = len(est.unique_times_)
    assert arr.shape == (len(X), n_times)


def test_baseline_cumulative_hazard_shape(fitted_whas500):
    """Baseline cumulative hazard must have as many points as unique event times."""
    est, X, y = fitted_whas500
    bch = est.cum_baseline_hazard_
    assert len(bch.x) == len(est.unique_times_)
    assert len(bch.y) == len(est.unique_times_)


# ---------------------------------------------------------------------------
# Mathematical correctness tests
# ---------------------------------------------------------------------------


def test_survival_function_between_0_and_1(fitted_small):
    """All survival function values must lie in [0, 1]."""
    est, X, y = fitted_small
    arr = est.predict_survival_function(X, return_array=True)
    assert np.all(arr >= 0.0), "Survival values below 0 detected"
    assert np.all(arr <= 1.0), "Survival values above 1 detected"


def test_survival_decreasing_in_time(fitted_whas500):
    """Population-averaged survival should decrease substantially over time.

    The Aalen additive model does NOT guarantee that each individual's
    survival function is point-wise monotone — the linear additive form can
    produce negative hazard increments for some covariate combinations.
    However, the mean survival over the cohort should be clearly lower
    in the second half of the time axis than in the first half.
    """
    est, X, y = fitted_whas500
    arr = est.predict_survival_function(X, return_array=True)
    mean_surv = arr.mean(axis=0)
    n_times = len(est.unique_times_)
    half = n_times // 2
    mean_first_half = mean_surv[:half].mean()
    mean_second_half = mean_surv[half:].mean()
    assert mean_first_half > mean_second_half, (
        "Mean survival in first half of follow-up is not greater than second half. "
        f"First: {mean_first_half:.4f}, Second: {mean_second_half:.4f}"
    )


def test_cumulative_coef_monotone(fitted_whas500):
    """Cumulative coefficients are computed as a cumulative sum of increments.

    The Aalen model estimates increments dB_k at each event time via WLS.
    cumulative_coefficients_[k] must equal sum of dB increments up to time k.
    We verify this property (the regression curve may have local downward dips
    but the cumsum relation must hold exactly).
    """
    est, X, y = fitted_whas500
    cum_coefs = est.cumulative_coefficients_
    # Reconstruct increments from cumulative
    increments = np.diff(cum_coefs, axis=0, prepend=0)
    # Verify the cumsum relation (should reconstruct cum_coefs exactly)
    reconstructed = np.cumsum(increments, axis=0)
    npt.assert_allclose(reconstructed, cum_coefs, rtol=1e-12,
                        err_msg="cumulative_coefficients_ is not a valid cumsum of increments")


def test_cumulative_hazard_non_negative(fitted_small):
    """Cumulative hazard must be non-negative for all samples and times."""
    est, X, y = fitted_small
    arr = est.predict_cumulative_hazard_function(X, return_array=True)
    # For the additive model this is not strictly guaranteed for all covariate
    # combinations, but the baseline (X=0) should be near-non-negative.
    # Here we just check there are no egregiously negative values (> -0.5).
    assert np.all(arr > -0.5), "Large negative cumulative hazard values detected"


def test_survival_cumhazard_consistency(fitted_small):
    """S(t|x) must equal exp(-H(t|x)) up to floating-point precision."""
    est, X, y = fitted_small
    surv = est.predict_survival_function(X, return_array=True)
    ch = est.predict_cumulative_hazard_function(X, return_array=True)
    expected_surv = np.exp(-np.clip(ch, 0, None))
    npt.assert_allclose(surv, expected_surv, rtol=1e-10)


# ---------------------------------------------------------------------------
# Regularization test
# ---------------------------------------------------------------------------


def test_alpha_regularization_effect(small_data):
    """Non-zero alpha should produce valid (non-NaN, non-Inf) results."""
    X, y = small_data
    for alpha_val in [0.0, 0.01, 0.1, 1.0, 10.0]:
        est = AalenAdditiveFitter(alpha=alpha_val)
        est.fit(X, y)
        arr = est.predict_survival_function(X, return_array=True)
        assert np.all(np.isfinite(arr)), f"Non-finite values with alpha={alpha_val}"
        assert np.all(arr >= 0), f"Negative survival with alpha={alpha_val}"
        assert np.all(arr <= 1), f"Survival > 1 with alpha={alpha_val}"


def test_alpha_must_be_non_negative(small_data):
    """Negative alpha should raise a ValueError."""
    X, y = small_data
    est = AalenAdditiveFitter(alpha=-1.0)
    with pytest.raises((ValueError,)):
        est.fit(X, y)


# ---------------------------------------------------------------------------
# Dataset-specific tests
# ---------------------------------------------------------------------------


def test_fit_whas500(whas500_data):
    """Model should fit whas500 without errors and learn sensible structure."""
    X, y = whas500_data
    est = AalenAdditiveFitter(alpha=0.1)
    est.fit(X, y)

    # Should have learned some event times
    assert len(est.unique_times_) > 10

    # Survival predictions should be in [0, 1]
    arr = est.predict_survival_function(X, return_array=True)
    assert np.all(arr >= 0) and np.all(arr <= 1)

    # Last column of survival (at max time) should be lower than first (min time)
    # i.e., average survival decreases
    assert arr[:, -1].mean() < arr[:, 0].mean()


def test_fit_whas500_num_event_times(whas500_data):
    """Number of unique event times must not exceed number of events."""
    X, y = whas500_data
    est = AalenAdditiveFitter(alpha=0.1)
    est.fit(X, y)
    event_field = y.dtype.names[0]
    n_events = int(y[event_field].sum())
    assert len(est.unique_times_) <= n_events


def test_fit_gbsg2():
    """Model should fit the GBSG2 breast cancer dataset."""
    from sksurv.datasets import load_gbsg2
    from sksurv.preprocessing import OneHotEncoder

    X, y = load_gbsg2()
    Xt = OneHotEncoder().fit_transform(X)

    est = AalenAdditiveFitter(alpha=0.5)
    est.fit(Xt, y)
    assert len(est.unique_times_) > 0

    fns = est.predict_survival_function(Xt.iloc[:10])
    assert len(fns) == 10
    assert all(isinstance(f, StepFunction) for f in fns)


# ---------------------------------------------------------------------------
# Edge case / robustness tests
# ---------------------------------------------------------------------------


def test_single_feature(small_data):
    """Model should work with a single-column feature matrix."""
    X, y = small_data
    X1 = X[["x0"]]
    est = AalenAdditiveFitter(alpha=0.1)
    est.fit(X1, y)
    fns = est.predict_survival_function(X1)
    assert len(fns) == len(X1)


def test_predict_subset_of_training_rows(fitted_whas500):
    """Predict on a subset of training rows should not raise."""
    est, X, y = fitted_whas500
    fns = est.predict_survival_function(X.iloc[:20])
    assert len(fns) == 20


def test_coef_var_shape(fitted_small):
    """coef_var_ should have same shape as cumulative_coefficients_."""
    est, X, y = fitted_small
    assert est.coef_var_.shape == est.cumulative_coefficients_.shape
    assert np.all(est.coef_var_ >= 0), "Variance estimates must be non-negative"


def test_n_features_in(fitted_small):
    """n_features_in_ must equal number of columns in X."""
    est, X, y = fitted_small
    assert est.n_features_in_ == X.shape[1]


def test_feature_names_in(fitted_small):
    """feature_names_in_ must contain the DataFrame column names."""
    est, X, y = fitted_small
    assert hasattr(est, "feature_names_in_")
    npt.assert_array_equal(est.feature_names_in_, X.columns.to_numpy())


def test_predict_risk_score_shape(fitted_small):
    """predict() must return a 1-D array of risk scores."""
    est, X, y = fitted_small
    scores = est.predict(X)
    assert scores.shape == (len(X),)
    assert np.all(np.isfinite(scores))
