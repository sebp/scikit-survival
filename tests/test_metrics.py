import os.path

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from sksurv.datasets import load_gbsg2
from sksurv.exceptions import NoComparablePairException
from sksurv.functions import StepFunction
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import (
    as_concordance_index_ipcw_scorer,
    as_cumulative_dynamic_auc_scorer,
    as_integrated_brier_score_scorer,
    brier_score,
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.preprocessing import OneHotEncoder
from sksurv.svm import FastSurvivalSVM
from sksurv.testing import FixtureParameterFactory, assert_cindex_almost_equal
from sksurv.util import Surv


def whas500_pred():
    WHAS500_DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "whas500_predictions.csv")

    dat = np.loadtxt(WHAS500_DATA_FILE, delimiter=",")
    event = dat[:, 0] == 1
    time = dat[:, 1]
    risk = dat[:, 2]
    return event, time, risk


@pytest.fixture()
def no_comparable_pairs():
    y = np.array(
        [
            (False, 849.0),
            (False, 28.0),
            (False, 55.0),
            (False, 727.0),
            (False, 505.0),
            (False, 1558.0),
            (False, 1292.0),
            (False, 1737.0),
            (False, 944.0),
            (False, 750.0),
            (False, 2513.0),
            (False, 472.0),
            (False, 2417.0),
            (False, 538.0),
            (False, 49.0),
            (False, 723.0),
            (True, 3563.0),
            (False, 1090.0),
            (False, 1167.0),
            (False, 587.0),
            (False, 1354.0),
            (False, 910.0),
            (False, 398.0),
            (False, 854.0),
            (False, 3534.0),
            (False, 280.0),
            (False, 183.0),
            (False, 883.0),
            (False, 32.0),
            (False, 144.0),
        ],
        dtype=[("event", bool), ("time", float)],
    )
    scores = np.random.randn(y.shape[0])
    return y, scores


class ConcordanceIndexCases(FixtureParameterFactory):
    @property
    def time(self):
        return [1, 5, 6, 11, 34, 45, 46, 50]

    @property
    def estimate(self):
        return [5, 8, 11, 34, 12, 3, 9, 12]

    @property
    def time_tied(self):
        return [1, 5, 6, 11, 11, 34, 45, 45, 50]

    @property
    def estimate_tied(self):
        return [5, 8, 11, 19, 34, 12, 3, 9, 12]

    def data_no_censoring_all_correct(self):
        time = self.time
        event = np.ones(len(time), dtype=bool)
        estimate = np.arange(len(time))[::-1]
        expected = [1.0, 28, 0, 0, 0]
        return event, time, estimate, expected

    def data_no_censoring_all_wrong(self):
        time = self.time
        event = np.ones(len(time), dtype=bool)
        # order is exactly reversed
        estimate = np.arange(len(time))
        expected = [0.0, 0, 28, 0, 0]
        return event, time, estimate, expected

    def data_no_ties(self):
        event = [False, True, True, False, False, True, False, False]
        expected = [0.2307692, 3, 10, 0, 0]
        return event, self.time, self.estimate, expected

    def data_with_tied_time(self):
        event = [False, True, True, False, True, False, True, False, False]
        expected = [0.4, 8, 12, 0, 2]
        return event, self.time_tied, self.estimate_tied, expected

    def data_with_tied_time2(self):
        event = [False, True, True, False, False, False, True, False, False]
        expected = [0.2, 3, 12, 0, 1]
        return event, self.time_tied, self.estimate_tied, expected

    def data_with_tied_event(self):
        event = [False, True, False, True, True, False, True, False, False]
        expected = [0.5294118, 9, 8, 0, 1]
        return event, self.time_tied, self.estimate_tied, expected

    def data_with_tied_risk(self):
        event = [False, True, True, False, True, True, False, False]
        estimate = [5, 15, 11, 34, 12, 3, 9, 12]
        expected = [0.59375, 9, 6, 1, 0]
        return event, self.time, estimate, expected

    def data_with_almost_tied_risk(self):
        event, time, estimate, expected = self.data_with_tied_risk()
        estimate[4] += 4.5e-9
        estimate[-1] -= 4.5e-9
        return event, time, estimate, expected

    def data_with_tied_event_and_time(self):
        event = [True, False, False, False, True, False, True, True, False, False, False, True]
        time = [34, 11, 11, 5, 1, 89, 13, 45, 7, 13, 9, 13]
        estimate = [1, 19, 13, 13, 15, 14, 19, 23, 11, 10, 11, 1]

        expected = [0.5681818, 12, 9, 1, 2]
        return event, time, estimate, expected

    def data_whas500(self):
        event, time, estimate = whas500_pred()
        expected = [0.7697907, 57849, 17300, 0, 14]
        return event, time, estimate, expected


@pytest.mark.parametrize("event,time,estimate,expected", ConcordanceIndexCases().get_cases())
def test_concordance_index(event, time, estimate, expected):
    assert_cindex_almost_equal(event, time, estimate, expected)


class ConcordanceIndexFailureCases(FixtureParameterFactory):
    @property
    def event(self):
        return np.array([True, False, False, True, True, False])

    @property
    def time(self):
        return np.array([1, 5, 10, 12, 7, 65])

    @property
    def estimate(self):
        return np.array([12, 8, 1, 89, 56, 13])

    @property
    def error_different_length(self):
        return r"Found input variables with inconsistent numbers of samples: .+"

    def data_different_length_0(self):
        return (self.event, self.time[:3], self.estimate), self.error_different_length

    def data_different_length_1(self):
        return (self.event, self.time, self.estimate[:3]), self.error_different_length

    def data_different_length_2(self):
        return (self.event[:3], self.time, self.estimate), self.error_different_length

    def data_different_length_3(self):
        return (self.event, self.time[:3], self.estimate[:3]), self.error_different_length

    def data_different_length_4(self):
        return (self.event[:3], self.time, self.estimate[:3]), self.error_different_length

    def data_different_length_5(self):
        return (self.event[:3], self.time[:3], self.estimate), self.error_different_length

    def data_boolean_event(self):
        event = np.array([1, 0, 0, 1, 1, 0])

        match = "only boolean arrays are supported as class labels for survival analysis.+"
        return (event, self.time, self.estimate), match

    def data_min_samples(self):
        event = np.array([False])
        time = np.array([10])
        estimate = np.array([12])

        match = "Need a minimum of two samples"
        return (event, time, estimate), match

    def data_all_censored(self):
        event = np.array([False, False])
        time = np.array([10, 12])
        estimate = np.array([12, 13])

        match = "All samples are censored"
        return (event, time, estimate), match

    def data_all_finite_event_indicator(self):
        event = self.event.tolist()
        event[2] = None
        event = np.array(event)

        match = "Input event_indicator contains NaN"
        return (event, self.time, self.estimate), match

    def data_all_finite_event_time(self):
        time = self.time.tolist()
        time[3] = np.nan
        time = np.array(time)

        match = "Input event_time contains NaN"
        return (self.event, time, self.estimate), match

    def data_all_finite_estimate(self):
        estimate = self.estimate.tolist()
        estimate[5] = np.inf
        estimate = np.array(estimate)

        match = "Input estimate contains infinity or a value too large"
        return (self.event, self.time, estimate), match

    def _get_not_1d(self, dim):
        event, time, risk = whas500_pred()

        risk = np.tile(risk[:, np.newaxis], (1, dim))

        match = "Expected 1D array, got 2D array instead:"
        return (event, time, risk), match

    def data_not_1d_dim2(self):
        return self._get_not_1d(2)

    def data_not_1d_dim3(self):
        return self._get_not_1d(3)

    def data_not_1d_dim10(self):
        return self._get_not_1d(10)


@pytest.mark.parametrize("inputs,match", ConcordanceIndexFailureCases().get_cases())
def test_concordance_index_failure(inputs, match):
    with pytest.raises(ValueError, match=match):
        concordance_index_censored(*inputs)


def test_concordance_index_no_comparable(no_comparable_pairs):
    y, scores = no_comparable_pairs

    with pytest.raises(NoComparablePairException):
        concordance_index_censored(y["event"], y["time"], scores)


class UnoCCases(FixtureParameterFactory):
    @property
    def y_train(self):
        return Surv.from_arrays(
            time=(2, 4, 6, 8, 10, 11, 15, 19),
            event=(False, True, False, True, False, False, False, False),
        )

    @property
    def y_ties(self):
        return Surv.from_arrays(
            event=np.array((0, 1, 1, 0, 1, 0, 1, 0, 0, 1), dtype=bool),
            time=(1, 5, 6, 10, 11, 34, 45, 46, 50, 56),
        )

    def data_no_ties(self):
        estimate = (5, 8, 11, 19, 34, 12, 3, 9, 12, 20)
        expected = (0.347890360332615, 8, 15, 0, 0)
        return self.y_ties, self.y_ties, estimate, expected, None

    def data_tied_risk_1(self):
        estimate = (5, 8, 11, 11, 34, 12, 3, 9, 12, 20)
        expected = (0.365629810028969, 8, 14, 1, 0)
        return self.y_ties, self.y_ties, estimate, expected, None

    def data_tied_risk_2(self):
        estimate = (5, 8, 11, 11, 34, 12, 11, 9, 12, 20)
        expected = (0.387865723332956, 7, 14, 2, 0)
        return self.y_ties, self.y_ties, estimate, expected, None

    def data_truncated_1(self):
        y_test = Surv.from_arrays(
            time=(1, 3, 5, 8, 12, 13),
            event=(True, False, False, True, True, True),
        )
        estimate = (5, 8, 13, 11, 9, 4)
        expected = (0.7543736528146774, 4, 4, 0, 0)
        tau = 19
        return self.y_train, y_test, estimate, expected, tau

    def data_truncated_2(self):
        y = Surv.from_arrays(
            time=(1, 5, 6, 10, 11, 34, 45, 46, 50, 56),
            event=np.array((0, 1, 1, 0, 1, 0, 1, 1, 1, 1), dtype=bool),
        )
        estimate = (5, 8, 11, 19, 34, 12, 3, 9, 12, 18)
        expected = (0.347890361949191, 8, 18, 0, 0)
        tau = 45.25
        return y, y, estimate, expected, tau

    def data_last_time_censored(self):
        y_test = Surv.from_arrays(
            time=(1, 3, 5, 7, 12, 13, 20),
            event=(True, False, False, True, True, False, False),
        )
        estimate = (5, 8, 13, 11, 9, 7, 4)
        expected = (0.8126567565914234, 6, 5, 0, 0)
        return self.y_train, y_test, estimate, expected, None

    def data_tied_event(self):
        y = Surv.from_arrays(
            event=[False, True, False, True, True, False, True, False, False, True],
            time=[1, 5, 6, 11, 11, 34, 45, 45, 50, 55],
        )
        estimate = (5, 8, 11, 19, 34, 12, 3, 9, 12, 18)
        expected = (0.4036321031048623, 11, 10, 0, 1)
        return y, y, estimate, expected, None

    def data_tied_event_and_time(self):
        y = Surv.from_arrays(
            event=[True, False, False, False, True, False, True, True, False, False, False, True, True],
            time=[34, 11, 11, 5, 1, 89, 13, 45, 7, 13, 9, 13, 90],
        )
        estimate = (1, 19, 13, 13, 15, 14, 19, 23, 11, 10, 11, 1, 18)
        expected = (0.46795357052737824, 14, 12, 1, 2)
        return y, y, estimate, expected, None

    def data_whas500(self):
        event, time, estimate = whas500_pred()
        y = Surv.from_arrays(event, time)
        expected = (0.7929275009049014, 57849, 17300, 0, 14)
        return y, y, estimate, expected, None


@pytest.mark.parametrize("y_train,y_test,estimate,expected,tau", UnoCCases().get_cases())
def test_uno_c(y_train, y_test, estimate, expected, tau):
    result = concordance_index_ipcw(y_train, y_test, estimate, tau=tau)
    assert_array_equal(result[1:], expected[1:])
    assert_almost_equal(result[0], expected[0])


class UnoCFailureCases(FixtureParameterFactory):
    @property
    def y_train(self):
        return Surv.from_arrays(
            time=(2, 4, 6, 8, 10, 11, 15, 19), event=(False, True, False, True, False, False, False, True)
        )

    @property
    def y_test(self):
        return Surv.from_arrays(time=(1, 3, 5, 7, 12, 13, 19), event=(True, False, False, True, True, False, True))

    @property
    def estimate(self):
        return (5, 8, 13, 11, 9, 7, 4)

    def data_last_time_uncensored_1(self):
        y_train = self.y_train
        y_test = self.y_test
        y_test["time"][-1] = 20
        match = "time must be smaller than largest observed time point:"

        inputs = (y_train, y_test, self.estimate)
        return inputs, match

    def data_last_time_uncensored_2(self):
        y_train = self.y_train
        y_test = Surv.from_arrays(time=(1, 23, 5, 27, 12), event=(True, False, True, True, False))
        estimate = (5, 13, 11, 9, 4)
        match = "time must be smaller than largest observed time point:"

        inputs = (y_train, y_test, estimate)
        return inputs, match

    def data_zero_prob_1(self):
        y_train = self.y_train
        y_train["event"][-1] = False
        match = "censoring survival function is zero at one or more time points"

        inputs = (y_train, self.y_test, self.estimate)
        return inputs, match

    def data_zero_prob_2(self):
        y_train = self.y_train
        y_train["time"][-1] = 18
        y_train["event"][-1] = False

        match = "censoring survival function is zero at one or more time points"

        inputs = (y_train, self.y_test, self.estimate)
        return inputs, match

    def data_zero_prob_3(self):
        y_train = self.y_train
        y_train["time"][-1] = 18
        y_train["event"][-1] = False

        y_test = self.y_test
        y_test["time"] = (1, 3, 5, 19, 12, 13, 7)
        match = "censoring survival function is zero at one or more time points"

        inputs = (y_train, y_test, self.estimate)
        return inputs, match

    def _get_not_1d(self, dim):
        event, time, risk = whas500_pred()
        y = Surv.from_arrays(event, time)

        risk = np.tile(risk[:, np.newaxis], (1, dim))

        match = "Expected 1D array, got 2D array instead:"
        inputs = (y, y, risk)
        return inputs, match

    def data_not_1d_dim2(self):
        return self._get_not_1d(2)

    def data_not_1d_dim3(self):
        return self._get_not_1d(3)

    def data_not_1d_dim10(self):
        return self._get_not_1d(10)


@pytest.mark.parametrize("inputs,expected_msg", UnoCFailureCases().get_cases())
def test_uno_c_failure(inputs, expected_msg):
    with pytest.raises(ValueError, match=expected_msg):
        concordance_index_ipcw(*inputs)


def test_uno_c_all_censored():
    y_train = Surv.from_arrays(
        time=(2, 4, 6, 8, 10, 11, 15, 19),
        event=(True, True, True, True, True, True, True, True),
    )
    y_test = Surv.from_arrays(
        time=(1, 3, 5, 7, 12, 13, 20),
        event=(True, False, False, True, True, False, False),
    )
    estimate = (5, 8, 13, 11, 9, 7, 4)

    ret_uno = concordance_index_ipcw(y_train, y_test, estimate)
    ret_harrell = concordance_index_censored(y_test["event"], y_test["time"], estimate)
    assert ret_uno == ret_harrell


def test_uno_c_no_comparable(no_comparable_pairs):
    y, scores = no_comparable_pairs

    with pytest.raises(NoComparablePairException):
        concordance_index_ipcw(y, y, scores)


class BaseUnoCAucCases(FixtureParameterFactory):
    @property
    def uno_auc_y(self):
        return Surv.from_arrays(
            time=[10.88, 19.78, 40.92, 98.7, 70.19, 10.15, 28.95, 29.57, 17.9, 63.78, 36.22, 83.14, 13.69, 99.51, 3.19],
            event=[1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1],
        )

    @property
    def uno_auc_data_15(self):
        estimate = [
            -1.019,
            -0.016,
            0.132,
            0.269,
            -0.777,
            -1.077,
            0.894,
            -1.227,
            -0.417,
            0.072,
            -1.275,
            -0.91,
            -0.825,
            -0.292,
            -0.045,
        ]
        return self.uno_auc_y, estimate

    @property
    def uno_auc_data_20(self):
        y_train = Surv.from_arrays(
            time=[
                77.6,
                57.6,
                66.6,
                67.0,
                31.5,
                5.5,
                67.4,
                43.7,
                31.7,
                71.9,
                81.1,
                56.2,
                88.1,
                2.9,
                62.0,
                17.2,
                88.0,
                26.4,
                93.5,
                79.9,
            ],
            event=[1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        )
        estimate = [
            -1.019,
            -0.016,
            0.132,
            0.269,
            -0.777,
            -1.077,
            0.894,
            -1.227,
            -0.417,
            0.072,
            -1.275,
            -0.91,
            -0.825,
            -0.292,
            -0.045,
        ]
        return y_train, self.uno_auc_y, estimate

    @property
    def uno_auc_time_dependent_20(self):
        y_train, y_test, _ = self.uno_auc_data_20

        estimate = np.array(
            [
                [
                    0.566,
                    0.576,
                    0.506,
                    1.871,
                    -0.04,
                    0.281,
                    0.335,
                    -1.181,
                    1.368,
                    -0.192,
                    0.955,
                    0.221,
                    0.057,
                    -0.996,
                    -1.27,
                ],
                [
                    -2.658,
                    -1.121,
                    1.903,
                    -0.7,
                    -1.013,
                    -0.472,
                    -0.668,
                    -0.537,
                    1.659,
                    0.657,
                    -1.317,
                    -1.103,
                    2.159,
                    0.625,
                    -0.067,
                ],
                [
                    -1.004,
                    -1.67,
                    0.775,
                    -2.512,
                    1.179,
                    0.073,
                    0.103,
                    -0.379,
                    -0.082,
                    -0.617,
                    1.287,
                    -0.449,
                    -0.477,
                    0.24,
                    0.838,
                ],
            ]
        ).T

        return y_train, y_test, estimate


class UnoCAucCases(BaseUnoCAucCases):
    def data_single_time(self):
        y_train, estimate = self.uno_auc_data_15
        times = 28
        iauc = 0.362963
        expected = np.array([iauc])

        return y_train, y_train, estimate, times, expected, iauc

    def data_two_times(self):
        y_train, estimate = self.uno_auc_data_15
        times = [15, 66]
        iauc = 0.3943949
        expected = np.array([0.3030303, 0.4500000])

        return y_train, y_train, estimate, times, expected, iauc

    def data_two_times_int(self):
        y_train, estimate = self.uno_auc_data_15
        y_train["time"] = y_train["time"] * 100.0
        y_train = y_train.astype([("event", bool), ("time", int)])
        estimate = (np.array(estimate) * 1000).astype(int)
        times = np.array([1500, 6600], dtype=int)
        iauc = 0.3943949
        expected = np.array([0.3030303, 0.4500000])

        return y_train, y_train, estimate, times, expected, iauc

    def data_min_to_max_times(self):
        y_train, estimate = self.uno_auc_data_15
        times = (y_train["time"].min(), 15, 66, y_train["time"].max() - 1e-6)
        iauc = 0.3999539
        expected = np.array([0.6428571, 0.3030303, 0.4500000, 0.3162996])

        return y_train, y_train, estimate, times, expected, iauc

    def data_train_test(self):
        y_train, y_test, estimate = self.uno_auc_data_20
        times = [15, 66]
        iauc = 0.385509
        expected = np.array([0.3030303, 0.4357061])

        return y_train, y_test, estimate, times, expected, iauc

    def data_tied_test_time(self):
        y_train, y_test, estimate = self.uno_auc_data_20
        y_test["time"][0] = y_test["time"][1]
        times = [15, 66]
        iauc = 0.4204885
        expected = np.array([0.3750000, 0.4357061])

        return y_train, y_test, estimate, times, expected, iauc

    def data_tied_test_score(self):
        y_train, y_test, estimate = self.uno_auc_data_20
        estimate[0] = estimate[-1]
        times = [15, 66]
        iauc = 0.495604291
        expected = np.array([0.4242424, 0.539036])

        return y_train, y_test, estimate, times, expected, iauc

    def data_tied_test_score_max(self):
        y_train, y_test, estimate = self.uno_auc_data_20
        o = np.argsort(estimate)
        estimate[o[0]] = estimate[o[1]]
        times = [15, 66]
        iauc = 0.385509
        expected = np.array([0.3030303, 0.4357061])

        return y_train, y_test, estimate, times, expected, iauc

    def data_tied_test_score_min(self):
        y_train, y_test, estimate = self.uno_auc_data_20
        o = np.argsort(estimate)
        estimate[o[-1]] = estimate[o[-2]]
        times = [15, 66]
        iauc = 0.374134
        expected = np.array([0.3030303, 0.4174082])

        return y_train, y_test, estimate, times, expected, iauc

    def data_time_dependent(self):
        y_train, y_test, estimate = self.uno_auc_time_dependent_20
        times = [15, 30, 72]
        iauc = 0.5522067
        expected = np.array([0.3636364, 0.5247813, 0.7603000])

        return y_train, y_test, estimate, times, expected, iauc

    def _compute_roc_auc(self, y, times, estimate):
        expected_auc = np.array([roc_auc_score(y["time"] > t, e) for t, e in zip(times, estimate)])
        km_delta = np.array([1 - 0.8, 0.8 - 0.5, 0.5 - 0.2])
        expected_iauc = np.sum(km_delta * expected_auc) / 0.8

        return expected_auc, expected_iauc

    @property
    def y_roc_auc(self):
        return Surv.from_arrays(
            time=[7, 9, 11, 12, 13, 15, 28, 39, 41, 76],
            event=[True, True, True, True, True, True, True, True, True, True],
        )

    @property
    def times_roc_auc(self):
        return [10, 14, 40]

    def data_time_dependent_without_censoring(self):
        estimate = np.array(
            [
                [1, 6, 18, 56, 32, 3, 99, 7, 67, 541],
                [6, 9, 11, 5, 3, 12, 56, 56.1, 81, 77],
                [13, 11, 12, 76, 55, 134, 70, 78, 75, 99],
            ]
        )
        y = self.y_roc_auc
        times = self.times_roc_auc
        expected_auc, expected_iauc = self._compute_roc_auc(y, times, estimate)

        return y, y, -estimate.T, times, expected_auc, expected_iauc

    def data_time_dependent_with_ties_without_censoring(self):
        estimate = np.array(
            [
                [1, 6, 7, 56, 32, 3, 99, 7, 79, 17],
                [3, 6, 11, 5, 17, 12, 17, 56.1, 81, 77],
                [13, 11, 12, 17, 17, 134, 70, 78, 13, 99],
            ]
        )
        y = self.y_roc_auc
        times = self.times_roc_auc
        expected_auc, expected_iauc = self._compute_roc_auc(y, times, estimate)

        return y, y, -estimate.T, times, expected_auc, expected_iauc

    def data_whas500(self):
        event, time, estimate = whas500_pred()
        y_train = Surv.from_arrays(event=event[:300], time=time[:300])
        y_test = Surv.from_arrays(event=event[300:], time=time[300:])
        estimate = estimate[300:]
        times = (200, 400, 600, 800, 1000, 1200, 1400)

        iauc = 0.8045058
        expected = np.array([0.7720669, 0.7765915, 0.7962623, 0.8759295, 0.8759295, 0.8759513, 0.9147647])
        return y_train, y_test, estimate, times, expected, iauc

    def data_whas500_unordered_time(self):
        y_train, y_test, estimate, _, expected, iauc = self.data_whas500()
        times = (1000, 600, 1400, 200, 400, 1200, 800, 1000, 200)
        return y_train, y_test, estimate, times, expected, iauc


@pytest.mark.parametrize("y_train,y_test,estimate,times,expect_auc,expect_iauc", UnoCAucCases().get_cases())
def test_uno_auc(y_train, y_test, estimate, times, expect_auc, expect_iauc):
    auc, iauc = cumulative_dynamic_auc(y_train, y_test, estimate, times)
    assert_array_almost_equal(auc, expect_auc)
    if isinstance(expect_iauc, np.ndarray):
        assert_almost_equal(iauc, expect_iauc)
    else:
        assert iauc == pytest.approx(expect_iauc)


class UnoCAucFailureCases(BaseUnoCAucCases):
    @property
    def times(self):
        return 33

    def data_time_too_big_1(self):
        y_train, y_test, _ = self.uno_auc_data_20
        y_test["time"][11] = 100
        match = "time must be smaller than largest observed time point:"

        return y_train, y_test, self.times, match

    def data__time_too_big_2(self):
        y_train, y_test, _ = self.uno_auc_data_20
        idx = np.argmax(y_test["time"])
        y_test["event"][idx] = True
        match = "time must be smaller than largest observed time point:"

        return y_train, y_test, self.times, match

    def data_ipcw_undefined_1(self):
        y_train, y_test, _ = self.uno_auc_data_20
        idx = np.argmax(y_train["time"])
        y_train["event"][idx] = 0
        y_test["time"][0] = y_train["time"][idx]
        match = "censoring survival function is zero at one or more time points"

        return y_train, y_test, self.times, match

    def data_ipcw_undefined_2(self):
        y_train, y_test, _ = self.uno_auc_data_20
        idx = np.argmax(y_train["time"])
        y_train["event"][idx] = 0
        y_test["time"][-1] = y_train["time"][idx] * 2
        y_test["event"][-1] = True
        match = "censoring survival function is zero at one or more time points"

        return y_train, y_test, self.times, match

    def data_all_censored_train(self):
        y_train, y_test, _ = self.uno_auc_data_20
        y_train["event"] = False
        match = "all samples are censored"

        return y_train, y_test, self.times, match

    def data_all_censored_test(self):
        y_train, y_test, _ = self.uno_auc_data_20
        y_test["event"] = False
        match = "all samples are censored"

        return y_train, y_test, self.times, match

    def data_times_nan(self):
        y_train, y_test, _ = self.uno_auc_data_20
        times = (0.2, np.nan)
        match = r"Input times contains NaN"

        return y_train, y_test, times, match

    def data_times_infinite_1(self):
        y_train, y_test, _ = self.uno_auc_data_20
        times = (0.2, np.inf)
        match = r"Input times contains infinity or a value too large for dtype\('float64'\)."

        return y_train, y_test, times, match

    def data_times_infinite_2(self):
        y_train, y_test, _ = self.uno_auc_data_20
        times = (0.2, -np.inf)
        match = r"Input times contains infinity or a value too large for dtype\('float64'\)."

        return y_train, y_test, times, match

    def data_times_too_big_1(self):
        y_train, y_test, _ = self.uno_auc_data_20
        idx = np.argmax(y_test["time"])
        y_test["event"][idx] = False
        t_max = y_test["time"][idx]
        times = (33, t_max / 2, t_max)
        match = r"all times must be within follow-up time of test data: \[3\.19; 99\.51\["

        return y_train, y_test, times, match

    def data_times_too_big_2(self):
        y_train, y_test, times, match = self.data_times_too_big_1()
        times = times[:-1] + (times[-1] + 0.1,)
        match = r"all times must be within follow-up time of test data: \[3\.19; 99\.51\["

        return y_train, y_test, times, match

    def data_times_too_big_3(self):
        y_train, y_test, _ = self.uno_auc_data_20
        max_train = np.max(y_train["time"])
        idx_test = y_test["time"] > max_train
        y_test["event"][idx_test] = False
        y_test["time"][idx_test] = max_train
        times = (33, max_train)
        match = r"all times must be within follow-up time of test data: \[3\.19; 93\.5\["

        return y_train, y_test, times, match

    def data_times_too_small_1(self):
        y_train, y_test, _ = self.uno_auc_data_20
        idx = np.argmin(y_test["time"])
        y_test["event"][idx] = True
        t_min = y_test["time"][idx]
        times = (t_min - 1e-6, 33)
        match = r"all times must be within follow-up time of test data: \[3\.19; 99\.51\["

        return y_train, y_test, times, match

    def data_times_too_small_2(self):
        y_train, y_test, _ = self.uno_auc_data_20
        idx = np.argmin(y_test["time"])
        y_test["event"][idx] = True
        t_min = y_test["time"][idx]
        times = (33, t_min - 0.1, t_min / 2)
        match = r"all times must be within follow-up time of test data: \[3\.19; 99\.51\["

        return y_train, y_test, times, match

    def data_times_empty(self):
        y_train, y_test, _ = self.uno_auc_data_20
        times = []
        match = r"Found array with 0 sample\(s\)"

        return y_train, y_test, times, match


@pytest.mark.parametrize("y_train,y_test,times,match", UnoCAucFailureCases().get_cases())
def test_uno_auc_failure(y_train, y_test, times, match):
    estimate = np.random.randn(y_test.shape[0])
    with pytest.raises(ValueError, match=match):
        cumulative_dynamic_auc(y_train, y_test, estimate, times)


class UnoAucShapeFailureCases(BaseUnoCAucCases):
    @property
    def times(self):
        return 33

    def data_estimate_2d_1col(self):
        y_train, y_test, estimate = self.uno_auc_data_20
        estimate = np.atleast_2d(estimate)
        match = r"Found input variables with inconsistent numbers of samples: \[15, 1\]"

        return y_train, y_test, self.times, estimate, match

    def data_estimate_2d_2cols(self):
        y_train, y_test, estimate = self.uno_auc_data_20
        times = [11, 33, 55]
        estimate = np.tile(estimate, (2, 1)).T
        match = "expected estimate with 3 columns, but got 2"

        return y_train, y_test, times, estimate, match

    def data_estimate_2d_4cols(self):
        y_train, y_test, estimate = self.uno_auc_data_20
        times = [11, 33, 55]
        estimate = np.tile(estimate, (4, 1)).T
        match = "expected estimate with 3 columns, but got 4"

        return y_train, y_test, times, estimate, match

    def data_estimate_3d(self):
        y_train, y_test, estimate = self.uno_auc_data_20
        estimate = np.atleast_3d(estimate)
        match = "Found array with dim 3. cumulative_dynamic_auc expected <= 2."

        return y_train, y_test, self.times, estimate, match


@pytest.mark.parametrize("y_train,y_test,times,estimate,match", UnoAucShapeFailureCases().get_cases())
def test_uno_auc_shape_failure(y_train, y_test, times, estimate, match):
    with pytest.raises(ValueError, match=match):
        cumulative_dynamic_auc(y_train, y_test, estimate, times)


@pytest.fixture()
def nottingham_prognostic_index():
    def _get_npi(times):
        X, y = load_gbsg2()

        grade = X.loc[:, "tgrade"].map({"I": 1, "II": 2, "III": 3}).astype(int)
        NPI = 0.2 * X.loc[:, "tsize"] / 10 + 1 + grade
        NPI[NPI < 3.4] = 1.0
        NPI[(NPI >= 3.4) & (NPI <= 5.4)] = 2.0
        NPI[NPI > 5.4] = 3.0

        preds = np.empty((X.shape[0], len(times)), dtype=float)
        for j, ts in enumerate(times):
            survs = {}
            for i in NPI.unique():
                idx = np.flatnonzero(NPI == i)
                yi = y[idx]
                t, s = kaplan_meier_estimator(yi["cens"], yi["time"])
                if t[-1] < ts and s[-1] == 0.0:
                    survs[i] = 0.0
                else:
                    fn = StepFunction(t, s)
                    survs[i] = fn(ts)

            preds[:, j] = NPI.map(survs).values

        return preds, y

    return _get_npi


@pytest.fixture(params=[365, 730, 1095, 1460, 1825])
def brier_npi_data(request, nottingham_prognostic_index):
    t = request.param

    pred, y = nottingham_prognostic_index([t])

    if t == 365:
        bs = 0.0762922458520448
    elif t == 730:
        bs = 0.182536421174199
    elif t == 1095:
        bs = 0.220017747254941
    elif t == 1460:
        bs = 0.234133800146671
    elif t == 1825:
        bs = 0.233822955042198
    else:
        raise AssertionError()

    return pred, y, t, bs


def test_brier_nottingham(brier_npi_data):
    pred, y, times, expected_score = brier_npi_data

    _, score = brier_score(y, y, pred.squeeze(), times=times)
    assert score[0] == pytest.approx(expected_score)


def test_brier_nottingham_many(nottingham_prognostic_index):
    times = [365, 730, 1095, 1460, 1825]
    pred, y = nottingham_prognostic_index(times)

    expected_score = np.array(
        [
            0.0762922458520448,
            0.182536421174199,
            0.220017747254941,
            0.234133800146671,
            0.233822955042198,
        ]
    )

    t1, score = brier_score(y, y, pred.squeeze(), times=times)
    assert_array_almost_equal(score, expected_score)

    t2, score = brier_score(y, y, pred.squeeze(), times=times[::-1])
    assert_array_almost_equal(score, expected_score)
    assert_array_equal(t1, t2)


def test_brier_times_too_large(nottingham_prognostic_index):
    pred, y = nottingham_prognostic_index([1825])

    with pytest.raises(ValueError, match="all times must be within follow-up time of test data:"):
        brier_score(y, y, pred, times=9999)


def test_brier_wrong_estimate_shape(nottingham_prognostic_index):
    pred, y = nottingham_prognostic_index([720, 1825])

    with pytest.raises(ValueError, match="expected estimate with 2 columns, but got 1"):
        brier_score(y, y, pred[:, :1], times=[720, 1825])

    with pytest.raises(ValueError, match="expected estimate with 3 columns, but got 2"):
        brier_score(y, y, pred, times=[720, 960, 1825])

    with pytest.raises(ValueError, match=r"Found input variables with inconsistent numbers of samples: \[686, 10\]"):
        brier_score(y, y, pred[:10], times=[720, 1825])


def test_brier_coxph():
    X, y = load_gbsg2()
    X["tgrade"] = X.loc[:, "tgrade"].map(len).astype(int)

    Xt = OneHotEncoder().fit_transform(X)

    survs = CoxPHSurvivalAnalysis(ties="efron").fit(Xt, y).predict_survival_function(Xt)

    preds = [fn(1825) for fn in survs]

    _, score = brier_score(y, y, preds, 1825)

    assert score[0] == pytest.approx(0.208817407492645, 1e-5)


def test_brier_score_int_dtype():
    times = np.arange(1, 31, dtype=int)
    rnd = np.random.RandomState(1)
    times = rnd.choice(times, 20)

    y_int = np.empty(20, dtype=[("event", bool), ("time", int)])
    y_int["event"] = np.ones(20, dtype=bool)
    y_int["event"][:10] = False
    y_int["time"] = times

    pred = rnd.randn(20, 10)
    tp = np.linspace(1.0, 2.0, 10)
    _, bs_int = brier_score(y_int, y_int, pred, times=tp)

    y_float = np.empty(20, dtype=[("event", bool), ("time", float)])
    y_float["event"][:] = y_int["event"]
    y_float["time"][:] = y_int["time"]
    _, bs_float = brier_score(y_float, y_float, pred, times=tp)

    assert_array_almost_equal(bs_float, bs_int)


def test_ibs_nottingham_1(nottingham_prognostic_index):
    times = np.linspace(365, 1825, 5)  # t=1..5 years
    preds, y = nottingham_prognostic_index(times)

    score = integrated_brier_score(y, y, preds, times=times)
    assert score == pytest.approx(0.197936392255733)

    score = integrated_brier_score(y, y, preds[:, :4], times=times[:4])
    assert score == pytest.approx(0.185922397142833)


def test_ibs_nottingham_2(nottingham_prognostic_index):
    times = np.arange(1095, 1826)  # t=3..5 years
    preds, y = nottingham_prognostic_index(times)

    score = integrated_brier_score(y, y, preds, times=times)

    assert score == pytest.approx(0.231553687189643)


def test_ibs_single_time_point(nottingham_prognostic_index):
    pred, y = nottingham_prognostic_index([1825])

    with pytest.raises(ValueError, match="At least two time points must be given"):
        integrated_brier_score(y, y, pred, times=1825)


@pytest.fixture(params=["cindex", "auc", "brier"])
def scorers_data(request, make_whas500):
    t = request.param

    whas500_data = make_whas500(to_numeric=True)
    data = train_test_split(whas500_data.x, whas500_data.y, random_state=0, stratify=whas500_data.y["fstat"])
    times = np.percentile(whas500_data.y["lenfol"], np.linspace(5, 81, 15))

    if t == "cindex":

        def func(*args, **kwargs):
            ret = concordance_index_ipcw(*args, **kwargs)
            return ret[0]

        wrapper_cls = as_concordance_index_ipcw_scorer
        args = {}
    elif t == "auc":

        def func(*args, **kwargs):
            ret = cumulative_dynamic_auc(*args, **kwargs)
            return ret[1]

        wrapper_cls = as_cumulative_dynamic_auc_scorer
        args = {"times": times}
    elif t == "brier":
        func = integrated_brier_score
        wrapper_cls = as_integrated_brier_score_scorer
        args = {"times": times}
    else:
        raise AssertionError()

    return func, wrapper_cls, args, data


def test_scorers(scorers_data):
    score_func, wrapper_cls, score_args, data = scorers_data
    X_train, X_test, y_train, y_test = data

    est_std = CoxPHSurvivalAnalysis().fit(X_train, y_train)
    if issubclass(wrapper_cls, as_integrated_brier_score_scorer):
        times = score_args["times"]
        pred = np.vstack([fn(times) for fn in est_std.predict_survival_function(X_test)])
        sign = -1
    else:
        pred = est_std.predict(X_test)
        sign = 1

    expected = sign * score_func(y_train, y_test, pred, **score_args)

    est_wrap = wrapper_cls(CoxPHSurvivalAnalysis(), **score_args)
    est_wrap.fit(X_train, y_train)
    actual = est_wrap.score(X_test, y_test)

    assert actual == pytest.approx(expected)

    assert_array_almost_equal(est_wrap.predict(X_test), est_std.predict(X_test))

    chf_expected = est_std.predict_cumulative_hazard_function(X_test)
    chf_actual = est_wrap.predict_cumulative_hazard_function(X_test)
    assert_array_almost_equal([v.x for v in chf_expected], [v.x for v in chf_actual])
    assert_array_almost_equal([v.y for v in chf_expected], [v.y for v in chf_actual])

    surv_expected = est_std.predict_survival_function(X_test)
    surv_actual = est_wrap.predict_survival_function(X_test)
    assert_array_almost_equal([v.x for v in surv_expected], [v.x for v in surv_actual])
    assert_array_almost_equal([v.y for v in surv_expected], [v.y for v in surv_actual])


def test_brier_scorer_no_predict_survival_function(make_whas500):
    with pytest.raises(
        AttributeError, match=r"FastSurvivalSVM\(\) object has no attribute 'predict_survival_function'"
    ):
        as_integrated_brier_score_scorer(FastSurvivalSVM(), times=[100, 200, 300])


@pytest.mark.parametrize("pred_func", ["predict_cumulative_hazard_function", "predict_survival_function"])
def test_scorer_no_predict_function(make_whas500, pred_func):
    whas500_data = make_whas500(to_numeric=True)
    scorer = as_concordance_index_ipcw_scorer(FastSurvivalSVM())
    scorer.fit(whas500_data.x, whas500_data.y)

    with pytest.raises(AttributeError, match=f"This 'as_concordance_index_ipcw_scorer' has no attribute {pred_func!r}"):
        getattr(scorer, pred_func)
