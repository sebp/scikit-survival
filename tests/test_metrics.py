import os.path

import numpy
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_almost_equal
import pytest

from sksurv.datasets import load_gbsg2
from sksurv.functions import StepFunction
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import (
    brier_score,
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.exceptions import NoComparablePairException
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.preprocessing import OneHotEncoder
from sksurv.util import Surv


@pytest.fixture
def whas500_pred():
    WHAS500_DATA_FILE = os.path.join(os.path.dirname(__file__), 'data', 'whas500_predictions.csv')

    dat = numpy.loadtxt(WHAS500_DATA_FILE, delimiter=",")
    event = dat[:, 0] == 1
    time = dat[:, 1]
    risk = dat[:, 2]
    return event, time, risk


@pytest.fixture
def no_comparable_pairs():
    y = numpy.array([(False, 849.), (False, 28.), (False, 55.), (False, 727.),
                     (False, 505.), (False, 1558.), (False, 1292.), (False, 1737.),
                     (False, 944.), (False, 750.), (False, 2513.), (False, 472.),
                     (False, 2417.), (False, 538.), (False, 49.), (False, 723.),
                     (True, 3563.), (False, 1090.), (False, 1167.), (False, 587.),
                     (False, 1354.), (False, 910.), (False, 398.), (False, 854.),
                     (False, 3534.), (False, 280.), (False, 183.), (False, 883.),
                     (False, 32.), (False, 144.)], dtype=[("event", bool), ("time", float)])
    scores = numpy.random.randn(y.shape[0])
    return y, scores


def test_concordance_index_no_censoring_all_correct():
    time = [1, 5, 6, 11, 34, 45, 46, 50]
    event = numpy.repeat(True, len(time))
    estimate = numpy.arange(len(time))[::-1]

    c, con, dis, tie_r, tie_t = concordance_index_censored(event, time, estimate)
    assert 28 == con
    assert 0 == dis
    assert 0 == tie_r
    assert 0 == tie_t
    assert 1.0 == c


def test_concordance_index_no_censoring_all_wrong():
    time = [1, 5, 6, 11, 34, 45, 46, 50]
    event = numpy.repeat(True, len(time))
    # order is exactly reversed
    estimate = numpy.arange(len(time))

    c, con, dis, tie_r, tie_t = concordance_index_censored(event, time, estimate)
    assert 0 == con
    assert 28 == dis
    assert 0 == tie_r
    assert 0 == tie_t
    assert 0.0 == c


def test_concordance_index_no_ties():
    event = [False, True, True, False, False, True, False, False]
    time = [1, 5, 6, 11, 34, 45, 46, 50]
    estimate = [5, 8, 11, 34, 12, 3, 9, 12]

    c, con, dis, tie_r, tie_t = concordance_index_censored(event, time, estimate)

    assert 3 == con
    assert 10 == dis
    assert 0 == tie_r
    assert 0 == tie_t
    assert round(abs(0.2307692 - c), 6) == 0


def test_concordance_index_with_tied_time():
    event = [False, True, True, False, True, False, True, False, False]
    time = [1, 5, 6, 11, 11, 34, 45, 45, 50]
    estimate = [5, 8, 11, 19, 34, 12, 3, 9, 12]

    c, con, dis, tie_r, tie_t = concordance_index_censored(event, time, estimate)

    assert 8 == con
    assert 12 == dis
    assert 0 == tie_r
    assert 2 == tie_t
    assert round(abs(0.4 - c), 6) == 0


def test_concordance_index_with_tied_time2():
    event = [False, True, True, False, False, False, True, False, False]
    time = [1, 5, 6, 11, 11, 34, 45, 45, 50]
    estimate = [5, 8, 11, 19, 34, 12, 3, 9, 12]

    c, con, dis, tie_r, tie_t = concordance_index_censored(event, time, estimate)

    assert 3 == con
    assert 12 == dis
    assert 0 == tie_r
    assert 1 == tie_t
    assert round(abs(0.2 - c), 6) == 0


def test_concordance_index_with_tied_event():
    event = [False, True, False, True, True, False, True, False, False]
    time = [1, 5, 6, 11, 11, 34, 45, 45, 50]
    estimate = [5, 8, 11, 19, 34, 12, 3, 9, 12]

    c, con, dis, tie_r, tie_t = concordance_index_censored(event[::-1], time[::-1], estimate[::-1])

    assert 9 == con
    assert 8 == dis
    assert 0 == tie_r
    assert 1 == tie_t
    assert round(abs(0.5294118 - c), 6) == 0


def test_concordance_index_with_tied_risk():
    event = [False, True, True, False, True, True, False, False]
    time = [1, 5, 6, 11, 34, 45, 46, 50]
    estimate = [5, 15, 11, 34, 12, 3, 9, 12]

    c, con, dis, tie_r, tie_t = concordance_index_censored(event, time, estimate)

    assert 9 == con
    assert 6 == dis
    assert 1 == tie_r
    assert 0 == tie_t
    assert round(abs(0.59375 - c), 6) == 0


def test_concordance_index_with_almost_tied_risk():
    event = [False, True, True, False, True, True, False, False]
    time = [1, 5, 6, 11, 34, 45, 46, 50]
    estimate = [5, 15, 11, 34, 12 + 4.5e-9, 3, 9, 12 - 4.5e-9]

    c, con, dis, tie_r, tie_t = concordance_index_censored(event, time, estimate)

    assert 9 == con
    assert 6 == dis
    assert 1 == tie_r
    assert 0 == tie_t
    assert round(abs(0.59375 - c), 6) == 0


def test_concordance_index_with_tied_event_and_time():
    event = [True, False, False, False, True, False, True, True, False, False, False, True]
    time = [34, 11, 11, 5, 1, 89, 13, 45, 7, 13, 9, 13]
    estimate = [1, 19, 13, 13, 15, 14, 19, 23, 11, 10, 11, 1]

    c, con, dis, tie_r, tie_t = concordance_index_censored(event, time, estimate)

    assert 12 == con
    assert 9 == dis
    assert 1 == tie_r
    assert 2 == tie_t
    assert round(abs(0.5681818 - c), 6) == 0


def test_concordance_index(whas500_pred):
    event, time, risk = whas500_pred

    c, con, dis, tie_r, tie_t = concordance_index_censored(event, time, risk)
    assert 57849 == con
    assert 17300 == dis
    assert 0 == tie_r
    assert 14 == tie_t
    assert round(abs(0.7697907 - c), 6) == 0


def test_concordance_index_different_length():
    event = numpy.array([True, False, False, True, True, False])
    time = numpy.array([1, 5, 10, 12, 7, 65])
    estimate = numpy.array([12, 8, 1, 89, 56, 13])

    msg = r"Found input variables with inconsistent numbers of samples: .+"
    with pytest.raises(ValueError, match=msg):
        concordance_index_censored(event, time[:3], estimate)

    with pytest.raises(ValueError, match=msg):
        concordance_index_censored(event, time, estimate[:3])

    with pytest.raises(ValueError, match=msg):
        concordance_index_censored(event[:3], time, estimate)

    with pytest.raises(ValueError, match=msg):
        concordance_index_censored(event, time[:3], estimate[:3])

    with pytest.raises(ValueError, match=msg):
        concordance_index_censored(event[:3], time, estimate[:3])

    with pytest.raises(ValueError, match=msg):
        concordance_index_censored(event[:3], time[:3], estimate)


def test_concordance_index_boolean_event():
    event = numpy.array([1, 0, 0, 1, 1, 0])
    time = numpy.array([1, 5, 10, 12, 7, 65])
    estimate = numpy.array([12, 8, 1, 89, 56, 13])

    with pytest.raises(ValueError,
                       match="only boolean arrays are supported as class labels for survival analysis.+"):
        concordance_index_censored(event, time, estimate)


def test_concordance_index_min_samples():
    event = numpy.array([False])
    time = numpy.array([10])
    estimate = numpy.array([12])

    with pytest.raises(ValueError, match="Need a minimum of two samples"):
        concordance_index_censored(event, time, estimate)


def test_concordance_index_all_censored():
    event = numpy.array([False, False])
    time = numpy.array([10, 12])
    estimate = numpy.array([12, 13])

    with pytest.raises(ValueError, match="All samples are censored"):
        concordance_index_censored(event, time, estimate)


def test_concordance_index_all_finite():
    event = numpy.array([True, False, None, True, True, False])
    time = numpy.array([1, 5, 10, 12, 7, 65], dtype=float)
    estimate = numpy.array([12, 8, 1, 89, 56, 13], dtype=float)

    msg = r"Input contains NaN, infinity or a value too large for .+"
    with pytest.raises(ValueError, match=msg):
        concordance_index_censored(event, time, estimate)

    event[2] = False
    time[3] = numpy.nan
    with pytest.raises(ValueError, match=msg):
        concordance_index_censored(event, time, estimate)

    time[3] = numpy.nan
    estimate[5] = numpy.inf
    with pytest.raises(ValueError, match=msg):
        concordance_index_censored(event, time, estimate)


def test_concordance_index_no_comparable(no_comparable_pairs):
    y, scores = no_comparable_pairs

    with pytest.raises(NoComparablePairException):
        concordance_index_censored(y["event"], y["time"], scores)


def assert_uno_c_almost_equal(y_train, y_test, estimate, expected, tau=None):
    result = concordance_index_ipcw(y_train, y_test, estimate, tau=tau)
    assert_array_equal(result[1:], expected[1:])
    assert_almost_equal(result[0], expected[0])


@pytest.fixture(params=[
    'no_ties',
    'tied_risk_1',
    'tied_risk_2',
    'truncated_1',
    'truncated_2',
    'last_time_censored',
    'tied_event',
    'tied_event_and_time',
    'whas500',
])
def uno_c_data(request, whas500_pred):
    p = request.param

    y = None
    y_train = None
    y_test = None
    estimate = None
    expected = None
    tau = None

    if p == 'no_ties':
        y = Surv.from_arrays(
            event=numpy.array((0, 1, 1, 0, 1, 0, 1, 0, 0, 1), dtype=bool),
            time=(1, 5, 6, 10, 11, 34, 45, 46, 50, 56))
        estimate = (5, 8, 11, 19, 34, 12, 3, 9, 12, 20)
        expected = (0.347890360332615, 8, 15, 0, 0)
    elif p == 'tied_risk_1':
        y = Surv.from_arrays(
            time=(1, 5, 6, 10, 11, 34, 45, 46, 50, 56),
            event=numpy.array((0, 1, 1, 0, 1, 0, 1, 0, 0, 1), dtype=bool))
        estimate = (5, 8, 11, 11, 34, 12, 3, 9, 12, 20)
        expected = (0.365629810028969, 8, 14, 1, 0)
    elif p == 'tied_risk_2':
        y = Surv.from_arrays(
            time=(1, 5, 6, 10, 11, 34, 45, 46, 50, 56),
            event=numpy.array((0, 1, 1, 0, 1, 0, 1, 0, 0, 1), dtype=bool))
        estimate = (5, 8, 11, 11, 34, 12, 11, 9, 12, 20)
        expected = (0.387865723332956, 7, 14, 2, 0)
    elif p == 'truncated_1':
        y_train = Surv.from_arrays(
            time=(2, 4, 6, 8, 10, 11, 15, 19),
            event=(False, True, False, True, False, False, False, False))
        y_test = Surv.from_arrays(
            time=(1, 3, 5, 8, 12, 13),
            event=(True, False, False, True, True, True))
        estimate = (5, 8, 13, 11, 9, 4)
        expected = (0.7543736528146774, 4, 4, 0, 0)
        tau = 19
    elif p == 'truncated_2':
        y = Surv.from_arrays(
            time=(1, 5, 6, 10, 11, 34, 45, 46, 50, 56),
            event=numpy.array((0, 1, 1, 0, 1, 0, 1, 1, 1, 1), dtype=bool))
        estimate = (5, 8, 11, 19, 34, 12, 3, 9, 12, 18)
        expected = (0.347890361949191, 8, 18, 0, 0)
        tau = 45.25
    elif p == 'last_time_censored':
        y_train = Surv.from_arrays(
            time=(2, 4, 6, 8, 10, 11, 15, 19),
            event=(False, True, False, True, False, False, False, False))
        y_test = Surv.from_arrays(
            time=(1, 3, 5, 7, 12, 13, 20),
            event=(True, False, False, True, True, False, False))
        estimate = (5, 8, 13, 11, 9, 7, 4)
        expected = (0.8126567565914234, 6, 5, 0, 0)
    elif p == 'tied_event':
        y = Surv.from_arrays(
            event=[False, True, False, True, True, False, True, False, False, True],
            time=[1, 5, 6, 11, 11, 34, 45, 45, 50, 55])
        estimate = (5, 8, 11, 19, 34, 12, 3, 9, 12, 18)
        expected = (0.4036321031048623, 11, 10, 0, 1)
    elif p == 'tied_event_and_time':
        y = Surv.from_arrays(
            event=[True, False, False, False, True, False, True, True, False, False, False, True, True],
            time=[34, 11, 11, 5, 1, 89, 13, 45, 7, 13, 9, 13, 90])
        estimate = (1, 19, 13, 13, 15, 14, 19, 23, 11, 10, 11, 1, 18)
        expected = (0.46795357052737824, 14, 12, 1, 2)
    elif p == 'whas500':
        event, time, estimate = whas500_pred
        y = Surv.from_arrays(event, time)
        expected = (0.7929275009049014, 57849, 17300, 0, 14)

    y_train = y if y_train is None else y_train
    y_test = y if y_test is None else y_test

    yield y_train, y_test, estimate, expected, tau


def test_uno_c(uno_c_data):
    data = uno_c_data
    assert_uno_c_almost_equal(*data)


@pytest.fixture(params=[
    'last_time_uncensored_1',
    'last_time_uncensored_2',
    'zero_prob_1',
    'zero_prob_2',
    'zero_prob_3',
])
def uno_c_failure_data(request):
    p = request.param

    if p == 'last_time_uncensored_1':
        y_train = Surv.from_arrays(
            time=(2, 4, 6, 8, 10, 11, 15, 19),
            event=(False, True, False, True, False, False, False, True))
        y_test = Surv.from_arrays(
            time=(1, 3, 5, 7, 12, 13, 20),
            event=(True, False, False, True, True, False, True))
        estimate = (5, 8, 13, 11, 9, 7, 4)
        match = "time must be smaller than largest " \
                "observed time point:"
    elif p == 'last_time_uncensored_2':
        y_train = Surv.from_arrays(
            time=(2, 4, 6, 8, 10, 11, 15, 19),
            event=(False, True, False, True, False, False, False, True))
        y_test = Surv.from_arrays(
            time=(1, 23, 5, 27, 12),
            event=(True, False, True, True, False))
        estimate = (5, 13, 11, 9, 4)
        match = "time must be smaller than largest " \
                "observed time point:"
    elif p == 'zero_prob_1':
        y_train = Surv.from_arrays(
            time=(2, 4, 6, 8, 10, 11, 15, 19),
            event=(False, True, False, True, False, False, False, False))
        y_test = Surv.from_arrays(
            time=(1, 3, 5, 7, 12, 13, 19),
            event=(True, False, False, True, True, False, True))
        estimate = (5, 8, 13, 11, 9, 7, 4)
        match = "censoring survival function is zero " \
                "at one or more time points"
    elif p == 'zero_prob_2':
        y_train = Surv.from_arrays(
            time=(2, 4, 6, 8, 10, 11, 15, 18),
            event=(False, True, False, True, False, False, False, False))
        y_test = Surv.from_arrays(
            time=(1, 3, 5, 7, 12, 13, 19),
            event=(True, False, False, True, True, False, True))
        estimate = (5, 8, 13, 11, 9, 7, 4)
        match = "censoring survival function is zero " \
                "at one or more time points"
    elif p == 'zero_prob_3':
        y_train = Surv.from_arrays(
            time=(2, 4, 6, 8, 10, 11, 15, 18),
            event=(False, True, False, True, False, False, False, False))
        y_test = Surv.from_arrays(
            time=(1, 3, 5, 19, 12, 13, 7),
            event=(True, False, False, True, True, False, True))
        estimate = (5, 8, 13, 11, 9, 7, 4)
        match = "censoring survival function is zero " \
                "at one or more time points"
    else:
        assert False

    yield y_train, y_test, estimate, match


def test_uno_c_failure(uno_c_failure_data):
    y_train, y_test, estimate, match = uno_c_failure_data

    with pytest.raises(ValueError, match=match):
        concordance_index_ipcw(y_train, y_test, estimate)


def test_uno_c_all_censored():
    y_train = Surv.from_arrays(
        time=(2, 4, 6, 8, 10, 11, 15, 19),
        event=(True, True, True, True, True, True, True, True))
    y_test = Surv.from_arrays(
        time=(1, 3, 5, 7, 12, 13, 20),
        event=(True, False, False, True, True, False, False))
    estimate = (5, 8, 13, 11, 9, 7, 4)

    ret_uno = concordance_index_ipcw(y_train, y_test, estimate)
    ret_harrell = concordance_index_censored(y_test['event'], y_test['time'], estimate)
    assert ret_uno == ret_harrell


def test_uno_c_no_comparable(no_comparable_pairs):
    y, scores = no_comparable_pairs

    with pytest.raises(NoComparablePairException):
        concordance_index_ipcw(y, y, scores)


@pytest.fixture()
def uno_auc_data_15():
    y = Surv.from_arrays(
        time=[10.88, 19.78, 40.92, 98.7, 70.19, 10.15, 28.95, 29.57, 17.9, 63.78, 36.22, 83.14, 13.69, 99.51, 3.19],
        event=[1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1])
    estimate = [-1.019, -0.016, 0.132, 0.269, -0.777, -1.077, 0.894, -1.227, -0.417, 0.072, -1.275, -0.91, -0.825,
                -0.292, -0.045]
    return y, estimate


@pytest.fixture()
def uno_auc_data_20():
    y_train = Surv.from_arrays(
        time=[77.6, 57.6, 66.6, 67.0, 31.5, 5.5, 67.4, 43.7, 31.7, 71.9, 81.1, 56.2, 88.1, 2.9, 62.0, 17.2, 88.0,
              26.4, 93.5, 79.9],
        event=[1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
    )
    y_test = Surv.from_arrays(
        time=[10.88, 19.78, 40.92, 98.7, 70.19, 10.15, 28.95, 29.57, 17.9, 63.78, 36.22, 83.14, 13.69, 99.51, 3.19],
        event=[1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1])
    estimate = [-1.019, -0.016, 0.132, 0.269, -0.777, -1.077, 0.894, -1.227, -0.417, 0.072, -1.275, -0.91, -0.825,
                -0.292, -0.045]
    return y_train, y_test, estimate


@pytest.fixture(params=[
    'single_time',
    'two_times',
    'two_times_int',
    'min_to_max_times',
    'train_test',
    'tied_test_time',
    'tied_test_score',
])
def uno_auc_data(request, uno_auc_data_15, uno_auc_data_20):
    p = request.param

    y_test = None
    if p == 'single_time':
        y_train, estimate = uno_auc_data_15
        times = 28
        iauc = 0.362963
        expected = numpy.array([iauc])
    elif p == 'two_times':
        y_train, estimate = uno_auc_data_15
        times = [15, 66]
        iauc = 0.3943949
        expected = numpy.array([0.3030303, 0.4500000])
    elif p == 'two_times_int':
        y_train, estimate = uno_auc_data_15
        y_train['time'] = y_train['time'] * 100.
        y_train = y_train.astype([('event', bool), ('time', int)])
        estimate = (numpy.array(estimate) * 1000).astype(int)
        times = numpy.array([1500, 6600], dtype=int)
        iauc = 0.3943949
        expected = numpy.array([0.3030303, 0.4500000])
    elif p == 'min_to_max_times':
        y_train, estimate = uno_auc_data_15
        times = (y_train['time'].min(), 15, 66, y_train['time'].max() - 1e-6)
        iauc = 0.3999539
        expected = numpy.array([0.6428571, 0.3030303, 0.4500000, 0.3162996])
    elif p == 'train_test':
        y_train, y_test, estimate = uno_auc_data_20
        times = [15, 66]
        iauc = 0.385509
        expected = numpy.array([0.3030303, 0.4357061])
    elif p == 'tied_test_time':
        y_train, y_test, estimate = uno_auc_data_20
        y_test['time'][0] = y_test['time'][1]
        times = [15, 66]
        iauc = 0.4204885
        expected = numpy.array([0.3750000, 0.4357061])
    elif p == 'tied_test_score':
        y_train, y_test, estimate = uno_auc_data_20
        estimate[0] = estimate[-1]
        times = [15, 66]
        iauc = 0.495604291
        expected = numpy.array([0.4242424, 0.539036])
    else:
        assert False

    if y_test is None:
        y_test = y_train
    yield y_train, y_test, estimate, times, expected, iauc


def test_uno_auc(uno_auc_data):
    y_train, y_test, estimate, times, expect_auc, expect_iauc = uno_auc_data

    auc, iauc = cumulative_dynamic_auc(y_train, y_test, estimate, times)
    assert_array_almost_equal(auc, expect_auc)
    assert_almost_equal(iauc, expect_iauc)


@pytest.fixture(params=[
    'whas500',
    'whas500_unordered_time',
])
def uno_auc_whas500_data(request, whas500_pred):
    p = request.param

    event, time, estimate = whas500_pred
    y_train = Surv.from_arrays(event=event[:300], time=time[:300])
    y_test = Surv.from_arrays(event=event[300:], time=time[300:])
    estimate = estimate[300:]
    if p == 'whas500_unordered_time':
        times = (1000, 600, 1400, 200, 400, 1200, 800, 1000, 200)
    elif p == 'whas500':
        times = (200, 400, 600, 800, 1000, 1200, 1400)
    else:
        assert False
    iauc = 0.8045058
    expected = numpy.array([0.7720669, 0.7765915, 0.7962623, 0.8759295, 0.8759295, 0.8759513, 0.9147647])
    yield y_train, y_test, estimate, times, expected, iauc


def test_uno_auc_whas500(uno_auc_whas500_data):
    y_train, y_test, estimate, times, expect_auc, expect_iauc = uno_auc_whas500_data

    auc, iauc = cumulative_dynamic_auc(y_train, y_test, estimate, times)
    assert_array_almost_equal(auc, expect_auc)
    assert_almost_equal(iauc, expect_iauc)


@pytest.fixture(params=[
    'estimate_2d',
    'estimate_3d',
    'test_time_too_big_1',
    'test_time_too_big_2',
    'ipcw_undefined_1',
    'ipcw_undefined_2',
    'all_censored_train',
    'all_censored_test',
])
def uno_auc_censoring_failure_data(request, uno_auc_data_20):
    p = request.param

    estimate = None
    times = 33
    if p == 'estimate_2d':
        y_train, y_test, estimate = uno_auc_data_20
        estimate = numpy.atleast_2d(estimate)
        match = "Expected 1D array, got 2D array instead"
    elif p == 'estimate_3d':
        y_train, y_test, estimate = uno_auc_data_20
        estimate = numpy.atleast_3d(estimate)
        match = r"Found array with dim 3\. Estimator expected <= 2\."
    elif p == 'test_time_too_big_1':
        y_train, y_test, _ = uno_auc_data_20
        y_test['time'][11] = 100
        match = "time must be smaller than largest observed time point:"
    elif p == 'test_time_too_big_2':
        y_train, y_test, _ = uno_auc_data_20
        idx = numpy.argmax(y_test['time'])
        y_test['event'][idx] = True
        match = "time must be smaller than largest observed time point:"
    elif p == 'ipcw_undefined_1':
        y_train, y_test, _ = uno_auc_data_20
        idx = numpy.argmax(y_train['time'])
        y_train['event'][idx] = 0
        y_test['time'][0] = y_train['time'][idx]
        match = "censoring survival function is zero at one or more time points"
    elif p == 'ipcw_undefined_2':
        y_train, y_test, _ = uno_auc_data_20
        idx = numpy.argmax(y_train['time'])
        y_train['event'][idx] = 0
        y_test['time'][-1] = y_train['time'][idx] * 2
        y_test['event'][-1] = True
        match = "censoring survival function is zero at one or more time points"
    elif p == 'all_censored_train':
        y_train, y_test, _ = uno_auc_data_20
        y_train['event'] = False
        match = "all samples are censored"
    elif p == 'all_censored_test':
        y_train, y_test, _ = uno_auc_data_20
        y_test['event'] = False
        match = "all samples are censored"
    else:
        assert False

    if estimate is None:
        estimate = numpy.random.randn(y_test.shape[0])
    yield y_train, y_test, times, estimate, match


def test_uno_auc_censoring_failure(uno_auc_censoring_failure_data):
    y_train, y_test, times, estimate, match = uno_auc_censoring_failure_data

    with pytest.raises(ValueError,
                       match=match):
        cumulative_dynamic_auc(y_train, y_test, estimate, times)


@pytest.fixture(params=[
    'nan',
    'infinite_1',
    'infinite_2',
    'too_big_1',
    'too_big_2',
    'too_big_3',
    'too_small_1',
    'too_small_2',
    'empty',
])
def uno_auc_times_failure_data(request, uno_auc_data_20):
    p = request.param

    if p == 'nan':
        y_train, y_test, _ = uno_auc_data_20
        times = (0.2, numpy.nan)
        match = r"Input contains NaN, infinity or a value too large for dtype\('float64'\)."
    elif p == 'infinite_1':
        y_train, y_test, _ = uno_auc_data_20
        times = (0.2, numpy.infty)
        match = r"Input contains NaN, infinity or a value too large for dtype\('float64'\)."
    elif p == 'infinite_2':
        y_train, y_test, _ = uno_auc_data_20
        times = (0.2, -numpy.infty)
        match = r"Input contains NaN, infinity or a value too large for dtype\('float64'\)."
    elif p == 'too_big_1':
        y_train, y_test, _ = uno_auc_data_20
        idx = numpy.argmax(y_test['time'])
        y_test['event'][idx] = False
        t_max = y_test['time'][idx]
        times = (33, t_max / 2, t_max)
        match = r"all times must be within follow-up time of test data: \[3\.19; 99\.51\["
    elif p == 'too_big_2':
        y_train, y_test, _ = uno_auc_data_20
        idx = numpy.argmax(y_test['time'])
        y_test['event'][idx] = False
        t_max = y_test['time'][idx]
        times = (33, t_max / 2, t_max + 0.1)
        match = r"all times must be within follow-up time of test data: \[3\.19; 99\.51\["
    elif p == 'too_big_3':
        y_train, y_test, _ = uno_auc_data_20
        max_train = numpy.max(y_train['time'])
        idx_test = y_test['time'] > max_train
        y_test['event'][idx_test] = False
        y_test['time'][idx_test] = max_train
        times = (33, max_train)
        match = r"all times must be within follow-up time of test data: \[3\.19; 93\.5\["
    elif p == 'too_small_1':
        y_train, y_test, _ = uno_auc_data_20
        idx = numpy.argmin(y_test['time'])
        y_test['event'][idx] = True
        t_min = y_test['time'][idx]
        times = (t_min - 1e-6, 33)
        match = r"all times must be within follow-up time of test data: \[3\.19; 99\.51\["
    elif p == 'too_small_2':
        y_train, y_test, _ = uno_auc_data_20
        idx = numpy.argmin(y_test['time'])
        y_test['event'][idx] = True
        t_min = y_test['time'][idx]
        times = (33, t_min - 0.1, t_min / 2)
        match = r"all times must be within follow-up time of test data: \[3\.19; 99\.51\["
    elif p == 'empty':
        y_train, y_test, _ = uno_auc_data_20
        times = []
        match = r'Found array with 0 sample\(s\)'
    else:
        assert False

    yield y_train, y_test, times, match


def test_uno_auc_times_failure(uno_auc_times_failure_data):
    y_train, y_test, times, match = uno_auc_times_failure_data

    estimate = numpy.random.randn(y_test.shape[0])
    with pytest.raises(ValueError,
                       match=match):
        cumulative_dynamic_auc(y_train, y_test, estimate, times)


@pytest.fixture
def nottingham_prognostic_index():
    def _get_npi(times):
        X, y = load_gbsg2()

        grade = X.loc[:, "tgrade"].map({"I": 1, "II": 2, "III": 3}).astype(int)
        NPI = 0.2 * X.loc[:, "tsize"] / 10 + 1 + grade
        NPI[NPI < 3.4] = 1.0
        NPI[(NPI >= 3.4) & (NPI <= 5.4)] = 2.0
        NPI[NPI > 5.4] = 3.0

        preds = numpy.empty((X.shape[0], len(times)), dtype=float)
        for j, ts in enumerate(times):
            survs = {}
            for i in NPI.unique():
                idx = numpy.flatnonzero(NPI == i)
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
        assert False

    yield pred, y, t, bs


def test_brier_nottingham(brier_npi_data):
    pred, y, times, expected_score = brier_npi_data

    _, score = brier_score(y, y, pred.squeeze(), times=times)
    assert round(abs(score[0] - expected_score), 6) == 0


def test_brier_nottingham_many(nottingham_prognostic_index):
    times = [365, 730, 1095, 1460, 1825]
    pred, y = nottingham_prognostic_index(times)

    expected_score = numpy.array([
        0.0762922458520448,
        0.182536421174199,
        0.220017747254941,
        0.234133800146671,
        0.233822955042198,
    ])

    t1, score = brier_score(y, y, pred.squeeze(), times=times)
    assert_array_almost_equal(score, expected_score)

    t2, score = brier_score(y, y, pred.squeeze(), times=times[::-1])
    assert_array_almost_equal(score, expected_score)
    assert_array_equal(t1, t2)


def test_brier_times_too_large(nottingham_prognostic_index):
    pred, y = nottingham_prognostic_index([1825])

    with pytest.raises(ValueError,
                       match="all times must be within follow-up time of test data:"):
        brier_score(y, y, pred, times=9999)


def test_brier_wrong_estimate_shape(nottingham_prognostic_index):
    pred, y = nottingham_prognostic_index([720, 1825])

    with pytest.raises(ValueError,
                       match="expected estimate with 2 columns, but got 1"):
        brier_score(y, y, pred[:, :1], times=[720, 1825])

    with pytest.raises(ValueError,
                       match="expected estimate with 3 columns, but got 2"):
        brier_score(y, y, pred, times=[720, 960, 1825])

    with pytest.raises(ValueError,
                       match="expected estimate with 686 samples, but got 10"):
        brier_score(y, y, pred[:10], times=[720, 1825])


def test_brier_coxph():
    X, y = load_gbsg2()
    X.loc[:, "tgrade"] = X.loc[:, "tgrade"].map(len).astype(int)

    Xt = OneHotEncoder().fit_transform(X)

    est = CoxPHSurvivalAnalysis(ties="efron").fit(Xt, y)
    survs = est.predict_survival_function(Xt)

    preds = [fn(1825) for fn in survs]

    _, score = brier_score(y, y, preds, 1825)

    assert round(abs(score[0] - 0.208817407492645), 5) == 0


def test_ibs_nottingham_1(nottingham_prognostic_index):
    times = numpy.linspace(365, 1825, 5)  # t=1..5 years
    preds, y = nottingham_prognostic_index(times)

    score = integrated_brier_score(y, y, preds, times=times)
    assert round(abs(score - 0.197936392255733), 6) == 0

    score = integrated_brier_score(y, y, preds[:, :4], times=times[:4])
    assert round(abs(score - 0.185922397142833), 6) == 0


def test_ibs_nottingham_2(nottingham_prognostic_index):
    times = numpy.arange(1095, 1826)  # t=3..5 years
    preds, y = nottingham_prognostic_index(times)

    score = integrated_brier_score(y, y, preds, times=times)

    assert round(abs(score - 0.231553687189643), 6) == 0


def test_ibs_single_time_point(nottingham_prognostic_index):
    pred, y = nottingham_prognostic_index([1825])

    with pytest.raises(ValueError,
                       match="At least two time points must be given"):
        integrated_brier_score(y, y, pred, times=1825)
