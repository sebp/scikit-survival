import os.path

import numpy
from numpy.testing import assert_array_equal, assert_almost_equal
import pytest

from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
from sksurv.util import Surv


@pytest.fixture
def whas500_pred():
    WHAS500_DATA_FILE = os.path.join(os.path.dirname(__file__), 'data', 'whas500_predictions.csv')

    dat = numpy.loadtxt(WHAS500_DATA_FILE, delimiter=",")
    event = dat[:, 0] == 1
    time = dat[:, 1]
    risk = dat[:, 2]
    return event, time, risk


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
    estimate = [5, 15, 11, 34, 12+4.5e-9, 3, 9, 12-4.5e-9]

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


def assert_uno_c_almost_equal(y_train, y_test, estimate, expected, tau=None):
    result = concordance_index_ipcw(y_train, y_test, estimate, tau=tau)
    assert_array_equal(result[1:], expected[1:])
    assert_almost_equal(result[0], expected[0])


@pytest.fixture(params=[
    'no_ties',
    'tied_risk_1',
    'tied_risk_2',
    'truncated',
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
    elif p == 'truncated':
        y_train = Surv.from_arrays(
            time=(2, 4, 6, 8, 10, 11, 15, 19),
            event=(False, True, False, True, False, False, False, False))
        y_test = Surv.from_arrays(
            time=(1, 3, 5, 8, 12, 13),
            event=(True, False, False, True, True, True))
        estimate = (5, 8, 13, 11, 9, 4)
        expected = (0.7543736528146774, 4, 4, 0, 0)
        tau = 19
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
        expected = (0.4401618580, 11, 10, 0, 1)
    elif p == 'tied_event_and_time':
        y = Surv.from_arrays(
            event=[True, False, False, False, True, False, True, True, False, False, False, True, True],
            time=[34, 11, 11, 5, 1, 89, 13, 45, 7, 13, 9, 13, 90])
        estimate = (1, 19, 13, 13, 15, 14, 19, 23, 11, 10, 11, 1, 18)
        expected = (0.4722222222, 14, 12, 1, 2)
    elif p == 'whas500':
        event, time, estimate = whas500_pred
        y = Surv.from_arrays(event, time)
        expected = (0.7929001679258981, 57849, 17300, 0, 14)

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
        match = "time must be smaller than largest "\
                "observed time point:"
    elif p == 'last_time_uncensored_2':
        y_train = Surv.from_arrays(
            time=(2, 4, 6, 8, 10, 11, 15, 19),
            event=(False, True, False, True, False, False, False, True))
        y_test = Surv.from_arrays(
            time=(1, 23, 5, 27, 12),
            event=(True, False, True, True, False))
        estimate = (5, 13, 11, 9, 4)
        match = "time must be smaller than largest "\
                "observed time point:"
    elif p == 'zero_prob_1':
        y_train = Surv.from_arrays(
            time=(2, 4, 6, 8, 10, 11, 15, 19),
            event=(False, True, False, True, False, False, False, False))
        y_test = Surv.from_arrays(
            time=(1, 3, 5, 7, 12, 13, 19),
            event=(True, False, False, True, True, False, True))
        estimate = (5, 8, 13, 11, 9, 7, 4)
        match = "censoring survival function is zero "\
                "at one or more time points"
    elif p == 'zero_prob_2':
        y_train = Surv.from_arrays(
            time=(2, 4, 6, 8, 10, 11, 15, 18),
            event=(False, True, False, True, False, False, False, False))
        y_test = Surv.from_arrays(
            time=(1, 3, 5, 7, 12, 13, 19),
            event=(True, False, False, True, True, False, True))
        estimate = (5, 8, 13, 11, 9, 7, 4)
        match = "censoring survival function is zero "\
                "at one or more time points"
    elif p == 'zero_prob_3':
        y_train = Surv.from_arrays(
            time=(2, 4, 6, 8, 10, 11, 15, 18),
            event=(False, True, False, True, False, False, False, False))
        y_test = Surv.from_arrays(
            time=(1, 3, 5, 19, 12, 13, 7),
            event=(True, False, False, True, True, False, True))
        estimate = (5, 8, 13, 11, 9, 7, 4)
        match = "censoring survival function is zero "\
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
