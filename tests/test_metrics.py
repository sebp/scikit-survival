import os.path

import numpy
import pytest

from sksurv.metrics import concordance_index_censored


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
    assert 0 == tie_t
    assert round(abs(0.4 - c), 6) == 0


def test_concordance_index_with_tied_time2():
    event = [False, True, True, False, False, False, True, False, False]
    time = [1, 5, 6, 11, 11, 34, 45, 45, 50]
    estimate = [5, 8, 11, 19, 34, 12, 3, 9, 12]

    c, con, dis, tie_r, tie_t = concordance_index_censored(event, time, estimate)

    assert 3 == con
    assert 12 == dis
    assert 0 == tie_r
    assert 0 == tie_t
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


def test_concordance_index_with_tied_event_and_time():
    event = [True, False, False, False, True, False, True, True, False, False, False, True]
    time = [34, 11, 11, 5, 1, 89, 13, 45, 7, 13, 9, 13]
    estimate = [1, 19, 13, 13, 15, 14, 19, 23, 11, 10, 11, 1]

    c, con, dis, tie_r, tie_t = concordance_index_censored(event, time, estimate)

    assert 12 == con
    assert 9 == dis
    assert 1 == tie_r
    assert 1 == tie_t
    assert round(abs(0.5681818 - c), 6) == 0


def test_concordance_index(whas500_pred):
    event, time, risk = whas500_pred

    c, con, dis, tie_r, tie_t = concordance_index_censored(event, time, risk)
    assert 57849 == con
    assert 17300 == dis
    assert 0 == tie_r
    assert 119 == tie_t
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
