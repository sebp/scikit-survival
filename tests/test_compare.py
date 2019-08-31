import numpy
from numpy.testing import assert_almost_equal
import pandas
from pandas.testing import assert_frame_equal
import pytest

from sksurv.datasets import load_veterans_lung_cancer
from sksurv.compare import compare_survival


def test_logrank_rossi(rossi):
    chisq, pval, stats, covar = compare_survival(rossi.y, rossi.x.loc[:, "race"], return_stats=True)

    expected_counts = numpy.array([53, 379])
    expected_obs = numpy.array([12, 102])
    expected_exp = numpy.array([14.709347908949, 99.2906520910509])
    expected_var = numpy.array([[12.7417883289863, -12.7417883289863],
                                [-12.7417883289863, 12.7417883289863]])

    assert_almost_equal(stats["counts"], expected_counts)
    assert_almost_equal(stats["observed"], expected_obs)
    assert_almost_equal(stats["expected"], expected_exp)
    assert_almost_equal(stats["statistic"], expected_obs - expected_exp)
    assert_almost_equal(covar, expected_var)

    expected_chisq = 0.576101713683918
    expected_pval = 0.447844394252168

    assert round(abs(chisq - expected_chisq), 6) == 0
    assert round(abs(pval - expected_pval), 6) == 0


def test_logrank_veterans():
    X, y = load_veterans_lung_cancer()

    chisq, pval, stats, covar = compare_survival(y, X.loc[:, "Celltype"], return_stats=True)

    expected_stats = pandas.DataFrame(
        columns=["counts", "observed", "expected", "statistic"],
        index=["adeno", "large", "smallcell", "squamous"])
    expected_stats.index.name = "group"
    expected_stats["counts"] = numpy.array([27, 27, 48, 35], dtype=numpy.intp)
    expected_stats["observed"] = numpy.array([26, 26, 45, 31], dtype=numpy.int_)
    expected_stats["expected"] = [15.6937646143605, 34.5494783863493, 30.1020793268148, 47.6546776724754]
    expected_stats["statistic"] = expected_stats["observed"] - expected_stats["expected"]
    assert_frame_equal(stats, expected_stats)

    expected_var = numpy.array([
        [12.9661700605014, -4.07011754397142, -4.40872930298506, -4.48732321354496],
        [-4.07011754397142, 24.1990352938484, -7.81168661717217, -12.3172311327048],
        [-4.40872930298506, -7.81168661717217, 21.7542679406138, -9.53385202045655],
        [-4.48732321354496, -12.3172311327048, -9.53385202045655, 26.3384063667063]
    ])

    assert_almost_equal(covar, expected_var)

    expected_chisq = 25.4037003457854
    expected_pval = 1.27124593900609e-05

    assert round(abs(chisq - expected_chisq), 6) == 0
    assert round(abs(pval - expected_pval), 6) == 0


def test_logrank_two_1():
    y = numpy.empty(10, dtype=[("status", bool), ("time", float)])
    y["time"] = [12, 1, 6, 9, 7, 4, 13, 5, 2, 3]
    y["status"] = [False, True, True, False, False, False, True, False, True, True]
    x = numpy.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    chisq, pval = compare_survival(y, x)
    expected_chisq = 0.0658861965695617
    expected_pval = 0.797423444741849

    assert round(abs(chisq - expected_chisq), 6) == 0
    assert round(abs(pval - expected_pval), 6) == 0


def test_logrank_two_2():
    y = numpy.empty(10, dtype=[("status", bool), ("time", float)])
    y["time"] = [12, 1, 6, 9, 7, 4, 13, 5, 2, 3]
    y["status"] = [False, False, True, False, False, False, True, False, True, True]
    x = numpy.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    chisq, pval = compare_survival(y, x)
    expected_chisq = 0.843638413831986
    expected_pval = 0.358358251735077

    assert round(abs(chisq - expected_chisq), 6) == 0
    assert round(abs(pval - expected_pval), 6) == 0


def test_logrank_two_3():
    y = numpy.empty(10, dtype=[("status", bool), ("time", float)])
    y["time"] = [12, 1, 6, 9, 7, 4, 13, 5, 2, 3]
    y["status"] = [False, False, True, False, False, False, True, False, True, True]
    x = numpy.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    chisq, pval = compare_survival(y, x)
    expected_chisq = 0.843638413831986
    expected_pval = 0.358358251735077

    assert round(abs(chisq - expected_chisq), 6) == 0
    assert round(abs(pval - expected_pval), 6) == 0


def test_logrank_two_4():
    y = numpy.empty(10, dtype=[("status", bool), ("time", float)])
    y["time"] = [12, 1, 2, 2, 7, 4, 1, 5, 2, 5]
    y["status"] = [False, False, True, False, False, False, True, False, True, True]
    x = numpy.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    chisq, pval = compare_survival(y, x)
    expected_chisq = 1.07692307692308
    expected_pval = 0.299386905803601

    assert round(abs(chisq - expected_chisq), 6) == 0
    assert round(abs(pval - expected_pval), 6) == 0


def test_logrank_two_5():
    y = numpy.empty(10, dtype=[("status", bool), ("time", float)])
    y["time"] = [12, 1, 2, 2, 7, 4, 1, 5, 2, 5]
    y["status"] = [False, False, True, True, False, False, False, True, False, True]
    x = numpy.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    chisq, pval = compare_survival(y, x)
    expected_chisq = 0.
    expected_pval = 1.

    assert round(abs(chisq - expected_chisq), 6) == 0
    assert round(abs(pval - expected_pval), 6) == 0


def test_logrank_two_6():
    y = numpy.empty(10, dtype=[("status", bool), ("time", float)])
    y["time"] = [12, 12, 2, 2, 7, 4, 1, 5, 2, 5]
    y["status"] = [True, True, False, False, False, False, False, True, False, True]
    x = numpy.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    chisq, pval = compare_survival(y, x)
    expected_chisq = 4.
    expected_pval = 0.0455002638963585

    assert round(abs(chisq - expected_chisq), 6) == 0
    assert round(abs(pval - expected_pval), 6) == 0


def test_logrank_three():
    y = numpy.empty(51, dtype=[("status", bool), ("time", float)])
    y["time"] = [88, 62, 81, 64, 19, 35, 49, 82, 57, 6, 97, 52, 4, 90, 16, 92, 4, 59, 90,
                 71, 39, 94, 68, 77, 67, 15, 67, 7, 32, 14, 39, 26, 24, 2, 4, 95, 2, 12,
                 94, 64, 68, 16, 99, 6, 57, 81, 53, 33, 31, 91, 91]
    y["status"] = [False, False, True, False, False, False, True, True, True, True,
                   True, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, True, False, False, True, False,
                   True, True, False, False, False, True, False, False, True, False, False,
                   True, False, True, False, True, False, False, True, False, False]
    x = numpy.array([1, 13, 1, 1, 13, 4, 1, 4, 4, 4, 1, 4, 13, 4, 4, 1, 13, 4, 4, 4, 4, 13,
                     4, 13, 1, 4, 4, 13, 13, 4, 13, 13, 1, 4, 4, 4, 13, 4, 13, 4, 13, 13, 13,
                     1, 13, 13, 1, 13, 13, 13, 13])

    chisq, pval = compare_survival(y, x)
    expected_chisq = 0.0398155305699465
    expected_pval = 0.980289085821579

    assert round(abs(chisq - expected_chisq), 6) == 0
    assert round(abs(pval - expected_pval), 6) == 0


def test_logrank_four():
    y = numpy.empty(40, dtype=[("status", bool), ("time", float)])
    y["time"] = [95, 54, 94, 72, 52, 68, 53, 53, 21, 69, 58, 67, 65, 22, 95, 78, 86,
                 18, 35, 43, 72, 88, 24, 53, 3, 41, 36, 3, 55, 89, 53, 8, 13, 94, 12, 64, 9, 56, 8, 7]
    y["status"] = [False, False, True, False, False, True, False, True, False, True,
                   False, True, False, True, True, True, True, True, False, True, False,
                   True, False, True, True, False, False, True, True, False, True, False,
                   True, True, False, False, True, True, True, False]
    x = numpy.array([3, 3, 2, -1, 3, -1, 2, 0, -1, 2, 2, -1, -1, 0, 2, 2, -1, -1, -1, 3, -1,
                     -1, 2, -1, 3, 2, -1, 2, 0, 2, 2, 2, 2, 3, -1, 2, -1, 2, 2, 2])

    chisq, pval = compare_survival(y, x)
    expected_chisq = 5.25844160740909
    expected_pval = 0.153821897350625

    assert round(abs(chisq - expected_chisq), 6) == 0
    assert round(abs(pval - expected_pval), 6) == 0


def test_groups():
    y = numpy.empty(shape=7, dtype=[("event", numpy.bool_), ("time", numpy.float_)])
    y["time"] = numpy.arange(1, 8)
    y["event"] = True
    group = numpy.ones(7)

    with pytest.raises(ValueError,
                       match="At least two groups must be specified, "
                             "but only one was provided."):
        compare_survival(y, group)
