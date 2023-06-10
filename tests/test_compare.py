import numpy as np
from numpy.testing import assert_almost_equal
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from sksurv.compare import compare_survival
from sksurv.datasets import load_veterans_lung_cancer
from sksurv.testing import FixtureParameterFactory
from sksurv.util import Surv


def assert_stats_frame_equal(computed_stats, expected_stats):
    expected_stats["statistic"] = expected_stats["observed"] - expected_stats["expected"]
    assert_frame_equal(computed_stats, expected_stats)


def test_logrank_rossi(rossi):
    chisq, pval, stats, covar = compare_survival(rossi.y, rossi.x.loc[:, "race"], return_stats=True)

    expected_stats = pd.DataFrame(
        {
            "counts": np.array([53, 379], dtype=np.intp),
            "observed": np.array([12, 102]),
            "expected": np.array([14.709347908949, 99.2906520910509]),
        },
        index=pd.Index([0, 1], name="group", dtype=object),
    )
    assert_stats_frame_equal(stats, expected_stats)

    expected_var = np.array([[12.7417883289863, -12.7417883289863], [-12.7417883289863, 12.7417883289863]])
    assert_almost_equal(covar, expected_var)

    expected_chisq = 0.576101713683918
    expected_pval = 0.447844394252168

    assert chisq == pytest.approx(expected_chisq)
    assert pval == pytest.approx(expected_pval)


def test_logrank_veterans():
    X, y = load_veterans_lung_cancer()

    chisq, pval, stats, covar = compare_survival(y, X.loc[:, "Celltype"], return_stats=True)

    expected_stats = pd.DataFrame(
        columns=["counts", "observed", "expected", "statistic"],
        index=pd.Index(["adeno", "large", "smallcell", "squamous"], name="group"),
    )
    expected_stats["counts"] = np.array([27, 27, 48, 35], dtype=np.intp)
    expected_stats["observed"] = np.array([26, 26, 45, 31], dtype=int)
    expected_stats["expected"] = [15.6937646143605, 34.5494783863493, 30.1020793268148, 47.6546776724754]
    assert_stats_frame_equal(stats, expected_stats)

    expected_var = np.array(
        [
            [12.9661700605014, -4.07011754397142, -4.40872930298506, -4.48732321354496],
            [-4.07011754397142, 24.1990352938484, -7.81168661717217, -12.3172311327048],
            [-4.40872930298506, -7.81168661717217, 21.7542679406138, -9.53385202045655],
            [-4.48732321354496, -12.3172311327048, -9.53385202045655, 26.3384063667063],
        ]
    )
    assert_almost_equal(covar, expected_var)

    expected_chisq = 25.4037003457854
    expected_pval = 1.27124593900609e-05

    assert chisq == pytest.approx(expected_chisq)
    assert pval == pytest.approx(expected_pval)


class LogRankToyCases(FixtureParameterFactory):
    @property
    def time_1(self):
        return [12, 1, 6, 9, 7, 4, 13, 5, 2, 3]

    @property
    def time_2(self):
        return [12, 1, 2, 2, 7, 4, 1, 5, 2, 5]

    @property
    def two_groups(self):
        return np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    def data_two_groups_1(self):
        status = [False, True, True, False, False, False, True, False, True, True]
        y = Surv.from_arrays(status, self.time_1)

        expected_chisq = 0.0658861965695617
        expected_pval = 0.797423444741849
        return (y, self.two_groups), (expected_chisq, expected_pval)

    def data_two_groups_2(self):
        status = [False, False, True, False, False, False, True, False, True, True]
        y = Surv.from_arrays(status, self.time_1)

        expected_chisq = 0.843638413831986
        expected_pval = 0.358358251735077
        return (y, self.two_groups), (expected_chisq, expected_pval)

    def data_two_groups_3(self):
        status = [False, False, True, False, False, False, True, False, True, True]
        y = Surv.from_arrays(status, self.time_2)

        expected_chisq = 1.07692307692308
        expected_pval = 0.299386905803601
        return (y, self.two_groups), (expected_chisq, expected_pval)

    def data_two_groups_4(self):
        status = [False, False, True, True, False, False, False, True, False, True]
        y = Surv.from_arrays(status, self.time_2)

        expected_chisq = 0.0
        expected_pval = 1.0
        return (y, self.two_groups), (expected_chisq, expected_pval)

    def data_two_groups_5(self):
        status = [True, True, False, False, False, False, False, True, False, True]
        time = [12, 12, 2, 2, 7, 4, 1, 5, 2, 5]
        y = Surv.from_arrays(status, time)

        expected_chisq = 4.0
        expected_pval = 0.0455002638963585
        return (y, self.two_groups), (expected_chisq, expected_pval)

    def data_three_groups(self):
        time = [
            88,
            62,
            81,
            64,
            19,
            35,
            49,
            82,
            57,
            6,
            97,
            52,
            4,
            90,
            16,
            92,
            4,
            59,
            90,
            71,
            39,
            94,
            68,
            77,
            67,
            15,
            67,
            7,
            32,
            14,
            39,
            26,
            24,
            2,
            4,
            95,
            2,
            12,
            94,
            64,
            68,
            16,
            99,
            6,
            57,
            81,
            53,
            33,
            31,
            91,
            91,
        ]
        status = [
            False,
            False,
            True,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            True,
            False,
            True,
            True,
            False,
            False,
            False,
            True,
            False,
            False,
            True,
            False,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            False,
            True,
            False,
            False,
        ]
        y = Surv.from_arrays(status, time)
        x = np.array(
            [
                1,
                13,
                1,
                1,
                13,
                4,
                1,
                4,
                4,
                4,
                1,
                4,
                13,
                4,
                4,
                1,
                13,
                4,
                4,
                4,
                4,
                13,
                4,
                13,
                1,
                4,
                4,
                13,
                13,
                4,
                13,
                13,
                1,
                4,
                4,
                4,
                13,
                4,
                13,
                4,
                13,
                13,
                13,
                1,
                13,
                13,
                1,
                13,
                13,
                13,
                13,
            ]
        )

        expected_chisq = 0.0398155305699465
        expected_pval = 0.980289085821579
        return (y, x), (expected_chisq, expected_pval)

    def data_four_groups(self):
        time = [
            95,
            54,
            94,
            72,
            52,
            68,
            53,
            53,
            21,
            69,
            58,
            67,
            65,
            22,
            95,
            78,
            86,
            18,
            35,
            43,
            72,
            88,
            24,
            53,
            3,
            41,
            36,
            3,
            55,
            89,
            53,
            8,
            13,
            94,
            12,
            64,
            9,
            56,
            8,
            7,
        ]
        status = [
            False,
            False,
            True,
            False,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            True,
            True,
            True,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            True,
            False,
            False,
            True,
            True,
            False,
            True,
            False,
            True,
            True,
            False,
            False,
            True,
            True,
            True,
            False,
        ]
        y = Surv.from_arrays(status, time)
        x = np.array(
            [
                3,
                3,
                2,
                -1,
                3,
                -1,
                2,
                0,
                -1,
                2,
                2,
                -1,
                -1,
                0,
                2,
                2,
                -1,
                -1,
                -1,
                3,
                -1,
                -1,
                2,
                -1,
                3,
                2,
                -1,
                2,
                0,
                2,
                2,
                2,
                2,
                3,
                -1,
                2,
                -1,
                2,
                2,
                2,
            ]
        )

        expected_chisq = 5.25844160740909
        expected_pval = 0.153821897350625
        return (y, x), (expected_chisq, expected_pval)


@pytest.mark.parametrize("inputs,expectation", LogRankToyCases().get_cases())
def test_toy_compare_survival(inputs, expectation):
    chisq, pval = compare_survival(*inputs)

    expected_chisq, expected_pval = expectation
    assert chisq == pytest.approx(expected_chisq)
    assert pval == pytest.approx(expected_pval)


class LogRankFailureCases(FixtureParameterFactory):
    def data_single_group(self):
        y = Surv.from_arrays([True] * 7, np.arange(1, 8))
        group = np.ones(7)

        err = pytest.raises(ValueError, match=r"At least two groups must be specified, but only one was provided\.")
        return (y, group), err

    def data_wrong_shape(self):
        y = Surv.from_arrays([True] * 7, np.arange(1, 8))
        group = np.empty((y.shape[0], 2, 4), dtype="str")
        group[:, 0, :] = "a"
        group[:, 1, :] = "b"

        err = pytest.raises(ValueError, match=r"Found array with dim 3\. compare_survival expected <= 2\.")
        return (y, group), err


@pytest.mark.parametrize("inputs,excepted_error", LogRankFailureCases().get_cases())
def test_compare_survival_failures(inputs, excepted_error):
    with excepted_error:
        compare_survival(*inputs)
