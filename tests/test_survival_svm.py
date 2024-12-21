from functools import partial
from os.path import dirname, join
import warnings

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest
from sklearn.decomposition import KernelPCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, normalize

from sksurv.bintrees import AVLTree, RBTree
from sksurv.column import encode_categorical
from sksurv.datasets import get_x_y, load_whas500
from sksurv.exceptions import NoComparablePairException
from sksurv.io import loadarff
from sksurv.kernels import ClinicalKernelTransform
from sksurv.metrics import concordance_index_censored
from sksurv.svm._prsvm import survival_constraints_simple
from sksurv.svm.naive_survival_svm import NaiveSurvivalSVM
from sksurv.svm.survival_svm import (
    FastKernelSurvivalSVM,
    FastSurvivalSVM,
    OrderStatisticTreeSurvivalCounter,
    SurvivalCounter,
)
from sksurv.testing import FixtureParameterFactory, assert_cindex_almost_equal
from sksurv.util import Surv

WHAS500_NOTIES_FILE = join(dirname(__file__), "data", "whas500-noties.arff")


@pytest.fixture(
    params=[
        "simple",
        "PRSVM",
        "direct-count",
        "rbtree",
        "avltree",
    ]
)
def optimizer_any(request):
    return request.param


@pytest.fixture(
    params=[
        "direct-count",
        "rbtree",
        "avltree",
    ]
)
def optimizer_regression(request):
    return request.param


@pytest.mark.parametrize(
    "svm_cls,expected_optimizer", [(FastSurvivalSVM, "avltree"), (FastKernelSurvivalSVM, "rbtree")]
)
def test_default_optimizer(svm_cls, expected_optimizer, make_whas500):
    whas500 = make_whas500(to_numeric=True)
    ssvm = svm_cls(tol=1e-4, max_iter=25)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        ssvm.fit(whas500.x, whas500.y)
    assert ssvm.optimizer == expected_optimizer
    assert 1 < ssvm.n_iter_ <= 25


@pytest.mark.parametrize("svm_cls", [FastSurvivalSVM, FastKernelSurvivalSVM])
def test_unknown_optimizer(svm_cls, fake_data):
    x, y = fake_data

    ssvm = svm_cls(optimizer="random stuff")
    msg = (
        f"The 'optimizer' parameter of {svm_cls.__name__} must be "
        r"a str among \{.+\} or None\. Got 'random stuff' instead\."
    )
    with pytest.raises(ValueError, match=msg):
        ssvm.fit(x, y)


class FastSurvivalSVMFailureCases(FixtureParameterFactory):
    _prefix = "The '{name}' parameter of {estimator} must be "

    @property
    def x(self):
        return np.arange(80).reshape(10, 8)

    def data_alpha_negative(self):
        params = {"alpha": -1}
        msg = (
            "The 'alpha' parameter of Fast(Kernel)?SurvivalSVM must be "
            r"a float in the range \(0\.0, inf\)\. "
            r"Got -1 instead\."
        )
        error = pytest.raises(ValueError, match=msg)
        return params, None, None, error

    def _rank_ratio_out_of_bounds(self, rank_ratio):
        params = {"rank_ratio": rank_ratio}
        msg = (
            "The 'rank_ratio' parameter of Fast(Kernel)?SurvivalSVM must be "
            r"a float in the range \[0\.0, 1\.0\]\. "
            f"Got {rank_ratio!r} instead\\."
        )
        error = pytest.raises(ValueError, match=msg)
        return params, None, None, error

    def data_rank_ratio_out_of_bounds_0(self):
        return self._rank_ratio_out_of_bounds(-1)

    def data_rank_ratio_out_of_bounds_1(self):
        return self._rank_ratio_out_of_bounds(1.2)

    def data_rank_ratio_out_of_bounds_2(self):
        return self._rank_ratio_out_of_bounds(np.nan)

    def data_rank_ratio_out_of_bounds_3(self):
        return self._rank_ratio_out_of_bounds(np.inf)

    def _regression_not_supported(self, optimizer):
        params = {"rank_ratio": 0, "optimizer": optimizer}
        error = pytest.raises(ValueError, match=f"optimizer {optimizer!r} does not implement regression objective")
        return params, None, None, error

    def data_regression_not_supported_simple(self):
        return self._regression_not_supported("simple")

    def data_regression_not_supported_prsvm(self):
        return self._regression_not_supported("PRSVM")

    def _y_invalid(self, y):
        params = {}
        error = pytest.raises(
            ValueError,
            match="y must be a structured array with the first field"
            " being a binary class event indicator and the second field"
            " the time of the event/censoring",
        )
        return params, self.x, y, error

    def data_y_invalid_0(self):
        return self._y_invalid([np.ones(10, dtype=bool), np.arange(10)])

    def data_y_invalid_1(self):
        return self._y_invalid(np.ones(10, dtype=int))

    def data_y_invalid_2(self):
        return self._y_invalid(np.ones(dtype=[("event", bool)], shape=10))

    def data_y_invalid_3(self):
        return self._y_invalid(np.ones(dtype=[("event", bool), ("time", float), ("too_much", int)], shape=10))

    def _invalid_event(self, event):
        params = {}

        y = np.empty(dtype=[("event", int), ("time", float)], shape=10)
        y["event"] = event
        y["time"] = np.arange(10)

        error = pytest.raises(ValueError, match="elements of event indicator must be boolean, but found int")
        return params, self.x, y, error

    def data_event_not_boolean(self):
        event = np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1], dtype=int)
        return self._invalid_event(event)

    def data_event_not_binary(self):
        event = np.array([0, 1, 2, 1, 1, 0, 1, 2, 3, 1], dtype=int)
        return self._invalid_event(event)

    def data_time_not_numeric(self):
        params = {}
        y = np.empty(dtype=[("event", bool), ("time", bool)], shape=10)
        y["event"] = np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1], dtype=bool)
        y["time"] = np.ones(10, dtype=bool)

        error = pytest.raises(ValueError, match="time must be numeric, but found bool")
        return params, self.x, y, error

    def data_all_censored(self):
        params = {}
        y = Surv.from_arrays(np.zeros(10, dtype=bool), [0, 1, 2, 1, 1, 0, 1, 2, 3, 1])

        error = pytest.raises(ValueError, match="all samples are censored")
        return params, self.x, y, error

    def data_zero_time(self):
        params = {"rank_ratio": 0.5}
        y = Surv.from_arrays([0, 1, 0, 1, 1, 0, 1, 0, 0, 1], [0, 1, 2, 1, 1, 0, 1, 2, 3, 1])

        error = pytest.raises(ValueError, match="observed time contains values smaller or equal to zero")
        return params, self.x, y, error

    def data_negative_time(self):
        params = {"rank_ratio": 0.5}
        y = Surv.from_arrays([0, 1, 0, 1, 1, 0, 1, 0, 0, 1], [1, 1, -2, 1, 1, 6, 1, 2, 3, 1])

        error = pytest.raises(ValueError, match="observed time contains values smaller or equal to zero")
        return params, self.x, y, error

    def data_ranking_with_fit_intercept(self):
        params = {"rank_ratio": 1.0, "fit_intercept": True}
        y = Surv.from_arrays(np.ones(10, dtype=bool), np.arange(1, 11, dtype=float))

        error = pytest.raises(ValueError, match="fit_intercept=True is only meaningful if rank_ratio < 1.0")
        return params, self.x, y, error


class FastSurvivalSVMFitAndPredictCases(FixtureParameterFactory):
    @property
    def params(self):
        return {"max_iter": 50, "random_state": 0}

    def data_hybrid(self):
        params = self.params
        params.update({"rank_ratio": 0.5, "fit_intercept": True})

        expected_intercept = 6.1409367385513729
        expected_coef = np.array(
            [
                -0.0209254120718,
                -0.265768317208,
                -0.154254689136,
                0.0800600947891,
                -0.290121131022,
                -0.0288851785213,
                0.0998004550073,
                0.0454100937492,
                -0.125863947621,
                0.0343588337797,
                -0.000710219364914,
                0.0546969104996,
                -0.5375338235,
                -0.0137995110308,
            ]
        )
        expected_rmse = 780.52617631863893
        return params, expected_intercept, expected_coef, expected_rmse

    def data_hybrid_no_intercept(self):
        params = self.params
        params.update({"rank_ratio": 0.5, "fit_intercept": False})

        expected_intercept = None
        expected_coef = np.array(
            [
                0.00669121,
                -0.2754864,
                -0.14124808,
                0.0748376,
                -0.2812598,
                0.07543884,
                0.09845683,
                0.08398258,
                -0.12182314,
                0.02637739,
                0.03060149,
                0.11870598,
                -0.52688224,
                -0.01762842,
            ]
        )
        expected_rmse = 1128.4460587629746
        return params, expected_intercept, expected_coef, expected_rmse

    def data_regression(self):
        params = self.params
        params.update({"rank_ratio": 0.0, "fit_intercept": True})

        expected_intercept = 6.4160179606675278
        expected_coef = np.array(
            [
                -0.0730891368237,
                -0.536630355029,
                -0.497411603275,
                0.269039958377,
                -0.730559850692,
                -0.0148443526234,
                0.285916578892,
                0.165960302339,
                -0.301749910087,
                0.334855938531,
                0.0886214732161,
                0.0554890272028,
                -2.12680470014,
                0.0421466831393,
            ]
        )
        expected_rmse = 1206.6556186869332
        return params, expected_intercept, expected_coef, expected_rmse

    def data_regression_no_intercept(self):
        params = self.params
        params.update({"rank_ratio": 0.0, "fit_intercept": False})

        expected_intercept = None
        expected_coef = np.array(
            [
                1.39989875,
                -1.16903161,
                -0.40195857,
                -0.05848903,
                -0.08421557,
                4.11924729,
                0.25135451,
                1.89067276,
                -0.25751401,
                -0.10213143,
                1.56333622,
                3.10136873,
                -2.23644848,
                -0.11620715,
            ]
        )
        expected_rmse = 15838.510668936022
        return params, expected_intercept, expected_coef, expected_rmse


class TestFastSurvivalSVM:
    @staticmethod
    @pytest.mark.parametrize("params,x,y,error", FastSurvivalSVMFailureCases().get_cases())
    def test_failure(params, x, y, error, fake_data):
        x_fake, y_fake = fake_data
        x = x_fake if x is None else x
        y = y_fake if y is None else y

        ssvm = FastSurvivalSVM(**params)
        with error:
            ssvm.fit(x, y)

    @staticmethod
    @pytest.mark.parametrize("optimizer", ["simple", "avltree", "direct-count", "PRSVM", "rbtree"])
    def test_fit_uncomparable(whas500_uncomparable, optimizer):
        ssvm = FastSurvivalSVM(optimizer=optimizer)
        with pytest.raises(NoComparablePairException):
            ssvm.fit(whas500_uncomparable.x, whas500_uncomparable.y)

    @staticmethod
    def test_survial_constraints_no_ties():
        y = np.array([True, True, False, True, False, False, False, False])
        time = np.array([20, 46, 56, 63, 77, 90, 100, 104])

        expected_order = np.arange(len(time)).astype(int)

        expected = np.array(
            [
                [-1, 1, 0, 0, 0, 0, 0, 0],
                [-1, 0, 1, 0, 0, 0, 0, 0],
                [-1, 0, 0, 1, 0, 0, 0, 0],
                [-1, 0, 0, 0, 1, 0, 0, 0],
                [-1, 0, 0, 0, 0, 1, 0, 0],
                [-1, 0, 0, 0, 0, 0, 1, 0],
                [-1, 0, 0, 0, 0, 0, 0, 1],
                [0, -1, 1, 0, 0, 0, 0, 0],
                [0, -1, 0, 1, 0, 0, 0, 0],
                [0, -1, 0, 0, 1, 0, 0, 0],
                [0, -1, 0, 0, 0, 1, 0, 0],
                [0, -1, 0, 0, 0, 0, 1, 0],
                [0, -1, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, -1, 1, 0, 0, 0],
                [0, 0, 0, -1, 0, 1, 0, 0],
                [0, 0, 0, -1, 0, 0, 1, 0],
                [0, 0, 0, -1, 0, 0, 0, 1],
            ],
            dtype=np.int8,
        )

        samples_order = FastSurvivalSVM._argsort_and_resolve_ties(time, None)
        assert_array_equal(expected_order, samples_order)

        A = survival_constraints_simple(np.asarray(y[samples_order], dtype=np.uint8))
        assert_array_equal(expected, A.todense())

    @staticmethod
    def test_survival_constraints_with_ties():
        y = np.array([True, True, False, False, True, False, True, True, False, False, False, True])
        time = np.array([20, 33, 33, 40, 50, 66, 66, 66, 89, 110, 110, 111])

        expected_order = np.array([0, 2, 1, 3, 4, 7, 5, 6, 8, 9, 10, 11])
        samples_order = FastSurvivalSVM._argsort_and_resolve_ties(time, np.random.RandomState(0))
        np.testing.assert_array_equal(expected_order, samples_order)

        expected = np.array(
            [
                [-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [-1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [-1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [-1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [-1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [-1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [-1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [-1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, -1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1],
            ],
            dtype=np.int8,
        )

        A = survival_constraints_simple(np.asarray(y[samples_order], dtype=np.uint8))
        assert_array_equal(expected, A.todense())

    @staticmethod
    @pytest.mark.slow()
    def test_fit_and_predict_ranking(make_whas500, optimizer_any):
        whas500 = make_whas500(to_numeric=True)
        ssvm = FastSurvivalSVM(optimizer=optimizer_any, random_state=0)
        ssvm.fit(whas500.x, whas500.y)

        assert not hasattr(ssvm, "intercept_")
        expected_coef = np.array(
            [
                -0.02066177,
                -0.26449933,
                -0.15205399,
                0.0794547,
                -0.28840498,
                -0.02864288,
                0.09901995,
                0.04505302,
                -0.12512215,
                0.03341365,
                -0.00110442,
                0.05446756,
                -0.53009875,
                -0.01394175,
            ]
        )
        assert_array_almost_equal(expected_coef, ssvm.coef_)

        assert whas500.x.shape[1] == ssvm.coef_.shape[0]

        c = ssvm.score(whas500.x, whas500.y)

        assert pytest.approx(0.7860650174985695) == c

    @staticmethod
    @pytest.mark.slow()
    @pytest.mark.parametrize(
        "params,expected_intercept,expected_coef,expected_rmse",
        FastSurvivalSVMFitAndPredictCases().get_cases(),
    )
    def test_fit_and_predict(
        params,
        expected_intercept,
        expected_coef,
        expected_rmse,
        make_whas500,
        optimizer_regression,
    ):
        whas500 = make_whas500(to_numeric=True)

        ssvm = FastSurvivalSVM(optimizer=optimizer_regression, **params)
        ssvm.fit(whas500.x, whas500.y)

        if expected_intercept is None:
            assert not hasattr(ssvm, "intercept_")
        else:
            assert pytest.approx(expected_intercept, 1e-7) == ssvm.intercept_

        assert_array_almost_equal(expected_coef, ssvm.coef_)

        pred = ssvm.predict(whas500.x)
        rmse = np.sqrt(mean_squared_error(whas500.y["lenfol"], pred))
        assert pytest.approx(expected_rmse, 1e-7) == rmse

    @staticmethod
    @pytest.mark.slow()
    def test_fit_timeit(make_whas500, optimizer_any):
        whas500 = make_whas500(to_numeric=True)
        idx = np.random.RandomState(0).choice(np.arange(whas500.x.shape[0]), replace=False, size=100)

        ssvm = FastSurvivalSVM(optimizer=optimizer_any, timeit=3, random_state=0)
        ssvm.fit(whas500.x[idx, :], whas500.y[idx])

        assert "timings" in ssvm.optimizer_result_


class TestKernelSurvivalSVM:
    @staticmethod
    @pytest.mark.parametrize("kernel", ["linear", "precomputed"])
    def test_fit_and_predict_linear(kernel, make_whas500):
        whas500 = make_whas500(to_numeric=True)
        ssvm = FastKernelSurvivalSVM(optimizer="rbtree", kernel=kernel, random_state=0)
        if kernel == "precomputed":
            x = np.dot(whas500.x, whas500.x.T)
        else:
            x = whas500.x
        ssvm.fit(x, whas500.y)

        assert ssvm.__sklearn_tags__().input_tags.pairwise is (kernel == "precomputed")

        assert whas500.x.shape[0] == ssvm.coef_.shape[0]

        i = np.arange(250)
        np.random.RandomState(0).shuffle(i)
        c = ssvm.score(x[i], whas500.y[i])
        assert c == pytest.approx(0.76923445664157997)

    @staticmethod
    @pytest.mark.parametrize("kernel", ["linear", "precomputed"])
    def test_fit_and_predict_linear_regression(kernel, make_whas500):
        whas500 = make_whas500(to_numeric=True)
        ssvm = FastKernelSurvivalSVM(
            optimizer="rbtree",
            rank_ratio=0.0,
            kernel=kernel,
            max_iter=50,
            tol=1e-8,
            fit_intercept=True,
            random_state=0xF1,
        )
        if kernel == "precomputed":
            x = np.dot(whas500.x, whas500.x.T)
        else:
            x = whas500.x
        ssvm.fit(x, whas500.y)

        assert ssvm.__sklearn_tags__().input_tags.pairwise is (kernel == "precomputed")

        assert float(ssvm.intercept_) == pytest.approx(6.416017539824949, 1e-5)

        i = np.arange(250)
        np.random.RandomState(0).shuffle(i)
        pred = ssvm.predict(x[i])
        rmse = np.sqrt(mean_squared_error(whas500.y["lenfol"][i], pred))
        assert rmse <= 1342.274550652291 + 0.293

        c = ssvm.score(x[i], whas500.y[i])
        assert c == pytest.approx(0.7630027323714108)

    @staticmethod
    def test_fit_and_predict_linear_regression_no_intercept(make_whas500):
        whas500 = make_whas500(to_numeric=True)
        ssvm = FastKernelSurvivalSVM(
            optimizer="rbtree", rank_ratio=0.0, kernel="linear", max_iter=50, fit_intercept=False, random_state=0
        )
        ssvm.fit(whas500.x, whas500.y)

        assert not hasattr(ssvm, "intercept_")

        pred = ssvm.predict(whas500.x)
        rmse = np.sqrt(mean_squared_error(whas500.y["lenfol"], pred))
        assert rmse == pytest.approx(15837.658418546907, 1e-4)

    @staticmethod
    @pytest.mark.slow()
    @pytest.mark.parametrize("optimizer", ["rbtree", "avltree"])
    @pytest.mark.filterwarnings("ignore:Optimization did not converge.*:sklearn.exceptions.ConvergenceWarning")
    def test_fit_and_predict_rbf(make_whas500, optimizer):
        whas500 = make_whas500(to_numeric=True)
        ssvm = FastKernelSurvivalSVM(optimizer=optimizer, kernel="rbf", tol=2e-6, max_iter=75, random_state=0)
        ssvm.fit(whas500.x, whas500.y)

        assert not ssvm.__sklearn_tags__().input_tags.pairwise
        assert whas500.x.shape[0] == ssvm.coef_.shape[0]

        c = ssvm.score(whas500.x, whas500.y)
        assert c >= 0.965

    @staticmethod
    @pytest.mark.slow()
    @pytest.mark.filterwarnings("ignore:Optimization did not converge.*:sklearn.exceptions.ConvergenceWarning")
    def test_fit_and_predict_regression_rbf(make_whas500):
        whas500 = make_whas500(to_numeric=True)
        ssvm = FastKernelSurvivalSVM(
            optimizer="rbtree", rank_ratio=0.0, kernel="rbf", tol=1e-6, max_iter=50, fit_intercept=True, random_state=0
        )
        ssvm.fit(whas500.x, whas500.y)

        assert not ssvm.__sklearn_tags__().input_tags.pairwise
        assert ssvm.intercept_ == pytest.approx(4.9267218894089533, 1e-7)

        pred = ssvm.predict(whas500.x)
        rmse = np.sqrt(mean_squared_error(whas500.y["lenfol"], pred))
        assert rmse == pytest.approx(783.525277)

    @staticmethod
    @pytest.mark.slow()
    @pytest.mark.filterwarnings("ignore:Optimization did not converge.*:sklearn.exceptions.ConvergenceWarning")
    def test_fit_and_predict_hybrid_polynomial(make_whas500):
        whas500 = make_whas500(with_mean=False, with_std=False, to_numeric=True)
        X = MinMaxScaler(feature_range=(0, 1)).fit_transform(whas500.x)

        ssvm = FastKernelSurvivalSVM(
            optimizer="rbtree",
            rank_ratio=0.5,
            kernel="poly",
            coef0=0,
            degree=2,
            max_iter=100,
            fit_intercept=True,
            random_state=0,
        )
        ssvm.fit(X, whas500.y)

        assert not ssvm.__sklearn_tags__().input_tags.pairwise
        assert pytest.approx(6.482593184472981, 1e-5) == ssvm.intercept_

        pred = ssvm.predict(X)
        rmse = np.sqrt(mean_squared_error(whas500.y["lenfol"], pred))
        assert pytest.approx(766.2061731844626, 1e-5) == rmse

    @staticmethod
    @pytest.mark.slow()
    @pytest.mark.filterwarnings("ignore:Optimization did not converge.*:sklearn.exceptions.ConvergenceWarning")
    def test_fit_and_predict_clinical_kernel(make_whas500):
        whas500 = make_whas500(to_numeric=True)

        trans = ClinicalKernelTransform()
        trans.fit(whas500.x_data_frame)

        ssvm = FastKernelSurvivalSVM(
            optimizer="rbtree", kernel=trans.pairwise_kernel, tol=7e-7, max_iter=100, random_state=0
        )
        ssvm.fit(whas500.x, whas500.y)

        assert not ssvm.__sklearn_tags__().input_tags.pairwise
        assert whas500.x.shape[0] == ssvm.coef_.shape[0]

        c = ssvm.score(whas500.x, whas500.y)
        assert c >= 0.854

    @staticmethod
    def _fit_and_compare(model_linear, model_kernel, x, xt, y):
        model_kernel.fit(x, y)
        pred_kernel = model_kernel.predict(x)

        model_linear.fit(xt, y)
        pred_linear = model_linear.predict(xt)

        assert len(pred_linear) == len(pred_kernel)

        expected_cindex = concordance_index_censored(y["fstat"], y["lenfol"], pred_linear)
        assert_cindex_almost_equal(y["fstat"], y["lenfol"], pred_kernel, expected_cindex)

    @staticmethod
    @pytest.mark.slow()
    @pytest.mark.filterwarnings("ignore:Optimization did not converge.*:sklearn.exceptions.ConvergenceWarning")
    def test_compare_builtin_kernel(make_whas500):
        whas500 = make_whas500(to_numeric=True)
        x = normalize(whas500.x)

        rsvm = FastKernelSurvivalSVM(
            optimizer="rbtree",
            kernel="poly",
            gamma=0.5,
            degree=2,
            coef0=0,
            tol=2.5e-8,
            max_iter=100,
            random_state=0xF38,
        )

        xt = KernelPCA(kernel="poly", copy_X=True, gamma=0.5, degree=2, coef0=0, random_state=0xF38).fit_transform(x)
        nrsvm = FastSurvivalSVM(optimizer="rbtree", tol=2.5e-8, max_iter=100, random_state=0xF38)

        TestKernelSurvivalSVM._fit_and_compare(nrsvm, rsvm, x, xt, whas500.y)

    @staticmethod
    @pytest.mark.slow()
    @pytest.mark.filterwarnings("ignore:Optimization did not converge.*:sklearn.exceptions.ConvergenceWarning")
    def test_compare_clinical_kernel(make_whas500):
        whas500 = make_whas500(to_numeric=True)

        trans = ClinicalKernelTransform().fit(whas500.x_data_frame)

        xt = KernelPCA(kernel=trans.pairwise_kernel, copy_X=True).fit_transform(whas500.x)

        nrsvm = FastSurvivalSVM(optimizer="rbtree", tol=1e-8, max_iter=500, random_state=0)

        rsvm = FastKernelSurvivalSVM(
            optimizer="rbtree", kernel=trans.pairwise_kernel, tol=1e-8, max_iter=500, random_state=0
        )
        TestKernelSurvivalSVM._fit_and_compare(nrsvm, rsvm, whas500.x, xt, whas500.y)

    @staticmethod
    def test_fit_precomputed_kernel_invalid_shape(fake_data):
        x, y = fake_data
        ssvm = FastKernelSurvivalSVM(optimizer="rbtree", kernel="precomputed", random_state=0)

        with pytest.raises(
            ValueError,
            match=r"Precomputed metric requires shape \(n_queries, n_indexed\)\. Got \(100, 11\) for 100 indexed\.",
        ):
            ssvm.fit(x, y)

    @staticmethod
    def test_fit_precomputed_kernel_not_symmetric():
        ssvm = FastKernelSurvivalSVM(optimizer="rbtree", kernel="precomputed", random_state=0)
        x = np.random.randn(100, 100)
        x[10, 12] = -1
        x[12, 10] = 9
        y = Surv.from_arrays(np.ones(100).astype(bool), np.ones(100))

        with pytest.raises(ValueError, match="kernel matrix is not symmetric"):
            ssvm.fit(x, y)

    @staticmethod
    def test_predict_precomputed_kernel_invalid_shape(make_whas500):
        whas500 = make_whas500(to_numeric=True)
        ssvm = FastKernelSurvivalSVM(optimizer="rbtree", kernel="precomputed", random_state=0)
        x = np.dot(whas500.x, whas500.x.T)
        ssvm.fit(x, whas500.y)

        x_new = np.random.randn(100, 14)
        with pytest.raises(
            ValueError,
            match=r"Precomputed metric requires shape \(n_queries, n_indexed\)\. Got \(100, 14\) for 500 indexed\.",
        ):
            ssvm.predict(x_new)

    @staticmethod
    @pytest.mark.parametrize("optimizer", ["avltree", "rbtree"])
    def test_fit_uncomparable(whas500_uncomparable, optimizer):
        ssvm = FastKernelSurvivalSVM(optimizer=optimizer)
        with pytest.raises(NoComparablePairException):
            ssvm.fit(whas500_uncomparable.x, whas500_uncomparable.y)


@pytest.fixture(
    params=[
        SurvivalCounter,
        partial(OrderStatisticTreeSurvivalCounter, tree_class=RBTree),
        partial(OrderStatisticTreeSurvivalCounter, tree_class=AVLTree),
    ]
)
def make_survival_counter(request):
    def _make_survival_counter(*args, **kwargs):
        cls = request.param
        if isinstance(cls, partial):
            kwargs.pop("n_relevance_levels")

        counter = cls(*args, **kwargs)
        return counter

    return _make_survival_counter


@pytest.fixture()
def counter_data_01():
    w = np.array([-0.9, -0.7, -0.1, 0.15, 0.2, 1.6])
    y = np.array([2, 0, 4, 3, 5, 1])
    event = np.array([True, True, False, True, False, True])
    x = np.eye(6)
    v = np.arange(6)
    return x, y, event, w, v


@pytest.fixture()
def counter_data_02():
    w = np.array([-0.9, -0.7, -0.1, 0.15, 0.2, 0.3, 0.8, 1.6, 1.85, 2.3])
    y = np.array([3, 0, 4, 6, 8, 5, 1, 7, 2, 9])
    event = np.array([0, 0, 0, 1, 0, 1, 1, 0, 1, 0], dtype=bool)
    x = np.eye(10)
    v = np.arange(10)
    return x, y, event, w, v


class TestSurvivalCounter:
    @staticmethod
    def test_calculate_01(make_survival_counter, counter_data_01):
        x, y, event, w, v = counter_data_01
        counter = make_survival_counter(x, y, event, n_relevance_levels=6)
        counter.update_sort_order(w)

        l_plus, xv_plus, l_minus, xv_minus = counter.calculate(v)

        assert_array_equal(np.array([1, 4, 0, 2, 0, 4]), l_plus)
        assert_array_equal(np.array([2, 9, 0, 6, 0, 9]), xv_plus)
        assert_array_equal(np.array([2, 0, 4, 2, 3, 0]), l_minus)
        assert_array_equal(np.array([6, 0, 9, 6, 9, 0]), xv_minus)

    @staticmethod
    def test_calculate_01_reverse(make_survival_counter, counter_data_01):
        x, y, event, w, v = counter_data_01
        counter = make_survival_counter(x, y[::-1], event[::-1], n_relevance_levels=6)
        counter.update_sort_order(w[::-1])

        l_plus, xv_plus, l_minus, xv_minus = counter.calculate(v[::-1])

        assert_array_equal(np.array([4, 0, 2, 0, 4, 1]), l_plus)
        assert_array_equal(np.array([9, 0, 6, 0, 9, 2]), xv_plus)
        assert_array_equal(np.array([0, 3, 2, 4, 0, 2]), l_minus)
        assert_array_equal(np.array([0, 9, 6, 9, 0, 6]), xv_minus)

    @staticmethod
    def test_calculate_02(make_survival_counter, counter_data_02):
        x, y, event, w, v = counter_data_02
        counter = make_survival_counter(x, y, event, n_relevance_levels=10)
        counter.update_sort_order(w)

        l_plus, xv_plus, l_minus, xv_minus = counter.calculate(v)

        assert_array_equal(np.array([0, 0, 0, 1, 0, 2, 6, 0, 7, 0]), l_plus)
        assert_array_equal(np.array([0, 0, 0, 4, 0, 7, 21, 0, 30, 0]), xv_plus)
        assert_array_equal(np.array([2, 0, 2, 3, 4, 2, 0, 2, 0, 1]), l_minus)
        assert_array_equal(np.array([14, 0, 14, 19, 22, 14, 0, 14, 0, 8]), xv_minus)

    @staticmethod
    def test_calculate_02_reverse(make_survival_counter, counter_data_02):
        x, y, event, w, v = counter_data_02
        counter = make_survival_counter(x, y[::-1], event[::-1], n_relevance_levels=10)
        counter.update_sort_order(w[::-1])

        l_plus, xv_plus, l_minus, xv_minus = counter.calculate(v[::-1])

        assert_array_equal(np.array([0, 7, 0, 6, 2, 0, 1, 0, 0, 0]), l_plus)
        assert_array_equal(np.array([0, 30, 0, 21, 7, 0, 4, 0, 0, 0]), xv_plus)
        assert_array_equal(np.array([1, 0, 2, 0, 2, 4, 3, 2, 0, 2]), l_minus)
        assert_array_equal(np.array([8, 0, 14, 0, 14, 22, 19, 14, 0, 14]), xv_minus)


@pytest.fixture()
def whas500_without_ties():
    # naive survival SVM does resolve ties in survival time differently,
    # therefore use data without ties
    data = loadarff(WHAS500_NOTIES_FILE)
    x, y = get_x_y(data, ["fstat", "lenfol"], "1")
    x = encode_categorical(x)
    return x, y


@pytest.fixture()
def whas500_with_ties():
    # naive survival SVM does resolve ties in survival time differently,
    # therefore use data without ties
    x, y = load_whas500()
    x = normalize(encode_categorical(x))
    return x, y


class TestNaiveSurvivalSVM:
    @staticmethod
    def test_survival_squared_hinge_loss(whas500_without_ties):
        x, y = whas500_without_ties

        nrsvm = NaiveSurvivalSVM(
            loss="squared_hinge",
            dual=False,
            tol=8e-7,
            max_iter=1000,
            random_state=0,
        )
        nrsvm.fit(x, y)

        assert nrsvm.n_iter_ > 10

        rsvm = FastSurvivalSVM(optimizer="avltree", tol=8e-7, max_iter=1000, random_state=0)
        rsvm.fit(x, y)

        assert_array_almost_equal(nrsvm.coef_.ravel(), rsvm.coef_, 3)

        pred_nrsvm = nrsvm.predict(x)
        pred_rsvm = rsvm.predict(x)

        assert len(pred_nrsvm) == len(pred_rsvm)

        expected_cindex = concordance_index_censored(y["fstat"], y["lenfol"], pred_nrsvm)
        assert_cindex_almost_equal(y["fstat"], y["lenfol"], pred_rsvm, expected_cindex)

    @staticmethod
    def test_fit_with_ties(whas500_with_ties):
        x, y = whas500_with_ties

        nrsvm = NaiveSurvivalSVM(
            loss="squared_hinge",
            dual=False,
            tol=1e-8,
            max_iter=1000,
            random_state=0,
        )
        nrsvm.fit(x, y)

        assert nrsvm.coef_.shape == (1, 14)

        cindex = nrsvm.score(x, y)
        assert cindex == pytest.approx(0.7760582309811175, 1e-7)

    @staticmethod
    def test_fit_uncomparable(whas500_uncomparable):
        ssvm = NaiveSurvivalSVM(
            loss="squared_hinge",
            dual=False,
            tol=1e-8,
            max_iter=1000,
            random_state=0,
        )
        with pytest.raises(NoComparablePairException):
            ssvm.fit(whas500_uncomparable.x, whas500_uncomparable.y)
