from functools import partial
from os.path import join, dirname
import warnings

import numpy
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest
from sklearn.decomposition import KernelPCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize

from sksurv.bintrees import AVLTree, RBTree
from sksurv.column import encode_categorical
from sksurv.datasets import load_whas500, get_x_y
from sksurv.exceptions import NoComparablePairException
from sksurv.io import loadarff
from sksurv.kernels import ClinicalKernelTransform
from sksurv.metrics import concordance_index_censored
from sksurv.svm._prsvm import survival_constraints_simple
from sksurv.svm.naive_survival_svm import NaiveSurvivalSVM
from sksurv.svm.survival_svm import FastSurvivalSVM, FastKernelSurvivalSVM, SurvivalCounter, \
    OrderStatisticTreeSurvivalCounter
from sksurv.util import Surv
from sksurv.testing import assert_cindex_almost_equal

WHAS500_NOTIES_FILE = join(dirname(__file__), 'data', 'whas500-noties.arff')


@pytest.fixture(params=[
    'simple',
    'PRSVM',
    'direct-count',
    'rbtree',
    'avltree',
])
def optimizer_any(request):
    return request.param


@pytest.fixture(params=[
    'direct-count',
    'rbtree',
    'avltree',
])
def optimizer_regression(request):
    return request.param


class TestFastSurvivalSVM(object):

    @staticmethod
    def test_alpha_negative(fake_data):
        x, y = fake_data

        ssvm = FastSurvivalSVM(alpha=-1)
        with pytest.raises(ValueError, match="alpha must be positive"):
            ssvm.fit(x, y)

    @staticmethod
    @pytest.mark.parametrize('value', [-1, 1.2, numpy.nan, numpy.inf])
    def test_rank_ratio_out_of_bounds(fake_data, value):
        x, y = fake_data

        ssvm = FastSurvivalSVM(rank_ratio=value)
        with pytest.raises(ValueError, match=r"rank_ratio must be in \[0; 1\]"):
            ssvm.fit(x, y)

    @staticmethod
    @pytest.mark.parametrize('value', ['simple', 'PRSVM'])
    def test_regression_not_supported(fake_data, value):
        x, y = fake_data

        ssvm = FastSurvivalSVM(rank_ratio=0, optimizer=value)
        with pytest.raises(ValueError,
                           match="optimizer {!r} does not implement regression objective".format(value)):
            ssvm.fit(x, y)

    @staticmethod
    def test_unknown_optimizer(fake_data):
        x, y = fake_data

        ssvm = FastSurvivalSVM(rank_ratio=0, optimizer='random stuff')
        with pytest.raises(ValueError,
                           match="unknown optimizer: random stuff"):
            ssvm.fit(x, y)

    @staticmethod
    @pytest.mark.parametrize('y', [
        [numpy.ones(100, dtype=bool), numpy.arange(100)],
        numpy.ones(100, dtype=int),
        numpy.ones(dtype=[('event', bool)], shape=10),
        numpy.ones(dtype=[('event', bool), ('time', float), ('too_much', int)], shape=10)
    ])
    def test_y_invalid(y):
        x = numpy.zeros((100, 10))

        rsvm = FastSurvivalSVM()
        with pytest.raises(ValueError,
                           match='y must be a structured array with the first field'
                                 ' being a binary class event indicator and the second field'
                                 ' the time of the event/censoring'):
            rsvm.fit(x, y)

    @staticmethod
    def test_event_not_boolean():
        x = numpy.arange(80).reshape(10, 8)
        y = numpy.empty(dtype=[('event', int), ('time', float)], shape=10)
        y['event'] = numpy.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1], dtype=int)
        y['time'] = numpy.arange(10)

        rsvm = FastSurvivalSVM()
        with pytest.raises(ValueError,
                           match="elements of event indicator must be boolean, but found int"):
            rsvm.fit(x, y)

    @staticmethod
    def test_event_not_binary():
        x = numpy.arange(80).reshape(10, 8)
        y = numpy.empty(dtype=[('event', int), ('time', float)], shape=10)
        y['event'] = numpy.array([0, 1, 2, 1, 1, 0, 1, 2, 3, 1], dtype=int)
        y['time'] = numpy.arange(10)

        rsvm = FastSurvivalSVM()
        with pytest.raises(ValueError,
                           match="elements of event indicator must be boolean, but found int"):
            rsvm.fit(x, y)

    @staticmethod
    def test_time_not_numeric():
        x = numpy.arange(80).reshape(10, 8)
        y = numpy.empty(dtype=[('event', bool), ('time', bool)], shape=10)
        y['event'] = numpy.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1], dtype=bool)
        y['time'] = numpy.ones(10, dtype=bool)

        rsvm = FastSurvivalSVM()
        with pytest.raises(ValueError,
                           match="time must be numeric, but found bool"):
            rsvm.fit(x, y)

    @staticmethod
    def test_all_censored():
        x = numpy.arange(80).reshape(10, 8)
        y = Surv.from_arrays(numpy.zeros(10, dtype=bool), [0, 1, 2, 1, 1, 0, 1, 2, 3, 1])

        rsvm = FastSurvivalSVM()
        with pytest.raises(ValueError,
                           match="all samples are censored"):
            rsvm.fit(x, y)

    @staticmethod
    def test_zero_time():
        x = numpy.arange(80).reshape(10, 8)
        y = Surv.from_arrays([0, 1, 0, 1, 1, 0, 1, 0, 0, 1], [0, 1, 2, 1, 1, 0, 1, 2, 3, 1])

        rsvm = FastSurvivalSVM(rank_ratio=0.5)
        with pytest.raises(ValueError,
                           match="observed time contains values smaller or equal to zero"):
            rsvm.fit(x, y)

    @staticmethod
    def test_negative_time():
        x = numpy.arange(80).reshape(10, 8)
        y = Surv.from_arrays([0, 1, 0, 1, 1, 0, 1, 0, 0, 1], [1, 1, -2, 1, 1, 6, 1, 2, 3, 1])

        rsvm = FastSurvivalSVM(rank_ratio=0.5)
        with pytest.raises(ValueError,
                           match="observed time contains values smaller or equal to zero"):
            rsvm.fit(x, y)

    @staticmethod
    def test_ranking_with_fit_intercept():
        x = numpy.zeros((100, 10))
        y = Surv.from_arrays(numpy.ones(100, dtype=bool), numpy.arange(1, 101, dtype=float))

        ssvm = FastSurvivalSVM(rank_ratio=1.0, fit_intercept=True)
        with pytest.raises(ValueError,
                           match="fit_intercept=True is only meaningful if rank_ratio < 1.0"):
            ssvm.fit(x, y)

    @staticmethod
    @pytest.mark.parametrize("optimizer", ("simple", "avltree", "direct-count", "PRSVM", "rbtree"))
    def test_fit_uncomparable(whas500_uncomparable, optimizer):
        ssvm = FastSurvivalSVM(optimizer=optimizer)
        with pytest.raises(NoComparablePairException):
            ssvm.fit(whas500_uncomparable.x, whas500_uncomparable.y)

    @staticmethod
    def test_survial_constraints_no_ties():
        y = numpy.array([True, True, False, True, False, False, False, False])
        time = numpy.array([20, 46, 56, 63, 77, 90, 100, 104])

        expected_order = numpy.arange(len(time)).astype(numpy.int)

        expected = numpy.array([
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
        ], dtype=numpy.int8)

        samples_order = FastSurvivalSVM._argsort_and_resolve_ties(time, None)
        assert_array_equal(expected_order, samples_order)

        A = survival_constraints_simple(numpy.asarray(y[samples_order], dtype=numpy.uint8))
        assert_array_equal(expected, A.todense())

    @staticmethod
    def test_survival_constraints_with_ties():
        y = numpy.array([True, True, False, False, True, False, True, True, False, False, False, True])
        time = numpy.array([20, 33, 33, 40, 50, 66, 66, 66, 89, 110, 110, 111])

        expected_order = numpy.array([0, 2, 1, 3, 4, 7, 5, 6, 8, 9, 10, 11])
        samples_order = FastSurvivalSVM._argsort_and_resolve_ties(time, numpy.random.RandomState(0))
        numpy.testing.assert_array_equal(expected_order, samples_order)

        expected = numpy.array([
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
        ], dtype=numpy.int8)

        A = survival_constraints_simple(numpy.asarray(y[samples_order], dtype=numpy.uint8))
        assert_array_equal(expected, A.todense())

    @staticmethod
    def test_default_optimizer(make_whas500):
        whas500 = make_whas500(to_numeric=True)
        ssvm = FastSurvivalSVM(tol=1e-4, max_iter=25)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            ssvm.fit(whas500.x, whas500.y)
        assert 'avltree' == ssvm.optimizer

    @staticmethod
    @pytest.mark.slow
    def test_fit_and_predict_ranking(make_whas500, optimizer_any):
        whas500 = make_whas500(to_numeric=True)
        ssvm = FastSurvivalSVM(optimizer=optimizer_any, random_state=0)
        ssvm.fit(whas500.x, whas500.y)

        assert not hasattr(ssvm, "intercept_")
        expected_coef = numpy.array([-0.02066177, -0.26449933, -0.15205399, 0.0794547, -0.28840498, -0.02864288,
                                     0.09901995, 0.04505302, -0.12512215, 0.03341365, -0.00110442, 0.05446756,
                                     -0.53009875, -0.01394175])
        assert_array_almost_equal(expected_coef, ssvm.coef_)

        assert whas500.x.shape[1] == ssvm.coef_.shape[0]

        c = ssvm.score(whas500.x, whas500.y)

        assert round(abs(0.7860650174985695 - c), 6) == 0

    @staticmethod
    @pytest.mark.slow
    def test_fit_and_predict_hybrid(make_whas500, optimizer_regression):
        whas500 = make_whas500(to_numeric=True)

        ssvm = FastSurvivalSVM(optimizer=optimizer_regression, rank_ratio=0.5,
                               max_iter=50, fit_intercept=True, random_state=0)
        ssvm.fit(whas500.x, whas500.y)

        assert round(abs(6.1409367385513729 - ssvm.intercept_), 7) == 0
        expected_coef = numpy.array(
            [-0.0209254120718, -0.265768317208, -0.154254689136, 0.0800600947891, -0.290121131022, -0.0288851785213,
             0.0998004550073, 0.0454100937492, -0.125863947621, 0.0343588337797, -0.000710219364914, 0.0546969104996,
             -0.5375338235, -0.0137995110308
             ])
        assert_array_almost_equal(expected_coef, ssvm.coef_)

        pred = ssvm.predict(whas500.x)
        rmse = numpy.sqrt(mean_squared_error(whas500.y['lenfol'], pred))
        assert round(abs(780.52617631863893 - rmse), 7) == 0

    @staticmethod
    @pytest.mark.slow
    def test_fit_and_predict_hybrid_no_intercept(make_whas500, optimizer_regression):
        whas500 = make_whas500(to_numeric=True)

        ssvm = FastSurvivalSVM(optimizer=optimizer_regression, rank_ratio=0.5,
                               max_iter=50, fit_intercept=False, random_state=0)
        ssvm.fit(whas500.x, whas500.y)

        assert not hasattr(ssvm, "intercept_")
        expected_coef = numpy.array([0.00669121, -0.2754864, -0.14124808, 0.0748376, -0.2812598, 0.07543884,
                                     0.09845683, 0.08398258, -0.12182314, 0.02637739, 0.03060149, 0.11870598,
                                     -0.52688224, -0.01762842])
        assert_array_almost_equal(expected_coef, ssvm.coef_)

        pred = ssvm.predict(whas500.x)
        rmse = numpy.sqrt(mean_squared_error(whas500.y['lenfol'], pred))
        assert round(abs(1128.4460587629746 - rmse), 7) == 0

    @staticmethod
    @pytest.mark.slow
    def test_fit_and_predict_regression(make_whas500, optimizer_regression):
        whas500 = make_whas500(to_numeric=True)

        ssvm = FastSurvivalSVM(optimizer=optimizer_regression, rank_ratio=0.0,
                               max_iter=50, fit_intercept=True, random_state=0)
        ssvm.fit(whas500.x, whas500.y)

        assert round(abs(6.4160179606675278 - ssvm.intercept_), 7) == 0
        expected_coef = numpy.array(
            [-0.0730891368237, -0.536630355029, -0.497411603275, 0.269039958377, -0.730559850692, -0.0148443526234,
             0.285916578892, 0.165960302339, -0.301749910087, 0.334855938531, 0.0886214732161, 0.0554890272028,
             -2.12680470014, 0.0421466831393
             ])
        assert_array_almost_equal(expected_coef, ssvm.coef_)

        pred = ssvm.predict(whas500.x)
        rmse = numpy.sqrt(mean_squared_error(whas500.y['lenfol'], pred))
        assert round(abs(1206.6556186869332 - rmse), 7) == 0

    @staticmethod
    @pytest.mark.slow
    def test_fit_and_predict_regression_no_intercept(make_whas500, optimizer_regression):
        whas500 = make_whas500(to_numeric=True)

        ssvm = FastSurvivalSVM(optimizer=optimizer_regression, rank_ratio=0.0,
                               max_iter=50, fit_intercept=False, random_state=0)
        ssvm.fit(whas500.x, whas500.y)

        assert not hasattr(ssvm, "intercept_")
        expected_coef = numpy.array([1.39989875, -1.16903161, -0.40195857, -0.05848903, -0.08421557, 4.11924729,
                                     0.25135451, 1.89067276, -0.25751401, -0.10213143, 1.56333622, 3.10136873,
                                     -2.23644848, -0.11620715])
        assert_array_almost_equal(expected_coef, ssvm.coef_)

        pred = ssvm.predict(whas500.x)
        rmse = numpy.sqrt(mean_squared_error(whas500.y['lenfol'], pred))
        assert round(abs(15838.510668936022 - rmse), 7) == 0

    @staticmethod
    @pytest.mark.slow
    def test_fit_timeit(make_whas500, optimizer_any):
        whas500 = make_whas500(to_numeric=True)
        rnd = numpy.random.RandomState(0)
        idx = rnd.choice(numpy.arange(whas500.x.shape[0]), replace=False, size=100)

        ssvm = FastSurvivalSVM(optimizer=optimizer_any, timeit=3, random_state=0)
        ssvm.fit(whas500.x[idx, :], whas500.y[idx])

        assert 'timings' in ssvm.optimizer_result_


class TestKernelSurvivalSVM(object):

    @staticmethod
    def test_default_optimizer(make_whas500):
        whas500 = make_whas500(to_numeric=True)
        ssvm = FastKernelSurvivalSVM(tol=1e-4, max_iter=25)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            ssvm.fit(whas500.x, whas500.y)
        assert 'rbtree' == ssvm.optimizer

    @staticmethod
    def test_unknown_optimizer(fake_data):
        x, y = fake_data

        ssvm = FastKernelSurvivalSVM(optimizer='random stuff')
        with pytest.raises(ValueError,
                           match="unknown optimizer: random stuff"):
            ssvm.fit(x, y)

    @staticmethod
    def test_fit_and_predict_linear(make_whas500):
        whas500 = make_whas500(to_numeric=True)
        ssvm = FastKernelSurvivalSVM(optimizer="rbtree", kernel='linear', random_state=0)
        ssvm.fit(whas500.x, whas500.y)

        assert not ssvm._pairwise
        assert whas500.x.shape[0] == ssvm.coef_.shape[0]

        i = numpy.arange(250)
        numpy.random.RandomState(0).shuffle(i)
        c = ssvm.score(whas500.x[i], whas500.y[i])
        assert round(abs(c - 0.76923445664157997), 6) == 0

    @staticmethod
    def test_fit_and_predict_linear_precomputed(make_whas500):
        whas500 = make_whas500(to_numeric=True)
        ssvm = FastKernelSurvivalSVM(optimizer="rbtree", kernel='precomputed', random_state=0)
        x = numpy.dot(whas500.x, whas500.x.T)
        ssvm.fit(x, whas500.y)

        assert ssvm._pairwise
        assert whas500.x.shape[0] == ssvm.coef_.shape[0]

        i = numpy.arange(250)
        numpy.random.RandomState(0).shuffle(i)
        c = ssvm.score(x[i], whas500.y[i])
        assert round(abs(c - 0.76923445664157997), 6) == 0

    @staticmethod
    def test_fit_and_predict_linear_regression(make_whas500):
        whas500 = make_whas500(to_numeric=True)
        ssvm = FastKernelSurvivalSVM(optimizer="rbtree", rank_ratio=0.0, kernel="linear",
                                     max_iter=50, fit_intercept=True, random_state=0)

        ssvm.fit(whas500.x, whas500.y)

        assert not ssvm._pairwise
        assert round(abs(ssvm.intercept_ - 6.3979746625712295), 5) == 0

        i = numpy.arange(250)
        numpy.random.RandomState(0).shuffle(i)
        pred = ssvm.predict(whas500.x[i])
        rmse = numpy.sqrt(mean_squared_error(whas500.y['lenfol'][i], pred))
        assert rmse <= 1339.3006854574726 + 0.275

    @staticmethod
    def test_fit_and_predict_linear_regression_precomputed(make_whas500):
        whas500 = make_whas500(to_numeric=True)
        ssvm = FastKernelSurvivalSVM(optimizer="rbtree", rank_ratio=0.0, kernel="precomputed",
                                     max_iter=50, fit_intercept=True, random_state=0)
        x = numpy.dot(whas500.x, whas500.x.T)
        ssvm.fit(x, whas500.y)

        assert ssvm._pairwise
        assert round(abs(ssvm.intercept_ - 6.3979746625712295), 5) == 0

        i = numpy.arange(250)
        numpy.random.RandomState(0).shuffle(i)
        pred = ssvm.predict(x[i])
        rmse = numpy.sqrt(mean_squared_error(whas500.y['lenfol'][i], pred))
        assert rmse <= 1339.3006854574726 + 0.275

    @staticmethod
    def test_fit_and_predict_linear_regression_no_intercept(make_whas500):
        whas500 = make_whas500(to_numeric=True)
        ssvm = FastKernelSurvivalSVM(optimizer="rbtree", rank_ratio=0.0, kernel="linear",
                                     max_iter=50, fit_intercept=False, random_state=0)
        ssvm.fit(whas500.x, whas500.y)

        assert not hasattr(ssvm, "intercept_")

        pred = ssvm.predict(whas500.x)
        rmse = numpy.sqrt(mean_squared_error(whas500.y['lenfol'], pred))
        assert round(abs(rmse - 15837.658418546907), 4) == 0

    @staticmethod
    @pytest.mark.slow
    @pytest.mark.parametrize('optimizer', ['rbtree', 'avltree'])
    def test_fit_and_predict_rbf(make_whas500, optimizer):
        whas500 = make_whas500(to_numeric=True)
        ssvm = FastKernelSurvivalSVM(optimizer=optimizer, kernel='rbf',
                                     tol=2e-6, max_iter=75, random_state=0)
        ssvm.fit(whas500.x, whas500.y)

        assert not ssvm._pairwise
        assert whas500.x.shape[0] == ssvm.coef_.shape[0]

        c = ssvm.score(whas500.x, whas500.y)
        assert c >= 0.965

    @staticmethod
    @pytest.mark.slow
    def test_fit_and_predict_regression_rbf(make_whas500):
        whas500 = make_whas500(to_numeric=True)
        ssvm = FastKernelSurvivalSVM(optimizer="rbtree", rank_ratio=0.0, kernel="rbf",
                                     tol=1e-6, max_iter=50, fit_intercept=True, random_state=0)
        ssvm.fit(whas500.x, whas500.y)

        assert not ssvm._pairwise
        assert round(abs(ssvm.intercept_ - 4.9267218894089533), 7) == 0

        pred = ssvm.predict(whas500.x)
        rmse = numpy.sqrt(mean_squared_error(whas500.y['lenfol'], pred))
        assert round(abs(rmse - 783.525277), 6) == 0

    @staticmethod
    @pytest.mark.slow
    def test_fit_and_predict_hybrid_rbf(make_whas500):
        whas500 = make_whas500(to_numeric=True)
        ssvm = FastKernelSurvivalSVM(optimizer="rbtree", rank_ratio=0.5, kernel="rbf",
                                     max_iter=50, fit_intercept=True, random_state=0)
        ssvm.fit(whas500.x, whas500.y)

        assert not ssvm._pairwise
        assert abs(5.0289145697617164 - ssvm.intercept_) <= 0.04

        pred = ssvm.predict(whas500.x)
        rmse = numpy.sqrt(mean_squared_error(whas500.y['lenfol'], pred))
        assert abs(880.20361811281487 - rmse) <= 75

    @staticmethod
    @pytest.mark.slow
    def test_fit_and_predict_clinical_kernel(make_whas500):
        whas500 = make_whas500(to_numeric=True)

        trans = ClinicalKernelTransform()
        trans.fit(whas500.x_data_frame)

        ssvm = FastKernelSurvivalSVM(optimizer="rbtree", kernel=trans.pairwise_kernel,
                                     tol=7e-7, max_iter=100, random_state=0)
        ssvm.fit(whas500.x, whas500.y)

        assert not ssvm._pairwise
        assert whas500.x.shape[0] == ssvm.coef_.shape[0]

        c = ssvm.score(whas500.x, whas500.y)
        assert c >= 0.854

    @staticmethod
    @pytest.mark.slow
    def test_compare_builtin_kernel(make_whas500):
        whas500 = make_whas500(to_numeric=True)
        x = normalize(whas500.x)

        rsvm = FastKernelSurvivalSVM(optimizer='rbtree', kernel="polynomial",
                                     gamma=0.5, degree=2, coef0=0,
                                     tol=2.5e-8, max_iter=100, random_state=0xf38)
        rsvm.fit(x, whas500.y)
        pred_rsvm = rsvm.predict(x)

        kpca = KernelPCA(kernel="polynomial", copy_X=True, gamma=0.5, degree=2, coef0=0,
                         random_state=0xf38)
        xt = kpca.fit_transform(x)
        nrsvm = FastSurvivalSVM(optimizer='rbtree', tol=2.5e-8, max_iter=100, random_state=0xf38)
        nrsvm.fit(xt, whas500.y)
        pred_nrsvm = nrsvm.predict(xt)

        assert len(pred_nrsvm) == len(pred_rsvm)

        expected_cindex = concordance_index_censored(whas500.y['fstat'], whas500.y['lenfol'], pred_nrsvm)
        assert_cindex_almost_equal(whas500.y['fstat'], whas500.y['lenfol'], pred_rsvm,
                                   expected_cindex)

    @staticmethod
    @pytest.mark.slow
    def test_compare_clinical_kernel(make_whas500):
        whas500 = make_whas500(to_numeric=True)

        trans = ClinicalKernelTransform()
        trans.fit(whas500.x_data_frame)

        kpca = KernelPCA(kernel=trans.pairwise_kernel, copy_X=True)
        xt = kpca.fit_transform(whas500.x)

        nrsvm = FastSurvivalSVM(optimizer='rbtree', tol=1e-8, max_iter=500, random_state=0)
        nrsvm.fit(xt, whas500.y)

        rsvm = FastKernelSurvivalSVM(optimizer='rbtree', kernel=trans.pairwise_kernel,
                                     tol=1e-8, max_iter=500, random_state=0)
        rsvm.fit(whas500.x, whas500.y)

        pred_nrsvm = nrsvm.predict(kpca.transform(whas500.x))
        pred_rsvm = rsvm.predict(whas500.x)

        assert len(pred_nrsvm) == len(pred_rsvm)

        expected_cindex = concordance_index_censored(whas500.y['fstat'], whas500.y['lenfol'], pred_nrsvm)
        assert_cindex_almost_equal(whas500.y['fstat'], whas500.y['lenfol'], pred_rsvm,
                                   expected_cindex)

    @staticmethod
    def test_fit_precomputed_kernel_invalid_shape(fake_data):
        x, y = fake_data
        ssvm = FastKernelSurvivalSVM(optimizer="rbtree", kernel='precomputed', random_state=0)

        with pytest.raises(ValueError,
                           match=r"Precomputed metric requires shape \(n_queries, n_indexed\)\. "
                                 r"Got \(100, 11\) for 100 indexed\."):
            ssvm.fit(x, y)

    @staticmethod
    def test_fit_precomputed_kernel_not_symmetric():
        ssvm = FastKernelSurvivalSVM(optimizer="rbtree", kernel='precomputed', random_state=0)
        x = numpy.random.randn(100, 100)
        x[10, 12] = -1
        x[12, 10] = 9
        y = Surv.from_arrays(numpy.ones(100).astype(bool), numpy.ones(100))

        with pytest.raises(ValueError,
                           match="kernel matrix is not symmetric"):
            ssvm.fit(x, y)

    @staticmethod
    def test_predict_precomputed_kernel_invalid_shape(make_whas500):
        whas500 = make_whas500(to_numeric=True)
        ssvm = FastKernelSurvivalSVM(optimizer="rbtree", kernel='precomputed', random_state=0)
        x = numpy.dot(whas500.x, whas500.x.T)
        ssvm.fit(x, whas500.y)

        x_new = numpy.random.randn(100, 14)
        with pytest.raises(ValueError,
                           match=r"Precomputed metric requires shape \(n_queries, n_indexed\)\. "
                                 r"Got \(100, 14\) for 500 indexed\."):
            ssvm.predict(x_new)

    @staticmethod
    @pytest.mark.parametrize("optimizer", ("avltree", "rbtree"))
    def test_fit_uncomparable(whas500_uncomparable, optimizer):
        ssvm = FastKernelSurvivalSVM(optimizer=optimizer)
        with pytest.raises(NoComparablePairException):
            ssvm.fit(whas500_uncomparable.x, whas500_uncomparable.y)


@pytest.fixture(params=[
    SurvivalCounter,
    partial(OrderStatisticTreeSurvivalCounter, tree_class=RBTree),
    partial(OrderStatisticTreeSurvivalCounter, tree_class=AVLTree)
])
def make_survival_counter(request):
    def _make_survival_counter(*args, **kwargs):
        cls = request.param
        if isinstance(cls, partial):
            kwargs.pop('n_relevance_levels')

        counter = cls(*args, **kwargs)
        return counter
    return _make_survival_counter


class TestSurvivalCounter(object):

    def setup_01(self):
        w = numpy.array([-0.9, -0.7, -0.1, 0.15, 0.2, 1.6])
        y = numpy.array([2,       0,    4,    3,   5,   1])
        event = numpy.array([True, True, False, True, False, True])
        x = numpy.eye(6)
        v = numpy.arange(6)
        return x, y, event, w, v

    def test_calculate_01(self, make_survival_counter):
        x, y, event, w, v = self.setup_01()
        counter = make_survival_counter(x, y, event, n_relevance_levels=6)
        counter.update_sort_order(w)

        l_plus, xv_plus, l_minus, xv_minus = counter.calculate(v)

        assert_array_equal(numpy.array([1, 4, 0, 2, 0, 4]), l_plus)
        assert_array_equal(numpy.array([2, 9, 0, 6, 0, 9]), xv_plus)
        assert_array_equal(numpy.array([2, 0, 4, 2, 3, 0]), l_minus)
        assert_array_equal(numpy.array([6, 0, 9, 6, 9, 0]), xv_minus)

    def test_calculate_01_reverse(self, make_survival_counter):
        x, y, event, w, v = self.setup_01()
        counter = make_survival_counter(x, y[::-1], event[::-1], n_relevance_levels=6)
        counter.update_sort_order(w[::-1])

        l_plus, xv_plus, l_minus, xv_minus = counter.calculate(v[::-1])

        assert_array_equal(numpy.array([4, 0, 2, 0, 4, 1]), l_plus)
        assert_array_equal(numpy.array([9, 0, 6, 0, 9, 2]), xv_plus)
        assert_array_equal(numpy.array([0, 3, 2, 4, 0, 2]), l_minus)
        assert_array_equal(numpy.array([0, 9, 6, 9, 0, 6]), xv_minus)

    def setup_02(self):
        w = numpy.array([-0.9, -0.7, -0.1, 0.15, 0.2, 0.3, 0.8, 1.6, 1.85, 2.3])
        y = numpy.array([3,       0,    4,    6,   8,   5,   1,   7,    2,   9])
        event = numpy.array([0,   0,    0,    1,   0,   1,   1,   0,    1,   0], dtype=bool)
        x = numpy.eye(10)
        v = numpy.arange(10)
        return x, y, event, w, v

    def test_calculate_02(self, make_survival_counter):
        x, y, event, w, v = self.setup_02()
        counter = make_survival_counter(x, y, event, n_relevance_levels=10)
        counter.update_sort_order(w)

        l_plus, xv_plus, l_minus, xv_minus = counter.calculate(v)

        assert_array_equal(numpy.array([0, 0, 0, 1, 0, 2, 6, 0, 7, 0]), l_plus)
        assert_array_equal(numpy.array([0, 0, 0, 4, 0, 7, 21, 0, 30, 0]), xv_plus)
        assert_array_equal(numpy.array([2, 0, 2, 3, 4, 2, 0, 2, 0, 1]), l_minus)
        assert_array_equal(numpy.array([14, 0, 14, 19, 22, 14, 0, 14, 0, 8]), xv_minus)

    def test_calculate_02_reverse(self, make_survival_counter):
        x, y, event, w, v = self.setup_02()
        counter = make_survival_counter(x, y[::-1], event[::-1], n_relevance_levels=10)
        counter.update_sort_order(w[::-1])

        l_plus, xv_plus, l_minus, xv_minus = counter.calculate(v[::-1])

        assert_array_equal(numpy.array([0, 7, 0, 6, 2, 0, 1, 0, 0, 0]), l_plus)
        assert_array_equal(numpy.array([0, 30, 0, 21, 7, 0, 4, 0, 0, 0]), xv_plus)
        assert_array_equal(numpy.array([1, 0, 2, 0, 2, 4, 3, 2, 0, 2]), l_minus)
        assert_array_equal(numpy.array([8, 0, 14, 0, 14, 22, 19, 14, 0, 14]), xv_minus)


@pytest.fixture
def whas500_without_ties():
    # naive survival SVM does resolve ties in survival time differently,
    # therefore use data without ties
    data = loadarff(WHAS500_NOTIES_FILE)
    x, y = get_x_y(data, ['fstat', 'lenfol'], '1')
    x = encode_categorical(x)
    return x, y


@pytest.fixture
def whas500_with_ties():
    # naive survival SVM does resolve ties in survival time differently,
    # therefore use data without ties
    x, y = load_whas500()
    x = normalize(encode_categorical(x))
    return x, y


class TestNaiveSurvivalSVM(object):

    @staticmethod
    def test_survival_squared_hinge_loss(whas500_without_ties):
        x, y = whas500_without_ties

        nrsvm = NaiveSurvivalSVM(loss='squared_hinge', dual=False, tol=8e-7, max_iter=1000, random_state=0)
        nrsvm.fit(x, y)

        rsvm = FastSurvivalSVM(optimizer='avltree', tol=8e-7, max_iter=1000, random_state=0)
        rsvm.fit(x, y)

        assert_array_almost_equal(nrsvm.coef_.ravel(), rsvm.coef_, 3)

        pred_nrsvm = nrsvm.predict(x)
        pred_rsvm = rsvm.predict(x)

        assert len(pred_nrsvm) == len(pred_rsvm)

        expected_cindex = concordance_index_censored(y['fstat'], y['lenfol'], pred_nrsvm)
        assert_cindex_almost_equal(y['fstat'], y['lenfol'], pred_rsvm,
                                   expected_cindex)

    @staticmethod
    def test_fit_with_ties(whas500_with_ties):
        x, y = whas500_with_ties

        nrsvm = NaiveSurvivalSVM(loss='squared_hinge', dual=False, tol=1e-8, max_iter=1000, random_state=0)
        nrsvm.fit(x, y)

        assert nrsvm.coef_.shape == (1, 14)

        cindex = nrsvm.score(x, y)
        assert round(abs(cindex - 0.7760582309811175), 7) == 0

    @staticmethod
    def test_fit_uncomparable(whas500_uncomparable):
        ssvm = NaiveSurvivalSVM(loss='squared_hinge', dual=False, tol=1e-8, max_iter=1000, random_state=0)
        with pytest.raises(NoComparablePairException):
            ssvm.fit(whas500_uncomparable.x, whas500_uncomparable.y)
