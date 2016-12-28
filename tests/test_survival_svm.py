import unittest
from functools import partial
from os.path import join, dirname

import numpy
from numpy.testing import TestCase, run_module_suite, assert_array_almost_equal, assert_array_equal
from sklearn.decomposition import KernelPCA
from sklearn.metrics import mean_squared_error

from sksurv.bintrees import AVLTree, RBTree
from sksurv.column import encode_categorical, standardize
from sksurv.datasets import load_whas500, get_x_y
from sksurv.io import loadarff
from sksurv.kernels import ClinicalKernelTransform
from sksurv.metrics import concordance_index_censored
from sksurv.svm._prsvm import survival_constraints_simple
from sksurv.svm.naive_survival_svm import NaiveSurvivalSVM
from sksurv.svm.survival_svm import FastSurvivalSVM, FastKernelSurvivalSVM, SurvivalCounter, \
    OrderStatisticTreeSurvivalCounter

WHAS500_NOTIES_FILE = join(dirname(__file__), 'data', 'whas500-noties.arff')


class TestSurvivalSVM(TestCase):
    def test_alpha_negative(self):
        x = numpy.zeros((100, 10))
        y = numpy.empty(dtype=[('event', bool), ('time', float)], shape=100)
        y['event'] = numpy.ones(100, dtype=bool)
        y['time'] = numpy.arange(100, dtype=float)

        ssvm = FastSurvivalSVM(alpha=-1)
        self.assertRaisesRegex(ValueError, "alpha must be positive",
                               ssvm.fit, x, y)

    def test_rank_ratio_out_of_bounds(self):
        x = numpy.zeros((100, 10))
        y = numpy.empty(dtype=[('event', bool), ('time', float)], shape=100)
        y['event'] = numpy.ones(100, dtype=bool)
        y['time'] = numpy.arange(100, dtype=float)

        ssvm = FastSurvivalSVM(rank_ratio=-1)
        self.assertRaisesRegex(ValueError, "rank_ratio must be in \[0; 1\]",
                               ssvm.fit, x, y)

        ssvm.set_params(rank_ratio=1.2)
        self.assertRaisesRegex(ValueError, "rank_ratio must be in \[0; 1\]",
                               ssvm.fit, x, y)

        ssvm.set_params(rank_ratio=numpy.nan)
        self.assertRaisesRegex(ValueError, "rank_ratio must be in \[0; 1\]",
                               ssvm.fit, x, y)

        ssvm.set_params(rank_ratio=numpy.inf)
        self.assertRaisesRegex(ValueError, "rank_ratio must be in \[0; 1\]",
                               ssvm.fit, x, y)

    def test_regression_not_supported(self):
        x = numpy.zeros((100, 10))
        y = numpy.empty(dtype=[('event', bool), ('time', float)], shape=100)
        y['event'] = numpy.ones(100, dtype=bool)
        y['time'] = numpy.arange(100, dtype=float)

        ssvm = FastSurvivalSVM(rank_ratio=0, optimizer='simple')
        self.assertRaisesRegex(ValueError,
                               "optimizer 'simple' does not implement regression objective",
                               ssvm.fit, x, y)

        ssvm.set_params(optimizer='PRSVM')
        self.assertRaisesRegex(ValueError,
                               "optimizer 'PRSVM' does not implement regression objective",
                               ssvm.fit, x, y)

    def test_unknown_optimizer(self):
        x = numpy.zeros((100, 10))
        y = numpy.empty(dtype=[('event', bool), ('time', float)], shape=100)
        y['event'] = numpy.ones(100, dtype=bool)
        y['time'] = numpy.arange(1, 101, dtype=float)

        ssvm = FastSurvivalSVM(rank_ratio=0, optimizer='random stuff')
        self.assertRaisesRegex(ValueError,
                               "unknown optimizer: random stuff",
                               ssvm.fit, x, y)

    def test_y_no_array(self):
        x = numpy.zeros((100, 10))
        y = [numpy.ones(100, dtype=bool), numpy.arange(100)]

        rsvm = FastSurvivalSVM()
        self.assertRaisesRegex(ValueError,
                               'y must be a structured array with the first field'
                               ' being a binary class event indicator and the second field'
                               ' the time of the event/censoring',
                               rsvm.fit, x, y)

    def test_only_one_label(self):
        x = numpy.zeros((100, 10))
        y = numpy.ones(100, dtype=int)

        rsvm = FastSurvivalSVM()
        self.assertRaisesRegex(ValueError,
                               'y must be a structured array with the first field'
                               ' being a binary class event indicator and the second field'
                               ' the time of the event/censoring',
                               rsvm.fit, x, y)

    def test_y_one_field(self):
        x = numpy.arange(80).reshape(10, 8)
        y = numpy.ones(dtype=[('event', bool)], shape=10)

        rsvm = FastSurvivalSVM()
        self.assertRaisesRegex(ValueError,
                               'y must be a structured array with the first field'
                               ' being a binary class event indicator and the second field'
                               ' the time of the event/censoring',
                               rsvm.fit, x, y)

    def test_y_three_fields(self):
        x = numpy.arange(80).reshape(10, 8)
        y = numpy.ones(dtype=[('event', bool), ('time', float), ('too_much', int)], shape=10)

        rsvm = FastSurvivalSVM()
        self.assertRaisesRegex(ValueError,
                               'y must be a structured array with the first field'
                               ' being a binary class event indicator and the second field'
                               ' the time of the event/censoring',
                               rsvm.fit, x, y)

    def test_event_not_boolean(self):
        x = numpy.arange(80).reshape(10, 8)
        y = numpy.empty(dtype=[('event', int), ('time', float)], shape=10)
        y['event'] = numpy.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1], dtype=int)
        y['time'] = numpy.arange(10)

        rsvm = FastSurvivalSVM()
        self.assertRaisesRegex(ValueError,
                               "elements of event indicator must be boolean, but found int",
                               rsvm.fit, x, y)

    def test_event_not_binary(self):
        x = numpy.arange(80).reshape(10, 8)
        y = numpy.empty(dtype=[('event', int), ('time', float)], shape=10)
        y['event'] = numpy.array([0, 1, 2, 1, 1, 0, 1, 2, 3, 1], dtype=int)
        y['time'] = numpy.arange(10)

        rsvm = FastSurvivalSVM()
        self.assertRaisesRegex(ValueError,
                               "elements of event indicator must be boolean, but found int",
                               rsvm.fit, x, y)

    def test_time_not_numeric(self):
        x = numpy.arange(80).reshape(10, 8)
        y = numpy.empty(dtype=[('event', bool), ('time', bool)], shape=10)
        y['event'] = numpy.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1], dtype=bool)
        y['time'] = numpy.ones(10, dtype=bool)

        rsvm = FastSurvivalSVM()
        self.assertRaisesRegex(ValueError,
                               "time must be numeric, but found bool",
                               rsvm.fit, x, y)

    def test_all_censored(self):
        x = numpy.arange(80).reshape(10, 8)
        y = numpy.empty(dtype=[('event', bool), ('time', float)], shape=10)
        y['event'] = numpy.zeros(10, dtype=bool)
        y['time'] = numpy.array([0, 1, 2, 1, 1, 0, 1, 2, 3, 1])

        rsvm = FastSurvivalSVM()
        self.assertRaisesRegex(ValueError,
                               "all samples are censored",
                               rsvm.fit, x, y)

    def test_zero_time(self):
        x = numpy.arange(80).reshape(10, 8)
        y = numpy.empty(dtype=[('event', bool), ('time', float)], shape=10)
        y['event'] = numpy.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1], dtype=bool)
        y['time'] = numpy.array([0, 1, 2, 1, 1, 0, 1, 2, 3, 1])

        rsvm = FastSurvivalSVM(rank_ratio=0.5)
        self.assertRaisesRegex(ValueError,
                               "observed time contains values smaller or equal to zero",
                               rsvm.fit, x, y)

    def test_negative_time(self):
        x = numpy.arange(80).reshape(10, 8)
        y = numpy.empty(dtype=[('event', bool), ('time', float)], shape=10)
        y['event'] = numpy.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1], dtype=bool)
        y['time'] = numpy.array([1, 1, -2, 1, 1, 6, 1, 2, 3, 1])

        rsvm = FastSurvivalSVM(rank_ratio=0.5)
        self.assertRaisesRegex(ValueError,
                               "observed time contains values smaller or equal to zero",
                               rsvm.fit, x, y)

    def test_ranking_with_fit_intercept(self):
        x = numpy.zeros((100, 10))
        y = numpy.empty(dtype=[('event', bool), ('time', float)], shape=100)
        y['event'] = numpy.ones(100, dtype=bool)
        y['time'] = numpy.arange(1, 101, dtype=float)

        ssvm = FastSurvivalSVM(rank_ratio=1.0, fit_intercept=True)
        self.assertRaisesRegex(ValueError,
                               "fit_intercept=True is only meaningful if rank_ratio < 1.0",
                               ssvm.fit, x, y)

    def test_survial_constraints_no_ties(self):
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

    def test_survival_constraints_with_ties(self):
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


class SurvivalModeCases(object):
    OPTIMIZER = None

    def setUp(self):
        x, self.y = load_whas500()
        self.x = encode_categorical(standardize(x))

    def test_default_optimizer(self):
        self.assertEqual('avltree', FastSurvivalSVM().fit(self.x.values, self.y).optimizer)

    def test_fit_and_predict_ranking(self):
        ssvm = FastSurvivalSVM(optimizer=self.OPTIMIZER, random_state=0)
        ssvm.fit(self.x.values, self.y)

        self.assertFalse(hasattr(ssvm, "intercept_"))
        expected_coef = numpy.array([-0.02066177, -0.26449933, -0.15205399, 0.0794547, -0.28840498, -0.02864288,
                                     0.09901995, 0.04505302, -0.12512215, 0.03341365, -0.00110442, 0.05446756,
                                     -0.53009875, -0.01394175])
        assert_array_almost_equal(expected_coef, ssvm.coef_)

        self.assertEquals(self.x.shape[1], ssvm.coef_.shape[0])

        c = ssvm.score(self.x.values, self.y)

        self.assertAlmostEqual(0.7860650174985695, c, 6)

    def test_fit_and_predict_hybrid(self):
        if self.OPTIMIZER in {'simple', 'PRSVM'}:
            raise unittest.SkipTest("regression not implemented for " + self.OPTIMIZER)

        ssvm = FastSurvivalSVM(optimizer=self.OPTIMIZER, rank_ratio=0.5,
                               max_iter=50, fit_intercept=True, random_state=0)
        ssvm.fit(self.x.values, self.y)

        self.assertAlmostEqual(6.1409367385513729, ssvm.intercept_)
        expected_coef = numpy.array(
            [-0.0209254120718, -0.265768317208, -0.154254689136, 0.0800600947891, -0.290121131022, -0.0288851785213,
             0.0998004550073, 0.0454100937492, -0.125863947621, 0.0343588337797, -0.000710219364914, 0.0546969104996,
             -0.5375338235, -0.0137995110308
             ])
        assert_array_almost_equal(expected_coef, ssvm.coef_)

        pred = ssvm.predict(self.x.values)
        rmse = numpy.sqrt(mean_squared_error(self.y['lenfol'], pred))
        self.assertAlmostEqual(780.52617631863893, rmse)

    def test_fit_and_predict_hybrid_no_intercept(self):
        if self.OPTIMIZER in {'simple', 'PRSVM'}:
            raise unittest.SkipTest("regression not implemented for " + self.OPTIMIZER)

        ssvm = FastSurvivalSVM(optimizer=self.OPTIMIZER, rank_ratio=0.5,
                               max_iter=50, fit_intercept=False, random_state=0)
        ssvm.fit(self.x.values, self.y)

        self.assertFalse(hasattr(ssvm, "intercept_"))
        expected_coef = numpy.array([0.00669121, -0.2754864, -0.14124808, 0.0748376, -0.2812598, 0.07543884,
                                     0.09845683, 0.08398258, -0.12182314, 0.02637739, 0.03060149, 0.11870598,
                                     -0.52688224, -0.01762842])
        assert_array_almost_equal(expected_coef, ssvm.coef_)

        pred = ssvm.predict(self.x.values)
        rmse = numpy.sqrt(mean_squared_error(self.y['lenfol'], pred))
        self.assertAlmostEqual(1128.4460587629746, rmse)

    def test_fit_and_predict_regression(self):
        if self.OPTIMIZER in {'simple', 'PRSVM'}:
            raise unittest.SkipTest("regression not implemented for " + self.OPTIMIZER)

        ssvm = FastSurvivalSVM(optimizer=self.OPTIMIZER, rank_ratio=0.0,
                               max_iter=50, fit_intercept=True, random_state=0)
        ssvm.fit(self.x.values, self.y)

        self.assertAlmostEqual(6.4160179606675278, ssvm.intercept_)
        expected_coef = numpy.array(
            [-0.0730891368237, -0.536630355029, -0.497411603275, 0.269039958377, -0.730559850692, -0.0148443526234,
             0.285916578892, 0.165960302339, -0.301749910087, 0.334855938531, 0.0886214732161, 0.0554890272028,
             -2.12680470014, 0.0421466831393
             ])
        assert_array_almost_equal(expected_coef, ssvm.coef_)

        pred = ssvm.predict(self.x.values)
        rmse = numpy.sqrt(mean_squared_error(self.y['lenfol'], pred))
        self.assertAlmostEqual(1206.6556186869332, rmse)

    def test_fit_and_predict_regression_no_intercept(self):
        if self.OPTIMIZER in {'simple', 'PRSVM'}:
            raise unittest.SkipTest("regression not implemented for " + self.OPTIMIZER)

        ssvm = FastSurvivalSVM(optimizer=self.OPTIMIZER, rank_ratio=0.0,
                               max_iter=50, fit_intercept=False, random_state=0)
        ssvm.fit(self.x.values, self.y)

        self.assertFalse(hasattr(ssvm, "intercept_"))
        expected_coef = numpy.array([1.39989875, -1.16903161, -0.40195857, -0.05848903, -0.08421557, 4.11924729,
                                     0.25135451, 1.89067276, -0.25751401, -0.10213143, 1.56333622, 3.10136873,
                                     -2.23644848, -0.11620715])
        assert_array_almost_equal(expected_coef, ssvm.coef_)

        pred = ssvm.predict(self.x.values)
        rmse = numpy.sqrt(mean_squared_error(self.y['lenfol'], pred))
        self.assertAlmostEqual(15838.510668936022, rmse)

    def test_fit_timeit(self):
        rnd = numpy.random.RandomState(0)
        idx = rnd.choice(numpy.arange(self.x.shape[0]), replace=False, size=100)

        ssvm = FastSurvivalSVM(optimizer=self.OPTIMIZER, timeit=3, random_state=0)
        ssvm.fit(self.x.values[idx, :], self.y[idx])

        self.assertTrue('timings' in ssvm.optimizer_result_)


class TestSimpleSurvivalMode(SurvivalModeCases, TestCase):
    OPTIMIZER = 'simple'


class TestPRSVMSurvivalMode(SurvivalModeCases, TestCase):
    OPTIMIZER = 'PRSVM'


class TestDirectCountSurvivalMode(SurvivalModeCases, TestCase):
    OPTIMIZER = 'direct-count'


class TestRBTreeSurvivalMode(SurvivalModeCases, TestCase):
    OPTIMIZER = 'rbtree'


class TestAVLTreeSurvivalMode(SurvivalModeCases, TestCase):
    OPTIMIZER = 'avltree'


class TestKernelSurvivalSVM(TestCase):

    def setUp(self):
        x, self.y = load_whas500()
        self.x = encode_categorical(standardize(x))

    def test_default_optimizer(self):
        self.assertEqual('rbtree', FastKernelSurvivalSVM().fit(self.x.values, self.y).optimizer)

    def test_unknown_optimizer(self):
        x = numpy.zeros((100, 10))
        y = numpy.empty(dtype=[('event', bool), ('time', float)], shape=100)
        y['event'] = numpy.ones(100, dtype=bool)
        y['time'] = numpy.arange(1, 101, dtype=float)

        ssvm = FastKernelSurvivalSVM(optimizer='random stuff')
        self.assertRaisesRegex(ValueError,
                               "unknown optimizer: random stuff",
                               ssvm.fit, x, y)

    def test_fit_and_predict_linear(self):
        ssvm = FastKernelSurvivalSVM(optimizer="rbtree", kernel='linear', random_state=0)
        ssvm.fit(self.x.values, self.y)

        self.assertFalse(ssvm._pairwise)
        self.assertEquals(self.x.shape[0], ssvm.coef_.shape[0])

        i = numpy.arange(250)
        numpy.random.RandomState(0).shuffle(i)
        c = ssvm.score(self.x.values[i], self.y[i])
        self.assertAlmostEqual(0.76923445664157997, c, 6)

    def test_fit_and_predict_linear_precomputed(self):
        ssvm = FastKernelSurvivalSVM(optimizer="rbtree", kernel='precomputed', random_state=0)
        x = numpy.dot(self.x.values, self.x.values.T)
        ssvm.fit(x, self.y)

        self.assertTrue(ssvm._pairwise)
        self.assertEquals(self.x.shape[0], ssvm.coef_.shape[0])

        i = numpy.arange(250)
        numpy.random.RandomState(0).shuffle(i)
        c = ssvm.score(x[i], self.y[i])
        self.assertAlmostEqual(0.76923445664157997, c, 6)

    def test_fit_and_predict_linear_regression(self):
        ssvm = FastKernelSurvivalSVM(optimizer="rbtree", rank_ratio=0.0, kernel="linear",
                                     max_iter=50, fit_intercept=True, random_state=0)

        ssvm.fit(self.x.values, self.y)

        self.assertFalse(ssvm._pairwise)
        self.assertAlmostEqual(6.3979746625712295, ssvm.intercept_, 5)

        i = numpy.arange(250)
        numpy.random.RandomState(0).shuffle(i)
        pred = ssvm.predict(self.x.values[i])
        rmse = numpy.sqrt(mean_squared_error(self.y['lenfol'][i], pred))
        self.assertLessEqual(abs(1339.3006854574726 - rmse), 0.25)

    def test_fit_and_predict_linear_regression_precomputed(self):
        ssvm = FastKernelSurvivalSVM(optimizer="rbtree", rank_ratio=0.0, kernel="precomputed",
                                     max_iter=50, fit_intercept=True, random_state=0)
        x = numpy.dot(self.x.values, self.x.values.T)
        ssvm.fit(x, self.y)

        self.assertTrue(ssvm._pairwise)
        self.assertAlmostEqual(6.3979746625712295, ssvm.intercept_, 5)

        i = numpy.arange(250)
        numpy.random.RandomState(0).shuffle(i)
        pred = ssvm.predict(x[i])
        rmse = numpy.sqrt(mean_squared_error(self.y['lenfol'][i], pred))
        self.assertLessEqual(abs(1339.3006854574726 - rmse), 0.25)

    def test_fit_and_predict_linear_regression_no_intercept(self):
        ssvm = FastKernelSurvivalSVM(optimizer="rbtree", rank_ratio=0.0, kernel="linear",
                                     max_iter=50, fit_intercept=False, random_state=0)
        ssvm.fit(self.x.values, self.y)

        self.assertFalse(hasattr(ssvm, "intercept_"))

        pred = ssvm.predict(self.x.values)
        rmse = numpy.sqrt(mean_squared_error(self.y['lenfol'], pred))
        self.assertAlmostEqual(15837.658418546907, rmse, 4)

    def test_fit_and_predict_rbf_rbtree(self):
        ssvm = FastKernelSurvivalSVM(optimizer="rbtree", kernel='rbf', random_state=0)
        ssvm.fit(self.x.values, self.y)

        self.assertFalse(ssvm._pairwise)
        self.assertEquals(self.x.shape[0], ssvm.coef_.shape[0])

        c = ssvm.score(self.x.values, self.y)
        self.assertAlmostEqual(0.92230102862313534, c, 3)

    def test_fit_and_predict_rbf_avltree(self):
        ssvm = FastKernelSurvivalSVM(optimizer="avltree", kernel='rbf', random_state=0)
        ssvm.fit(self.x.values, self.y)

        self.assertFalse(ssvm._pairwise)
        self.assertEquals(self.x.shape[0], ssvm.coef_.shape[0])

        c = ssvm.score(self.x.values, self.y)
        self.assertLessEqual(abs(0.92460312179802795 - c), 1e-3)

    def test_fit_and_predict_regression_rbf(self):
        ssvm = FastKernelSurvivalSVM(optimizer="rbtree", rank_ratio=0.0, kernel="rbf",
                                     max_iter=50, fit_intercept=True, random_state=0)
        ssvm.fit(self.x.values, self.y)

        self.assertFalse(ssvm._pairwise)
        self.assertAlmostEqual(4.9267218894089533, ssvm.intercept_)

        pred = ssvm.predict(self.x.values)
        rmse = numpy.sqrt(mean_squared_error(self.y['lenfol'], pred))
        self.assertAlmostEqual(783.525277, rmse, 6)

    def test_fit_and_predict_hybrid_rbf(self):
        ssvm = FastKernelSurvivalSVM(optimizer="rbtree", rank_ratio=0.5, kernel="rbf",
                                     max_iter=50, fit_intercept=True, random_state=0)
        ssvm.fit(self.x.values, self.y)

        self.assertFalse(ssvm._pairwise)
        self.assertLessEqual(abs(5.0289145697617164 - ssvm.intercept_), 0.04)

        pred = ssvm.predict(self.x.values)
        rmse = numpy.sqrt(mean_squared_error(self.y['lenfol'], pred))
        self.assertLessEqual(abs(880.20361811281487 - rmse), 75)

    def test_fit_and_predict_clinical_kernel(self):
        x_full, y = load_whas500()

        trans = ClinicalKernelTransform()
        trans.fit(x_full)

        x = encode_categorical(standardize(x_full))

        ssvm = FastKernelSurvivalSVM(optimizer="rbtree", kernel=trans.pairwise_kernel, max_iter=100, random_state=0)
        ssvm.fit(x.values, y)

        self.assertFalse(ssvm._pairwise)
        self.assertEquals(x.shape[0], ssvm.coef_.shape[0])

        c = ssvm.score(x.values, y)
        self.assertLessEqual(abs(0.83699051218246412 - c), 1e-3)

    def test_compare_rbf(self):
        x, y = load_whas500()
        x = encode_categorical(standardize(x))

        kpca = KernelPCA(kernel="rbf")
        xt = kpca.fit_transform(x)

        nrsvm = FastSurvivalSVM(optimizer='rbtree', tol=1e-8, max_iter=1000, random_state=0)
        nrsvm.fit(xt, y)

        rsvm = FastKernelSurvivalSVM(optimizer='rbtree', kernel="rbf",
                                     tol=1e-8, max_iter=1000, random_state=0)
        rsvm.fit(x, y)

        pred_nrsvm = nrsvm.predict(kpca.transform(x))
        pred_rsvm = rsvm.predict(x)

        self.assertEqual(len(pred_nrsvm), len(pred_rsvm))

        c1 = concordance_index_censored(y['fstat'], y['lenfol'], pred_nrsvm)
        c2 = concordance_index_censored(y['fstat'], y['lenfol'], pred_rsvm)

        self.assertAlmostEqual(c1[0], c2[0])
        self.assertTupleEqual(c1[1:], c2[1:])

    def test_compare_clinical_kernel(self):
        x_full, y = load_whas500()

        trans = ClinicalKernelTransform()
        trans.fit(x_full)

        x = encode_categorical(standardize(x_full))

        kpca = KernelPCA(kernel=trans.pairwise_kernel)
        xt = kpca.fit_transform(x)

        nrsvm = FastSurvivalSVM(optimizer='rbtree', tol=1e-8, max_iter=1000, random_state=0)
        nrsvm.fit(xt, y)

        rsvm = FastKernelSurvivalSVM(optimizer='rbtree', kernel=trans.pairwise_kernel,
                                     tol=1e-8, max_iter=1000, random_state=0)
        rsvm.fit(x, y)

        pred_nrsvm = nrsvm.predict(kpca.transform(x))
        pred_rsvm = rsvm.predict(x)

        self.assertEqual(len(pred_nrsvm), len(pred_rsvm))

        c1 = concordance_index_censored(y['fstat'], y['lenfol'], pred_nrsvm)
        c2 = concordance_index_censored(y['fstat'], y['lenfol'], pred_rsvm)

        self.assertAlmostEqual(c1[0], c2[0])
        self.assertTupleEqual(c1[1:], c2[1:])

    def test_fit_precomputed_kernel_invalid_shape(self):
        ssvm = FastKernelSurvivalSVM(optimizer="rbtree", kernel='precomputed', random_state=0)
        x = numpy.empty((100, 14))
        y = numpy.fromiter(zip(numpy.ones(100), numpy.ones(100)), dtype=[('event', bool), ('time', float)])

        self.assertRaisesRegex(ValueError, r"Precomputed metric requires shape \(n_queries, n_indexed\)\. "
                                           r"Got \(100, 14\) for 100 indexed\.",
                               ssvm.fit, x, y)

    def test_fit_precomputed_kernel_not_symmetric(self):
        ssvm = FastKernelSurvivalSVM(optimizer="rbtree", kernel='precomputed', random_state=0)
        x = numpy.random.randn(100, 100)
        x[10, 12] = -1
        x[12, 10] = 9
        y = numpy.fromiter(zip(numpy.ones(100), numpy.ones(100)), dtype=[('event', bool), ('time', float)])

        self.assertRaisesRegex(ValueError, "kernel matrix is not symmetric",
                               ssvm.fit, x, y)

    def test_predict_precomputed_kernel_invalid_shape(self):
        ssvm = FastKernelSurvivalSVM(optimizer="rbtree", kernel='precomputed', random_state=0)
        x = numpy.dot(self.x.values, self.x.values.T)
        ssvm.fit(x, self.y)

        x_new = numpy.empty((100, 14))
        self.assertRaisesRegex(ValueError, r"Precomputed metric requires shape \(n_queries, n_indexed\)\. "
                                           r"Got \(100, 14\) for 500 indexed\.",
                               ssvm.predict, x_new)


class SurvivalCounterCases(object):
    TEST_CLASS = None

    def setup_01(self):
        w = numpy.array([-0.9, -0.7, -0.1, 0.15, 0.2, 1.6])
        y = numpy.array([2,       0,    4,    3,   5,   1])
        event = numpy.array([True, True, False, True, False, True])
        x = numpy.eye(6)
        v = numpy.arange(6)
        return x, y, event, w, v

    def test_calculate_01(self):
        x, y, event, w, v = self.setup_01()
        if issubclass(self.TEST_CLASS, SurvivalCounter):
            counter = self.TEST_CLASS(x, y, event, n_relevance_levels=6)
        else:
            counter = self.TEST_CLASS(x, y, event)
        counter.update_sort_order(w)

        l_plus, xv_plus, l_minus, xv_minus = counter.calculate(v)

        assert_array_equal(numpy.array([1, 4, 0, 2, 0, 4]), l_plus)
        assert_array_equal(numpy.array([2, 9, 0, 6, 0, 9]), xv_plus)
        assert_array_equal(numpy.array([2, 0, 4, 2, 3, 0]), l_minus)
        assert_array_equal(numpy.array([6, 0, 9, 6, 9, 0]), xv_minus)

    def test_calculate_01_reverse(self):
        x, y, event, w, v = self.setup_01()
        if issubclass(self.TEST_CLASS, SurvivalCounter):
            counter = self.TEST_CLASS(x, y[::-1], event[::-1], n_relevance_levels=6)
        else:
            counter = self.TEST_CLASS(x, y[::-1], event[::-1])
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

    def test_calculate_02(self):
        x, y, event, w, v = self.setup_02()
        if issubclass(self.TEST_CLASS, SurvivalCounter):
            counter = self.TEST_CLASS(x, y, event, n_relevance_levels=10)
        else:
            counter = self.TEST_CLASS(x, y, event)
        counter.update_sort_order(w)

        l_plus, xv_plus, l_minus, xv_minus = counter.calculate(v)

        assert_array_equal(numpy.array([0, 0, 0, 1, 0, 2, 6, 0, 7, 0]), l_plus)
        assert_array_equal(numpy.array([0, 0, 0, 4, 0, 7, 21, 0, 30, 0]), xv_plus)
        assert_array_equal(numpy.array([2, 0, 2, 3, 4, 2, 0, 2, 0, 1]), l_minus)
        assert_array_equal(numpy.array([14, 0, 14, 19, 22, 14, 0, 14, 0, 8]), xv_minus)

    def test_calculate_02_reverse(self):
        x, y, event, w, v = self.setup_02()
        if issubclass(self.TEST_CLASS, SurvivalCounter):
            counter = self.TEST_CLASS(x, y[::-1], event[::-1], n_relevance_levels=10)
        else:
            counter = self.TEST_CLASS(x, y[::-1], event[::-1])
        counter.update_sort_order(w[::-1])

        l_plus, xv_plus, l_minus, xv_minus = counter.calculate(v[::-1])

        assert_array_equal(numpy.array([0, 7, 0, 6, 2, 0, 1, 0, 0, 0]), l_plus)
        assert_array_equal(numpy.array([0, 30, 0, 21, 7, 0, 4, 0, 0, 0]), xv_plus)
        assert_array_equal(numpy.array([1, 0, 2, 0, 2, 4, 3, 2, 0, 2]), l_minus)
        assert_array_equal(numpy.array([8, 0, 14, 0, 14, 22, 19, 14, 0, 14]), xv_minus)


class TestSurvivalCounter(SurvivalCounterCases, TestCase):
    TEST_CLASS = SurvivalCounter


class TestRBTreeSurvivalCounter(SurvivalCounterCases, TestCase):
    TEST_CLASS = partial(OrderStatisticTreeSurvivalCounter, tree_class=RBTree)


class TestAVLTreeSurvivalCounter(SurvivalCounterCases, TestCase):
    TEST_CLASS = partial(OrderStatisticTreeSurvivalCounter, tree_class=AVLTree)


class TestNaiveSurvivalSVM(TestCase):

    def setUp(self):
        # naive survival SVM does resolve ties in survival time differently,
        # therefore use data without ties
        data = loadarff(WHAS500_NOTIES_FILE)
        x, self.y = get_x_y(data, ['fstat', 'lenfol'], '1')
        self.x = encode_categorical(x)

    def test_survival_squared_hinge_loss(self):
        nrsvm = NaiveSurvivalSVM(loss='squared_hinge', dual=False, tol=1e-8, max_iter=1000, random_state=0)
        nrsvm.fit(self.x, self.y)

        rsvm = FastSurvivalSVM(optimizer='avltree', tol=1e-8, max_iter=1000, random_state=0)
        rsvm.fit(self.x, self.y)

        assert_array_almost_equal(nrsvm.coef_.ravel(), rsvm.coef_, 3)

        pred_nrsvm = nrsvm.predict(self.x)
        pred_rsvm = rsvm.predict(self.x)

        self.assertEqual(len(pred_nrsvm), len(pred_rsvm))

        c1 = concordance_index_censored(self.y['fstat'], self.y['lenfol'], pred_nrsvm)
        c2 = concordance_index_censored(self.y['fstat'], self.y['lenfol'], pred_rsvm)

        self.assertAlmostEqual(c1[0], c2[0])
        self.assertTupleEqual(c1[1:], c2[1:])


if __name__ == '__main__':
    run_module_suite()
