from functools import partial
from os.path import join, dirname
import unittest

import numpy
from numpy.testing import TestCase, run_module_suite, assert_array_almost_equal, assert_array_equal
from sklearn.metrics import mean_squared_error

from survival.bintrees import AVLTree, RBTree
from survival.svm.naive_survival_svm import NaiveSurvivalSVM
from survival.svm.survival_svm import FastSurvivalSVM, SurvivalCounter, \
    OrderStatisticTreeSurvivalCounter
from survival.svm._prsvm import survival_constraints_simple
from survival.io import loadarff
from survival import column
from survival.metrics import concordance_index_censored

WHAS500_FILE = join(dirname(__file__), '..', 'data', 'whas500.arff')
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
        y['time'] = numpy.arange(100, dtype=float)

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
        dat = loadarff(WHAS500_FILE)
        self.y = numpy.empty(dtype=[('event', bool), ('time', float)], shape=dat.shape[0])
        self.y['time'] = (dat['lenfol']).values
        self.y['event'] = (dat['fstat'] == '1').values
        self.x = column.encode_categorical(dat.drop(['lenfol', 'fstat'], axis=1))

    def test_fit_and_predict_ranking(self):
        ssvm = FastSurvivalSVM(optimizer=self.OPTIMIZER, random_state=0)
        ssvm.fit(self.x.values, self.y)

        pred = ssvm.predict(self.x.values)
        c = concordance_index_censored(self.y['event'], self.y['time'], pred)
        self.assertAlmostEqual(0.78597186921981665, c[0], 6)

    def test_fit_and_predict_ranking_kernel(self):
        ssvm = FastSurvivalSVM(optimizer=self.OPTIMIZER, kernel='rbf', random_state=0)
        ssvm.fit(self.x.values, self.y)

        pred = ssvm.predict(self.x.values)
        c = concordance_index_censored(self.y['event'], self.y['time'], pred)
        self.assertAlmostEqual(1.0, c[0], 6)

    def test_fit_and_predict_regression(self):
        if self.OPTIMIZER in {'simple', 'PRSVM'}:
            raise unittest.SkipTest("regression not implemented for " + self.OPTIMIZER)

        ssvm = FastSurvivalSVM(optimizer=self.OPTIMIZER, rank_ratio=0.5,
                               max_iter=1000, fit_intercept=True, random_state=0)
        self.y['time'] = numpy.log(self.y['time'])
        ssvm.fit(self.x.values, self.y)

        pred = ssvm.predict(self.x.values)
        rmse = numpy.sqrt(mean_squared_error(self.y['time'], pred))
        self.assertEqual(5, int(rmse))

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

        assert_array_equal(numpy.array([1, 4, 0, 2, 0, 4]), l_plus)
        assert_array_equal(numpy.array([2, 9, 0, 6, 0, 9]), xv_plus)
        assert_array_equal(numpy.array([2, 0, 4, 2, 3, 0]), l_minus)
        assert_array_equal(numpy.array([6, 0, 9, 6, 9, 0]), xv_minus)

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

        assert_array_equal(numpy.array([0, 0, 0, 1, 0, 2, 6, 0, 7, 0]), l_plus)
        assert_array_equal(numpy.array([0, 0, 0, 4, 0, 7, 21, 0, 30, 0]), xv_plus)
        assert_array_equal(numpy.array([2, 0, 2, 3, 4, 2, 0, 2, 0, 1]), l_minus)
        assert_array_equal(numpy.array([14, 0, 14, 19, 22, 14, 0, 14, 0, 8]), xv_minus)


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
        dat = loadarff(WHAS500_NOTIES_FILE)
        self.y = numpy.empty(dtype=[('fstat', bool), ('lenfol', float)], shape=dat.shape[0])
        self.y['lenfol'] = dat['lenfol'].values
        self.y['fstat'] = (dat['fstat'] == '1').values
        self.x = column.encode_categorical(dat.drop(['lenfol', 'fstat'], axis=1)).values

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
