from os.path import join, dirname

import numpy
from numpy.testing import TestCase, run_module_suite, assert_array_almost_equal
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error

from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis
from sksurv.datasets import load_arff_file
from sksurv.kernels import ClinicalKernelTransform
from sksurv.linear_model import IPCRidge
from sksurv.meta import EnsembleSelection, EnsembleSelectionRegressor
from sksurv.metrics import concordance_index_censored
from sksurv.svm import FastSurvivalSVM, FastKernelSurvivalSVM
from sksurv.util import check_arrays_survival

WHAS500_FILE = join(dirname(__file__), '..', 'data', 'whas500.arff')


def score_cindex(est, X_test, y_test, **predict_params):
    prediction = est.predict(X_test, **predict_params)

    res = concordance_index_censored(y_test['fstat'], y_test['lenfol'], prediction)
    return res[0]


class TestEnsembleSelectionSurvivalAnalysis(TestCase):
    def setUp(self):
        self.x, self.y, _, _ = load_arff_file(WHAS500_FILE, ['fstat', 'lenfol'], '1',
                                              standardize_numeric=False)

    def test_fit(self):
        boosting_grid = ParameterGrid({"n_estimators": [100, 250, 500], "subsample": [1.0, 0.75, 0.5]})
        svm_grid = ParameterGrid({"alpha": 2. ** numpy.arange(-9, 9, 2)})

        base_estimators = []
        for i, params in enumerate(boosting_grid):
            model = ComponentwiseGradientBoostingSurvivalAnalysis(random_state=0, **params)
            base_estimators.append(("gbm_%d" % i, model))

        for i, params in enumerate(svm_grid):
            model = FastSurvivalSVM(max_iter=100, random_state=0, **params)
            base_estimators.append(("svm_%d" % i, model))

        cv = KFold(n_splits=5, shuffle=True, random_state=0)
        meta = EnsembleSelection(base_estimators, n_estimators=0.4, scorer=score_cindex, cv=cv, n_jobs=4)
        self.assertEqual(len(meta), 0)

        meta.fit(self.x.values, self.y)
        self.assertEqual(len(meta), 18)
        self.assertTupleEqual(meta.scores_.shape, (18,))

        p = meta.predict(self.x.values)

        score = concordance_index_censored(self.y['fstat'], self.y['lenfol'], p)
        expected_score = numpy.array([0.7860650, 59067, 16072, 10, 119])
        assert_array_almost_equal(score, expected_score)

    def test_fit_spearman_correlation(self):
        boosting_grid = ParameterGrid({"n_estimators": [100, 250, 500], "subsample": [0.75, 0.5]})
        svm_grid = ParameterGrid({"alpha": 2. ** numpy.arange(-7, 7, 2)})

        base_estimators = []
        for i, params in enumerate(boosting_grid):
            model = ComponentwiseGradientBoostingSurvivalAnalysis(random_state=0, **params)
            base_estimators.append(("gbm_%d" % i, model))

        for i, params in enumerate(svm_grid):
            model = FastSurvivalSVM(max_iter=100, random_state=0, **params)
            base_estimators.append(("svm_%d" % i, model))

        cv = KFold(n_splits=5, shuffle=True, random_state=0)
        meta = EnsembleSelection(base_estimators, n_estimators=0.4, scorer=score_cindex,
                                 correlation="spearman", cv=cv, n_jobs=4)
        self.assertEqual(len(meta), 0)

        meta.fit(self.x.values, self.y)

        p = meta.predict(self.x.values)

        score = concordance_index_censored(self.y['fstat'], self.y['lenfol'], p)
        expected_score = numpy.array([0.7860517, 59067, 16074, 8, 119])
        assert_array_almost_equal(score, expected_score)

    def test_fit_kendall_correlation(self):
        boosting_grid = ParameterGrid({"n_estimators": [100, 250, 500], "subsample": [0.75, 0.5]})
        svm_grid = ParameterGrid({"alpha": 2. ** numpy.arange(-7, 7, 2)})

        base_estimators = []
        for i, params in enumerate(boosting_grid):
            model = ComponentwiseGradientBoostingSurvivalAnalysis(random_state=0, **params)
            base_estimators.append(("gbm_%d" % i, model))

        for i, params in enumerate(svm_grid):
            model = FastSurvivalSVM(max_iter=100, random_state=0, **params)
            base_estimators.append(("svm_%d" % i, model))

        cv = KFold(n_splits=5, shuffle=True, random_state=0)
        meta = EnsembleSelection(base_estimators, n_estimators=0.4, scorer=score_cindex,
                                 correlation="kendall", cv=cv, n_jobs=4)
        self.assertEqual(len(meta), 0)

        meta.fit(self.x.values, self.y)

        p = meta.predict(self.x.values)

        score = concordance_index_censored(self.y['fstat'], self.y['lenfol'], p)
        expected_score = numpy.array([0.7741487, 58162, 16958, 29, 119])
        assert_array_almost_equal(score, expected_score)

    def test_fit_custom_kernel(self):
        svm_grid = ParameterGrid({"alpha": 2. ** numpy.arange(-5, 5, 2)})

        transform = ClinicalKernelTransform(fit_once=True)
        transform.prepare(self.x)

        base_estimators = []
        for i, params in enumerate(svm_grid):
            model = FastSurvivalSVM(max_iter=100, random_state=0, **params)
            base_estimators.append(("svm_linear_%d" % i, model))

        for i, params in enumerate(svm_grid):
            model = FastKernelSurvivalSVM(kernel=transform.pairwise_kernel, max_iter=100, random_state=0, **params)
            base_estimators.append(("svm_kernel_%d" % i, model))

        cv = KFold(n_splits=5, shuffle=True, random_state=0)
        meta = EnsembleSelection(base_estimators, n_estimators=0.4, scorer=score_cindex, cv=cv, n_jobs=4)

        meta.fit(self.x.values, self.y)
        self.assertEqual(len(meta), 10)
        self.assertTupleEqual(meta.scores_.shape, (10,))

        p = meta.predict(self.x.values)

        score = concordance_index_censored(self.y['fstat'], self.y['lenfol'], p)
        expected_score = numpy.array([0.7987066, 60007, 15112, 30, 119])
        assert_array_almost_equal(score, expected_score)

    def test_min_score(self):
        base_estimators = [('gbm', ComponentwiseGradientBoostingSurvivalAnalysis()),
                           ('svm', FastSurvivalSVM())]
        meta = EnsembleSelection(base_estimators, scorer=score_cindex, min_score=1.0)

        self.assertRaisesRegex(ValueError, "no base estimator exceeds min_score, try decreasing it",
                               meta.fit, self.x, self.y)

    def test_min_correlation(self):
        base_estimators = [('gbm', ComponentwiseGradientBoostingSurvivalAnalysis()),
                           ('svm', FastSurvivalSVM())]
        meta = EnsembleSelection(base_estimators, scorer=score_cindex, min_correlation=1.2)

        self.assertRaisesRegex(ValueError, "min_correlation must be in \[-1; 1\], but was 1.2",
                               meta.fit, self.x, self.y)

        meta.set_params(min_correlation=-2.1)
        self.assertRaisesRegex(ValueError, "min_correlation must be in \[-1; 1\], but was -2.1",
                               meta.fit, self.x, self.y)

        meta.set_params(min_correlation=numpy.nan)
        self.assertRaisesRegex(ValueError, "min_correlation must be in \[-1; 1\], but was nan",
                               meta.fit, self.x, self.y)

    def test_scorer(self):
        base_estimators = [('gbm', ComponentwiseGradientBoostingSurvivalAnalysis()),
                           ('svm', FastSurvivalSVM())]
        meta = EnsembleSelection(base_estimators, scorer=None)

        self.assertRaisesRegex(TypeError, "scorer is not callable",
                               meta.fit, self.x, self.y)

        meta.set_params(scorer=numpy.zeros(10))
        self.assertRaisesRegex(TypeError, "scorer is not callable",
                               meta.fit, self.x, self.y)

    def test_n_estimators(self):
        base_estimators = [('gbm', ComponentwiseGradientBoostingSurvivalAnalysis()),
                           ('svm', FastSurvivalSVM())]
        meta = EnsembleSelection(base_estimators, scorer=score_cindex, n_estimators=0)

        self.assertRaisesRegex(ValueError, "n_estimators must not be zero or negative",
                               meta.fit, self.x, self.y)

        meta.set_params(n_estimators=1000)
        self.assertRaisesRegex(ValueError, "n_estimators \(1000\) must not exceed number of base learners \(2\)",
                               meta.fit, self.x, self.y)

    def test_correlation(self):
        base_estimators = [('gbm', ComponentwiseGradientBoostingSurvivalAnalysis()),
                           ('svm', FastSurvivalSVM())]
        meta = EnsembleSelection(base_estimators, scorer=score_cindex, correlation=None)
        self.assertRaisesRegex(ValueError,
                               "correlation must be one of 'pearson', 'kendall', and 'spearman', but got None",
                               meta.fit, self.x, self.y)

        meta = EnsembleSelection(base_estimators, scorer=score_cindex, correlation=2143)
        self.assertRaisesRegex(ValueError,
                               "correlation must be one of 'pearson', 'kendall', and 'spearman', but got 2143",
                               meta.fit, self.x, self.y)

        meta = EnsembleSelection(base_estimators, scorer=score_cindex, correlation="clearly wrong")
        self.assertRaisesRegex(ValueError,
                               "correlation must be one of 'pearson', 'kendall', and 'spearman', but got 'clearly wrong'",
                               meta.fit, self.x, self.y)


def _score_rmse(est, X_test, y_test, **predict_params):
    prediction = est.predict(X_test, **predict_params)

    m = y_test['fstat']
    res = mean_squared_error(y_test['lenfol'][m], prediction[m])
    return numpy.sqrt(res)


class DummySurvivalRegressor(DummyRegressor):
    def __init__(self, strategy="mean", constant=None, quantile=None):
        super().__init__(strategy=strategy, constant=constant, quantile=quantile)

    def fit(self, X, y, sample_weight=None):
        X, event, time = check_arrays_survival(X, y)
        return super().fit(X, time)


class TestEnsembleSelectionRegressor(TestCase):
    def setUp(self):
        self.x, self.y, _, _ = load_arff_file(WHAS500_FILE, ['fstat', 'lenfol'], '1',
                                              standardize_numeric=False)

    def _create_ensemble(self, n):
        aft_grid = ParameterGrid({"alpha": 2. ** numpy.arange(-9, 9, 2)})
        svm_grid = ParameterGrid({"alpha": 2. ** numpy.arange(-9, 9, 2)})

        base_estimators = []
        for i, params in enumerate(aft_grid):
            model = IPCRidge(max_iter=1000, **params)
            base_estimators.append(("aft_%d" % i, model))

        for i, params in enumerate(svm_grid):
            model = FastSurvivalSVM(rank_ratio=0, fit_intercept=True, max_iter=1000, **params)
            base_estimators.append(("svm_%d" % i, model))

        cv = KFold(n_splits=5, shuffle=True, random_state=0)
        meta = EnsembleSelectionRegressor(base_estimators, n_estimators=0.4,
                                          scorer=_score_rmse,
                                          cv=cv, n_jobs=1)
        return meta

    def test_fit(self):
        meta = self._create_ensemble(self.x.shape[0])
        self.assertEqual(len(meta), 0)

        meta.fit(self.x, self.y)
        self.assertEqual(len(meta), 7)
        self.assertTupleEqual(meta.scores_.shape, (14,))

    def test_predict(self):
        meta = self._create_ensemble(400)
        meta.fit(self.x.iloc[:400].values, self.y[:400])
        p = meta.predict(self.x.iloc[400:].values)

        score = numpy.sqrt(mean_squared_error(self.y[400:]['lenfol'], p))
        self.assertLessEqual(abs(score - 1496.0799422558771), 0.1)

    def test_fit_dummy(self):
        base_estimators = [
            ('dummy_0', DummySurvivalRegressor(strategy="mean")),
            ('dummy_1', DummySurvivalRegressor(strategy="median")),
            ('dummy_2', DummySurvivalRegressor(strategy="quantile", quantile=0.1)),
            ('dummy_3', DummySurvivalRegressor(strategy="quantile", quantile=0.9)),
            ('dummy_4', DummySurvivalRegressor(strategy="quantile", quantile=0.89)),
            ('dummy_5', DummySurvivalRegressor(strategy="quantile", quantile=0.91)),
        ]

        meta = EnsembleSelectionRegressor(base_estimators, n_estimators=1, min_score=5, cv=5,
                                          scorer=_score_rmse)

        self.assertRaisesRegex(ValueError, "no base estimator exceeds min_score, try decreasing it",
                               meta.fit, self.x, self.y)


if __name__ == '__main__':
    run_module_suite()
