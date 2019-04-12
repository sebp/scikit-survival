import numpy
import pytest
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error

from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis
from sksurv.kernels import ClinicalKernelTransform
from sksurv.linear_model import IPCRidge
from sksurv.meta import EnsembleSelection, EnsembleSelectionRegressor
from sksurv.metrics import concordance_index_censored
from sksurv.svm import FastSurvivalSVM, FastKernelSurvivalSVM
from sksurv.testing import assert_cindex_almost_equal
from sksurv.util import check_arrays_survival


def score_cindex(est, X_test, y_test, **predict_params):
    prediction = est.predict(X_test, **predict_params)

    res = concordance_index_censored(y_test['fstat'], y_test['lenfol'], prediction)
    return res[0]


def _create_survival_ensemble(**kwargs):
    boosting_grid = ParameterGrid({"n_estimators": [100, 250], "subsample": [1.0, 0.75, 0.5]})
    alphas = numpy.exp(numpy.linspace(numpy.log(0.001), numpy.log(2), 5))
    svm_grid = ParameterGrid({"alpha": alphas})

    base_estimators = []
    for i, params in enumerate(boosting_grid):
        model = ComponentwiseGradientBoostingSurvivalAnalysis(random_state=0, **params)
        base_estimators.append(("gbm_%d" % i, model))

    for i, params in enumerate(svm_grid):
        model = FastSurvivalSVM(max_iter=100, tol=1e-6, random_state=0, **params)
        base_estimators.append(("svm_%d" % i, model))

    cv = KFold(n_splits=3, shuffle=True, random_state=0)
    meta = EnsembleSelection(base_estimators, n_estimators=0.4, scorer=score_cindex, cv=cv, **kwargs)
    return meta


class TestEnsembleSelectionSurvivalAnalysis(object):
    @staticmethod
    @pytest.mark.slow
    def test_fit(make_whas500):
        whas500 = make_whas500(with_mean=False, with_std=False, to_numeric=True)
        meta = _create_survival_ensemble()
        assert len(meta) == 0

        meta.fit(whas500.x, whas500.y)
        assert len(meta) == 11
        assert meta.scores_.shape == (11,)

        p = meta.predict(whas500.x)

        assert_cindex_almost_equal(whas500.y['fstat'], whas500.y['lenfol'], p,
                                   (0.7863312, 59088, 16053, 8, 14))

    @staticmethod
    @pytest.mark.slow
    def test_fit_spearman_correlation(make_whas500):
        whas500 = make_whas500(with_mean=False, with_std=False, to_numeric=True)
        meta = _create_survival_ensemble(correlation="spearman")
        assert len(meta) == 0

        meta.fit(whas500.x, whas500.y)

        p = meta.predict(whas500.x)

        assert_cindex_almost_equal(whas500.y['fstat'], whas500.y['lenfol'], p,
                                   (0.7863312, 59088, 16053, 8, 14))

    @staticmethod
    @pytest.mark.slow
    def test_fit_kendall_correlation(make_whas500):
        whas500 = make_whas500(with_mean=False, with_std=False, to_numeric=True)
        meta = _create_survival_ensemble(correlation="kendall")
        assert len(meta) == 0

        meta.fit(whas500.x, whas500.y)

        p = meta.predict(whas500.x)

        assert_cindex_almost_equal(whas500.y['fstat'], whas500.y['lenfol'], p,
                                   (0.7663043, 57570, 17545, 34, 14))

    @staticmethod
    @pytest.mark.slow
    def test_fit_custom_kernel(make_whas500):
        whas500 = make_whas500(with_mean=False, with_std=False, to_numeric=True)
        alphas = numpy.exp(numpy.linspace(numpy.log(0.001), numpy.log(0.5), 5))
        svm_grid = ParameterGrid({"alpha": alphas})

        transform = ClinicalKernelTransform(fit_once=True)
        transform.prepare(whas500.x_data_frame)

        base_estimators = []
        for i, params in enumerate(svm_grid):
            model = FastSurvivalSVM(max_iter=100, random_state=0, **params)
            base_estimators.append(("svm_linear_%d" % i, model))

        for i, params in enumerate(svm_grid):
            model = FastKernelSurvivalSVM(kernel=transform.pairwise_kernel, max_iter=45, tol=1e-5,
                                          random_state=0, **params)
            base_estimators.append(("svm_kernel_%d" % i, model))

        cv = KFold(n_splits=3, shuffle=True, random_state=0)
        meta = EnsembleSelection(base_estimators, n_estimators=0.4, scorer=score_cindex, cv=cv, n_jobs=4)

        meta.fit(whas500.x, whas500.y)
        assert len(meta) == 10
        assert meta.scores_.shape == (10,)

        p = meta.predict(whas500.x)

        assert_cindex_almost_equal(whas500.y['fstat'], whas500.y['lenfol'], p,
                                   (0.7978084, 59938, 15178, 33, 14))

    @staticmethod
    def test_min_score(make_whas500):
        whas500 = make_whas500(with_mean=False, with_std=False, to_numeric=True)
        base_estimators = [('gbm', ComponentwiseGradientBoostingSurvivalAnalysis()),
                           ('svm', FastSurvivalSVM())]
        meta = EnsembleSelection(base_estimators, scorer=score_cindex, min_score=1.0, cv=3)

        with pytest.raises(ValueError,
                           match="no base estimator exceeds min_score, try decreasing it"):
            meta.fit(whas500.x, whas500.y)

    @staticmethod
    def test_min_correlation(make_whas500):
        whas500 = make_whas500(with_mean=False, with_std=False, to_numeric=True)
        base_estimators = [('gbm', ComponentwiseGradientBoostingSurvivalAnalysis()),
                           ('svm', FastSurvivalSVM())]
        meta = EnsembleSelection(base_estimators, scorer=score_cindex, min_correlation=1.2)

        with pytest.raises(ValueError,
                           match=r"min_correlation must be in \[-1; 1\], but was 1.2"):
            meta.fit(whas500.x, whas500.y)

        meta.set_params(min_correlation=-2.1)
        with pytest.raises(ValueError,
                           match=r"min_correlation must be in \[-1; 1\], but was -2.1"):
            meta.fit(whas500.x, whas500.y)

        meta.set_params(min_correlation=numpy.nan)
        with pytest.raises(ValueError,
                           match=r"min_correlation must be in \[-1; 1\], but was nan"):
            meta.fit(whas500.x, whas500.y)

    @staticmethod
    def test_scorer(make_whas500):
        whas500 = make_whas500(with_mean=False, with_std=False, to_numeric=True)
        base_estimators = [('gbm', ComponentwiseGradientBoostingSurvivalAnalysis()),
                           ('svm', FastSurvivalSVM())]
        meta = EnsembleSelection(base_estimators, scorer=None)

        with pytest.raises(TypeError, match="scorer is not callable"):
            meta.fit(whas500.x, whas500.y)

        meta.set_params(scorer=numpy.zeros(10))
        with pytest.raises(TypeError, match="scorer is not callable"):
            meta.fit(whas500.x, whas500.y)

    @staticmethod
    def test_n_estimators(make_whas500):
        whas500 = make_whas500(with_mean=False, with_std=False, to_numeric=True)
        base_estimators = [('gbm', ComponentwiseGradientBoostingSurvivalAnalysis()),
                           ('svm', FastSurvivalSVM())]
        meta = EnsembleSelection(base_estimators, scorer=score_cindex, n_estimators=0)

        with pytest.raises(ValueError, match="n_estimators must not be zero or negative"):
            meta.fit(whas500.x, whas500.y)

        meta.set_params(n_estimators=1000)
        with pytest.raises(ValueError,
                           match=r"n_estimators \(1000\) must not exceed number "
                                 r"of base learners \(2\)"):
            meta.fit(whas500.x, whas500.y)

    @staticmethod
    def test_correlation(make_whas500):
        whas500 = make_whas500(with_mean=False, with_std=False, to_numeric=True)
        base_estimators = [('gbm', ComponentwiseGradientBoostingSurvivalAnalysis()),
                           ('svm', FastSurvivalSVM())]
        meta = EnsembleSelection(base_estimators, scorer=score_cindex, correlation=None)
        with pytest.raises(ValueError,
                           match="correlation must be one of 'pearson', 'kendall', "
                                 "and 'spearman', but got None"):
            meta.fit(whas500.x, whas500.y)

        meta = EnsembleSelection(base_estimators, scorer=score_cindex, correlation=2143)
        with pytest.raises(ValueError,
                           match="correlation must be one of 'pearson', 'kendall', "
                                 "and 'spearman', but got 2143"):
            meta.fit(whas500.x, whas500.y)

        meta = EnsembleSelection(base_estimators, scorer=score_cindex, correlation="clearly wrong")
        with pytest.raises(ValueError,
                           match="correlation must be one of 'pearson', 'kendall', "
                                 "and 'spearman', but got 'clearly wrong'"):
            meta.fit(whas500.x, whas500.y)


def _score_rmse(est, X_test, y_test, **predict_params):
    prediction = est.predict(X_test, **predict_params)

    m = y_test['fstat']
    res = mean_squared_error(y_test['lenfol'][m], prediction[m])
    return numpy.sqrt(res)


class DummySurvivalRegressor(DummyRegressor):
    def __init__(self, strategy="mean", constant=None, quantile=None):
        super().__init__(strategy=strategy, constant=constant, quantile=quantile)

    def fit(self, X, y, sample_weight=None):
        X, _, time = check_arrays_survival(X, y)
        return super().fit(X, time)


def _create_regression_ensemble():
    aft_grid = ParameterGrid({"alpha": 2. ** numpy.arange(-2, 12, 2)})
    svm_grid = ParameterGrid({"alpha": 2. ** numpy.arange(-12, 0, 2)})

    base_estimators = []
    for i, params in enumerate(aft_grid):
        model = IPCRidge(max_iter=1000, **params)
        base_estimators.append(("aft_%d" % i, model))

    for i, params in enumerate(svm_grid):
        model = FastSurvivalSVM(rank_ratio=0, fit_intercept=True, max_iter=100,
                                random_state=1, **params)
        base_estimators.append(("svm_%d" % i, model))

    cv = KFold(n_splits=4, shuffle=True, random_state=0)
    meta = EnsembleSelectionRegressor(base_estimators, n_estimators=0.4,
                                      scorer=_score_rmse,
                                      cv=cv, n_jobs=1)
    return meta


class TestEnsembleSelectionRegressor(object):
    @staticmethod
    @pytest.mark.slow
    def test_fit_and_predict(make_whas500):
        whas500 = make_whas500(with_mean=True, with_std=True, to_numeric=True)
        meta = _create_regression_ensemble()
        assert len(meta) == 0

        meta.fit(whas500.x[:400], whas500.y[:400])
        assert meta.scores_.shape[0] >= len(meta)
        assert len(meta) == 5
        assert meta.scores_.shape[0] == 9

        p = meta.predict(whas500.x[400:])
        score = numpy.sqrt(mean_squared_error(whas500.y[400:]['lenfol'], p))
        assert abs(score - 423.82894756865056) <= 0.1

    @staticmethod
    def test_fit_dummy(make_whas500):
        whas500 = make_whas500(with_mean=False, with_std=False, to_numeric=True)
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

        with pytest.raises(ValueError,
                           match="no base estimator exceeds min_score, try decreasing it"):
            meta.fit(whas500.x, whas500.y)

    @staticmethod
    def test_invalid_scorer(make_whas500):
        whas500 = make_whas500(with_mean=False, with_std=False, to_numeric=True)
        base_estimators = [
            ('dummy_0', DummySurvivalRegressor(strategy="mean")),
            ('dummy_1', DummySurvivalRegressor(strategy="median")),
        ]

        def _score(est, X_test, y_test, **predict_params):
            return 'invalid'

        meta = EnsembleSelectionRegressor(base_estimators, n_estimators=1, min_score=5, cv=5,
                                          scorer=_score)

        with pytest.raises(ValueError,
                           match=r"scoring must return a number, got invalid "
                                 r"\(<class 'str'>\) instead\."):
            meta.fit(whas500.x, whas500.y)
