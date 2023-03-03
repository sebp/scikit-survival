import numpy as np
from numpy.testing import assert_array_equal
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, ParameterGrid

from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis
from sksurv.kernels import ClinicalKernelTransform
from sksurv.linear_model import IPCRidge
from sksurv.meta import EnsembleSelection, EnsembleSelectionRegressor
from sksurv.metrics import concordance_index_censored
from sksurv.svm import FastKernelSurvivalSVM, FastSurvivalSVM
from sksurv.testing import FixtureParameterFactory, assert_cindex_almost_equal
from sksurv.tree import SurvivalTree
from sksurv.util import check_array_survival


def score_cindex(est, X_test, y_test, **predict_params):
    prediction = est.predict(X_test, **predict_params)

    res = concordance_index_censored(y_test['fstat'], y_test['lenfol'], prediction)
    return res[0]


def _create_survival_ensemble(**kwargs):
    boosting_grid = ParameterGrid({"n_estimators": [100, 250], "subsample": [1.0, 0.75, 0.5]})
    alphas = np.exp(np.linspace(np.log(0.001), np.log(2), 5))
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


class TestEnsembleSelectionSurvivalAnalysis:
    @staticmethod
    @pytest.mark.slow()
    def test_fit(make_whas500):
        whas500 = make_whas500(with_mean=False, with_std=False, to_numeric=True)
        meta = _create_survival_ensemble()
        assert len(meta) == 0

        meta.fit(whas500.x, whas500.y)
        assert len(meta) == 11
        assert meta.scores_.shape == (11,)

        p = meta.predict(whas500.x)

        assert_cindex_almost_equal(
            whas500.y['fstat'], whas500.y['lenfol'], p, (0.7863312, 59088, 16053, 8, 14)
        )

        c_index = meta.score(whas500.x, whas500.y)
        assert round(abs(c_index - 0.7863312), 6) == 0

    @staticmethod
    @pytest.mark.slow()
    def test_fit_spearman_correlation(make_whas500):
        whas500 = make_whas500(with_mean=False, with_std=False, to_numeric=True)
        meta = _create_survival_ensemble(correlation="spearman")
        assert len(meta) == 0

        meta.fit(whas500.x, whas500.y)

        p = meta.predict(whas500.x)

        assert_cindex_almost_equal(
            whas500.y['fstat'], whas500.y['lenfol'], p, (0.7863312, 59088, 16053, 8, 14)
        )

    @staticmethod
    @pytest.mark.slow()
    def test_fit_kendall_correlation(make_whas500):
        whas500 = make_whas500(with_mean=False, with_std=False, to_numeric=True)
        meta = _create_survival_ensemble(correlation="kendall")
        assert len(meta) == 0

        meta.fit(whas500.x, whas500.y)

        p = meta.predict(whas500.x)

        assert_cindex_almost_equal(
            whas500.y['fstat'], whas500.y['lenfol'], p, (0.7663043, 57570, 17545, 34, 14)
        )

    @staticmethod
    @pytest.mark.slow()
    def test_fit_custom_kernel(make_whas500):
        whas500 = make_whas500(with_mean=False, with_std=False, to_numeric=True)
        alphas = np.exp(np.linspace(np.log(0.001), np.log(0.5), 5))
        svm_grid = ParameterGrid({"alpha": alphas})

        transform = ClinicalKernelTransform(fit_once=True)
        transform.prepare(whas500.x_data_frame)

        base_estimators = []
        for i, params in enumerate(svm_grid):
            model = FastSurvivalSVM(max_iter=100, random_state=0, **params)
            base_estimators.append(("svm_linear_%d" % i, model))

        for i, params in enumerate(svm_grid):
            model = FastKernelSurvivalSVM(
                kernel=transform.pairwise_kernel, max_iter=45, tol=1e-5, random_state=0, **params
            )
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
    def test_feature_names_in(make_whas500):
        whas500 = make_whas500(with_mean=False, with_std=False, to_numeric=True)

        base_estimators = [
            ('tree%d' % i, SurvivalTree(max_depth=1, max_features=1, random_state=i))
            for i in range(10)
        ]

        cv = KFold(n_splits=3, shuffle=True, random_state=0)
        meta = EnsembleSelection(base_estimators, n_estimators=0.5, scorer=score_cindex, cv=cv)

        meta.fit(whas500.x_data_frame, whas500.y)
        feature_names = whas500.x_data_frame.columns.values
        assert meta.n_features_in_ == len(feature_names)
        assert_array_equal(meta.feature_names_in_, feature_names)

        warn_msg = "X does not have valid feature names, but SurvivalTree was fitted with feature names"
        with pytest.warns(UserWarning, match=warn_msg):
            meta.predict(whas500.x[:10])

        meta.fit(whas500.x, whas500.y)
        assert meta.n_features_in_ == len(feature_names)

        with pytest.raises(AttributeError, match="object has no attribute 'feature_names_in_'"):
            meta.feature_names_in_  # pylint: disable=pointless-statement

        warn_msg = "X has feature names, but SurvivalTree was fitted without feature names"
        with pytest.warns(UserWarning, match=warn_msg):
            meta.predict(whas500.x_data_frame.iloc[:10])


class EnsembleSelectionFailureCases(FixtureParameterFactory):
    _msg_min_correlation = (
        "The 'min_correlation' parameter of EnsembleSelection must be "
        r"a float in the range \[-1, 1\]\. "
        "Got {value} instead\\."
    )
    _msg_correlation = (
        "The 'correlation' parameter of EnsembleSelection must be "
        r"a str among {{.+}}\. "
        "Got {value!r} instead\\."
    )
    _msg_scorer = (
        "The 'scorer' parameter of EnsembleSelection must be "
        r"a callable\. Got {value} instead\."
    )

    def data_min_score(self):
        params = {"scorer": score_cindex, "min_score": 1.0, "cv": 3}
        match = "no base estimator exceeds min_score, try decreasing it"
        return params, ValueError, match

    def data_min_correlation_0(self):
        params = {"scorer": score_cindex, "min_correlation": 1.2}
        match = self._msg_min_correlation.format(value=1.2)
        return params, ValueError, match

    def data_min_correlation_1(self):
        params = {"scorer": score_cindex, "min_correlation": -2.1}
        match = self._msg_min_correlation.format(value=-2.1)
        return params, ValueError, match

    def data_min_correlation_2(self):
        params = {"scorer": score_cindex, "min_correlation": np.nan}
        match = self._msg_min_correlation.format(value=np.nan)
        return params, ValueError, match

    def data_scorer_none(self):
        params = {"scorer": None}
        match = self._msg_scorer.format(value=None)
        return params, TypeError, match

    def data_scorer_not_callable(self):
        params = {"scorer": np.zeros(10)}
        value = r"array\(\[0\., 0\., 0\., 0\., 0\., 0\., 0\., 0\., 0\., 0\.\]\)"
        match = self._msg_scorer.format(value=value)
        return params, TypeError, match

    def data_n_estimators_0(self):
        params = {"scorer": score_cindex, "n_estimators": 0}
        match = (
            "The 'n_estimators' parameter of EnsembleSelection must be "
            r"an int in the range \[1, inf\) or a float in the range \(0\.0, 1\.0\]\. "
            r"Got 0 instead\."
        )
        return params, ValueError, match

    def data_n_estimators_1000(self):
        params = {"scorer": score_cindex, "n_estimators": 1000}
        match = r"n_estimators \(1000\) must not exceed number of base learners \(2\)"
        return params, ValueError, match

    def data_correlation_none(self):
        params = {"scorer": score_cindex, "correlation": None}
        match = self._msg_correlation.format(value=None)
        return params, ValueError, match

    def data_correlation_int(self):
        params = {"scorer": score_cindex, "correlation": 2143}
        match = self._msg_correlation.format(value=2143)
        return params, ValueError, match

    def data_correlation_str(self):
        params = {"scorer": score_cindex, "correlation": "clearly wrong"}
        match = self._msg_correlation.format(value='clearly wrong')
        return params, ValueError, match


@pytest.mark.parametrize("params,error_cls,error", EnsembleSelectionFailureCases().get_cases())
def test_ensemble_selection_failures(params, error_cls, error, make_whas500):
    whas500 = make_whas500(with_mean=False, with_std=False, to_numeric=True)
    base_estimators = [
        ('gbm', ComponentwiseGradientBoostingSurvivalAnalysis()),
        ('svm', FastSurvivalSVM()),
    ]
    meta = EnsembleSelection(base_estimators, **params)

    with pytest.raises(error_cls, match=error):
        meta.fit(whas500.x, whas500.y)


def _score_rmse(est, X_test, y_test, **predict_params):
    prediction = est.predict(X_test, **predict_params)

    m = y_test['fstat']
    res = mean_squared_error(y_test['lenfol'][m], prediction[m])
    return np.sqrt(res)


class DummySurvivalRegressor(DummyRegressor):
    def __init__(self, strategy="mean", constant=None, quantile=None):
        super().__init__(strategy=strategy, constant=constant, quantile=quantile)
        if hasattr(DummyRegressor, "n_features_in_"):
            delattr(DummyRegressor, "n_features_in_")

    def fit(self, X, y, sample_weight=None):
        _, time = check_array_survival(X, y)
        return super().fit(X, time)


def _create_regression_ensemble():
    aft_grid = ParameterGrid({"alpha": 2. ** np.arange(-2, 12, 2)})
    svm_grid = ParameterGrid({"alpha": 2. ** np.arange(-12, 0, 2)})

    base_estimators = []
    for i, params in enumerate(aft_grid):
        model = IPCRidge(max_iter=1000, **params)
        base_estimators.append(("aft_%d" % i, model))

    for i, params in enumerate(svm_grid):
        model = FastSurvivalSVM(
            rank_ratio=0, fit_intercept=True, max_iter=100, random_state=1, **params
        )
        base_estimators.append(("svm_%d" % i, model))

    cv = KFold(n_splits=4, shuffle=True, random_state=0)
    meta = EnsembleSelectionRegressor(
        base_estimators, n_estimators=0.4, scorer=_score_rmse, cv=cv, n_jobs=1,
    )
    return meta


class TestEnsembleSelectionRegressor:
    @staticmethod
    @pytest.mark.slow()
    def test_fit_and_predict(make_whas500):
        whas500 = make_whas500(with_mean=True, with_std=True, to_numeric=True)
        meta = _create_regression_ensemble()
        assert len(meta) == 0

        meta.fit(whas500.x_data_frame.iloc[:400], whas500.y[:400])
        assert meta.scores_.shape[0] >= len(meta)
        assert len(meta) == 5
        assert meta.scores_.shape[0] == 9

        feature_names = whas500.x_data_frame.columns.values
        assert meta.n_features_in_ == len(feature_names)
        assert_array_equal(meta.feature_names_in_, feature_names)

        p = meta.predict(whas500.x_data_frame.iloc[400:])
        score = np.sqrt(mean_squared_error(whas500.y[400:]['lenfol'], p))
        assert abs(score - 423.82894756865056) <= 0.1

        c_index = meta.score(whas500.x_data_frame.iloc[:400], whas500.y[:400])
        assert round(abs(c_index - 0.767067946974157), 6) == 0

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

        meta = EnsembleSelectionRegressor(
            base_estimators, n_estimators=1, min_score=5, cv=5, scorer=_score_rmse,
        )

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

        def _score(est, X_test, y_test, **predict_params):  # pylint: disable=unused-argument
            return 'invalid'

        meta = EnsembleSelectionRegressor(
            base_estimators, n_estimators=1, min_score=5, cv=5, scorer=_score,
        )

        with pytest.raises(ValueError,
                           match=r"scoring must return a number, got invalid "
                                 r"\(<class 'str'>\) instead\."):
            meta.fit(whas500.x, whas500.y)
