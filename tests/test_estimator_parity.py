"""Estimator-level parity tests against the pandas reference.

Parity tests compare pandas and one other dataframe library within a single
test, with pandas as the reference, so the compared library is a fixture
parameter (``COMPARISON_OUTPUT_TYPES``) rather than a test-level backend
axis. A new dataframe library joins these tests by adding its dataset-loader
``output_type`` name to that list; inputs that exist in only one library
(such as polars' LazyFrame) get their own explicit tests instead.
"""

import numpy as np
import pytest

import sksurv.datasets as sdata
from sksurv.metrics import concordance_index_censored
from sksurv.preprocessing import OneHotEncoder
from sksurv.testing import all_survival_estimators
from sksurv.testing._dataframe import COMPARISON_OUTPUT_TYPES


def _make_survival_estimator_constructors():
    def _make_constructor(estimator_cls):
        def _ctor():
            estimator = estimator_cls()
            params = estimator.get_params()

            if "random_state" in params:
                estimator.set_params(random_state=0xF1)
            if "n_estimators" in params:
                estimator.set_params(n_estimators=5)

            name = estimator_cls.__name__
            match name:
                case "CoxnetSurvivalAnalysis":
                    estimator.set_params(n_alphas=5, fit_baseline_model=True)
                case "IPCRidge":
                    estimator.set_params(alpha=1.0)
                case "NaiveSurvivalSVM":
                    estimator.set_params(max_iter=1000, tol=1e-6)
                case "FastSurvivalSVM" | "FastKernelSurvivalSVM":
                    estimator.set_params(max_iter=100, tol=1e-6)
                case "MinlipSurvivalAnalysis" | "HingeLossSurvivalSVM":
                    estimator.set_params(solver="clarabel")

            return estimator

        return _ctor

    return [
        (estimator_cls.__name__, _make_constructor(estimator_cls))
        for estimator_cls in sorted(all_survival_estimators(), key=lambda cls: cls.__name__)
    ]


@pytest.fixture(scope="module", params=COMPARISON_OUTPUT_TYPES)
def whas500_encoded_small(request):
    X_pd, y = sdata.load_whas500()
    X_other, _ = sdata.load_whas500(output_type=request.param)
    X_pd_enc = OneHotEncoder().fit_transform(X_pd[:100])
    X_other_enc = OneHotEncoder().fit_transform(X_other[:100])
    return X_pd_enc, X_other_enc, y[:100]


@pytest.fixture(scope="module", params=COMPARISON_OUTPUT_TYPES)
def whas500_pair_small(request):
    X_pd, y = sdata.load_whas500()
    X_other, _ = sdata.load_whas500(output_type=request.param)
    return X_pd[:100], X_other[:100], y[:100]


def _assert_step_functions_equal(functions_pd, functions_other):
    assert len(functions_pd) == len(functions_other)
    for function_pd, function_other in zip(functions_pd, functions_other):
        np.testing.assert_array_equal(function_pd.x, function_other.x, strict=True)
        # Different intermediate dtype handling can produce
        # machine-epsilon-scale differences in cumulative products.
        np.testing.assert_allclose(function_pd.y, function_other.y, rtol=1e-12, strict=True)


class TestSurvivalEstimatorParity:
    ESTIMATORS = _make_survival_estimator_constructors()

    @staticmethod
    @pytest.mark.filterwarnings("ignore:NaiveSurvivalSVM is deprecated.*:DeprecationWarning")
    @pytest.mark.parametrize("name,ctor", ESTIMATORS, ids=[t[0] for t in ESTIMATORS])
    def test_estimator_matches_pandas(name, ctor, whas500_encoded_small):
        X_pd, X_other, y = whas500_encoded_small
        est_pd = ctor()
        est_pd.fit(X_pd, y)
        pred_pd = est_pd.predict(X_pd)

        est_other = ctor()
        est_other.fit(X_other, y)
        pred_other = est_other.predict(X_other)

        np.testing.assert_equal(est_pd.feature_names_in_, est_other.feature_names_in_, strict=True)

        # Iterative solvers (e.g. ecos used by Minlip / HingeLossSurvivalSVM)
        # can reach the same solution along slightly different paths when the
        # input is built through different dataframe libraries, leaving
        # convergence-level differences on a handful of elements. Allow a
        # tight tolerance instead of bit-exact equality.
        assert pred_pd.shape == (y.shape[0],)
        assert pred_pd.dtype == pred_other.dtype
        np.testing.assert_array_almost_equal(pred_pd, pred_other)
        assert est_pd.score(X_pd, y) == est_other.score(X_other, y)

        cindex_pd = concordance_index_censored(y["fstat"], y["lenfol"], pred_pd)
        cindex_other = concordance_index_censored(y["fstat"], y["lenfol"], pred_other)
        assert cindex_pd == cindex_other

        for method_name in ("predict_survival_function", "predict_cumulative_hazard_function"):
            if hasattr(est_pd, method_name):
                functions_pd = getattr(est_pd, method_name)(X_pd[:10])
                functions_other = getattr(est_other, method_name)(X_other[:10])
                _assert_step_functions_equal(functions_pd, functions_other)


class TestSklearnPipelineParity:
    @staticmethod
    def test_pipeline_matches_pandas(whas500_pair_small):
        from sklearn.pipeline import Pipeline

        from sksurv.linear_model import CoxPHSurvivalAnalysis

        X_pd, X_other, y = whas500_pair_small
        pipe_pd = Pipeline([("onehot", OneHotEncoder()), ("model", CoxPHSurvivalAnalysis())]).fit(X_pd, y)
        pipe_other = Pipeline([("onehot", OneHotEncoder()), ("model", CoxPHSurvivalAnalysis())]).fit(X_other, y)
        pred_pd = pipe_pd.predict(X_pd)
        pred_other = pipe_other.predict(X_other)
        np.testing.assert_allclose(pred_pd, pred_other, strict=True)

    @staticmethod
    def test_cross_val_score_does_not_raise(whas500_pair_small):
        from sklearn.model_selection import KFold, cross_val_score

        from sksurv.linear_model import CoxPHSurvivalAnalysis

        _X_pd, X_other, y = whas500_pair_small
        X_other_enc = OneHotEncoder().fit_transform(X_other)
        scores = cross_val_score(CoxPHSurvivalAnalysis(), X_other_enc, y, cv=KFold(3))
        assert scores.shape == (3,)

    @staticmethod
    def test_gridsearchcv_matches_pandas(whas500_encoded_small):
        from sklearn.model_selection import GridSearchCV

        from sksurv.linear_model import CoxPHSurvivalAnalysis

        X_pd, X_other, y = whas500_encoded_small
        param_grid = {"alpha": [0.01, 0.1, 1.0]}
        gs_pd = GridSearchCV(CoxPHSurvivalAnalysis(), param_grid, cv=3).fit(X_pd, y)
        gs_other = GridSearchCV(CoxPHSurvivalAnalysis(), param_grid, cv=3).fit(X_other, y)
        assert gs_pd.best_params_ == gs_other.best_params_


class TestMetaEstimatorsParity:
    @staticmethod
    def test_stacking_matches_pandas(whas500_pair_small):
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        from sksurv.meta import Stacking

        X_pd, X_other, y = whas500_pair_small
        X_pd_enc = OneHotEncoder().fit_transform(X_pd)
        X_other_enc = OneHotEncoder().fit_transform(X_other)

        s_pd = Stacking(
            meta_estimator=CoxPHSurvivalAnalysis(),
            base_estimators=[
                ("cox", CoxPHSurvivalAnalysis()),
                ("rsf", RandomSurvivalForest(n_estimators=5, random_state=0)),
            ],
            probabilities=False,
        ).fit(X_pd_enc, y)
        s_other = Stacking(
            meta_estimator=CoxPHSurvivalAnalysis(),
            base_estimators=[
                ("cox", CoxPHSurvivalAnalysis()),
                ("rsf", RandomSurvivalForest(n_estimators=5, random_state=0)),
            ],
            probabilities=False,
        ).fit(X_other_enc, y)
        pred_pd = s_pd.predict(X_pd_enc)
        pred_other = s_other.predict(X_other_enc)
        np.testing.assert_allclose(pred_pd, pred_other, strict=True)

    @staticmethod
    def test_ensemble_selection_matches_pandas(whas500_pair_small):
        from sklearn.model_selection import KFold

        from sksurv.linear_model import CoxPHSurvivalAnalysis
        from sksurv.meta import EnsembleSelection

        X_pd, X_other, y = whas500_pair_small
        X_pd_enc = OneHotEncoder().fit_transform(X_pd)
        X_other_enc = OneHotEncoder().fit_transform(X_other)

        def cindex(est, X, y):
            return concordance_index_censored(y["fstat"], y["lenfol"], est.predict(X))[0]

        es_pd = EnsembleSelection(
            base_estimators=[
                ("cox1", CoxPHSurvivalAnalysis()),
                ("cox2", CoxPHSurvivalAnalysis(alpha=0.1)),
            ],
            cv=KFold(3),
            scorer=cindex,
        ).fit(X_pd_enc, y)
        es_other = EnsembleSelection(
            base_estimators=[
                ("cox1", CoxPHSurvivalAnalysis()),
                ("cox2", CoxPHSurvivalAnalysis(alpha=0.1)),
            ],
            cv=KFold(3),
            scorer=cindex,
        ).fit(X_other_enc, y)
        pred_pd = es_pd.predict(X_pd_enc)
        pred_other = es_other.predict(X_other_enc)
        np.testing.assert_allclose(pred_pd, pred_other, strict=True)


@pytest.fixture(scope="module")
def whas500_polars_encoded_small():
    X_pl, y = sdata.load_whas500(output_type="polars")
    return OneHotEncoder().fit_transform(X_pl[:100]), y[:100]


class TestSurvivalEstimatorLazyFrame:
    """polars-specific: estimators must reject LazyFrame inputs."""

    ESTIMATORS = TestSurvivalEstimatorParity.ESTIMATORS

    @staticmethod
    @pytest.mark.filterwarnings("ignore:NaiveSurvivalSVM is deprecated.*:DeprecationWarning")
    @pytest.mark.parametrize("name,ctor", ESTIMATORS, ids=[t[0] for t in ESTIMATORS])
    def test_estimator_lazyframe_rejected(name, ctor, whas500_polars_encoded_small):
        X_pl, y = whas500_polars_encoded_small
        # fit must reject a LazyFrame
        with pytest.raises(TypeError, match=r"polars\.LazyFrame is not supported"):
            ctor().fit(X_pl.lazy(), y)

        # predict must also reject a LazyFrame (fit on eager first)
        est = ctor().fit(X_pl, y)
        with pytest.raises(TypeError, match=r"polars\.LazyFrame is not supported"):
            est.predict(X_pl.lazy())

    @staticmethod
    def test_gb_staged_predict_lazyframe_rejected(whas500_polars_encoded_small):
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis

        X_pl, y = whas500_polars_encoded_small
        gb = GradientBoostingSurvivalAnalysis(n_estimators=3, random_state=0).fit(X_pl, y)
        with pytest.raises(TypeError, match=r"polars\.LazyFrame is not supported"):
            list(gb.staged_predict(X_pl.lazy()))
