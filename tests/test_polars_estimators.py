"""Polars / LazyFrame input tests owned by sksurv survival estimators."""

from dataframe_test_utils import to_polars_dataframe
import numpy as np
import pandas as pd
import polars as pl
import pytest

import sksurv.datasets as sdata
from sksurv.metrics import concordance_index_censored
from sksurv.preprocessing import OneHotEncoder
from sksurv.svm import NaiveSurvivalSVM
from sksurv.testing import all_survival_estimators
from sksurv.util import Surv


@pytest.fixture()
def survival_smoke_data():
    def _make(dataframe_library):
        rng = np.random.default_rng(0)
        n = 30
        data = {f"f{i}": rng.standard_normal(n) for i in range(5)}
        event = rng.binomial(1, 0.5, n).astype(bool)
        time = rng.exponential(10, n)
        y = Surv.from_arrays(event, time)
        X_pd = pd.DataFrame(data)
        if dataframe_library == "pandas":
            X = X_pd
        elif dataframe_library == "polars":
            X = to_polars_dataframe(X_pd)
        else:
            raise ValueError(dataframe_library)
        return X, y

    return _make


class TestNaiveSurvivalSVMPolars:
    @staticmethod
    @pytest.mark.parametrize("dataframe_library,dataframe_class", [("pandas", pd.DataFrame), ("polars", pl.DataFrame)])
    def test_dataframe_container_preserved(survival_smoke_data, dataframe_library, dataframe_class):
        from sklearn.utils import check_random_state

        X, y = survival_smoke_data(dataframe_library)
        est = NaiveSurvivalSVM(random_state=0)
        rs = check_random_state(0)
        x_pairs, _ = est._get_survival_pairs(X, y, rs)

        assert isinstance(
            x_pairs, dataframe_class
        ), f"{dataframe_library} input should yield {dataframe_library} internal container, got {type(x_pairs)!r}"


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
                    estimator.set_params(solver="ecos")

            return estimator

        return _ctor

    return [
        (estimator_cls.__name__, _make_constructor(estimator_cls))
        for estimator_cls in sorted(all_survival_estimators(), key=lambda cls: cls.__name__)
    ]


@pytest.fixture(scope="module")
def whas500_encoded_small():
    X_pd, y = sdata.load_whas500()
    X_pl, _ = sdata.load_whas500(output_type="polars")
    X_pd_enc = OneHotEncoder().fit_transform(X_pd.iloc[:100])
    X_pl_enc = OneHotEncoder().fit_transform(X_pl.head(100))
    return X_pd_enc, X_pl_enc, y[:100]


def _assert_step_functions_equal(functions_pd, functions_pl):
    assert len(functions_pd) == len(functions_pl)
    for function_pd, function_pl in zip(functions_pd, functions_pl):
        np.testing.assert_array_equal(function_pd.x, function_pl.x, strict=True)
        # Different intermediate dtype handling can produce
        # machine-epsilon-scale differences in cumulative products.
        np.testing.assert_allclose(function_pd.y, function_pl.y, rtol=1e-12, strict=True)


class TestSurvivalEstimatorPolarsParity:
    ESTIMATORS = _make_survival_estimator_constructors()

    @staticmethod
    @pytest.mark.parametrize("name,ctor", ESTIMATORS, ids=[t[0] for t in ESTIMATORS])
    def test_estimator_polars_matches_pandas(name, ctor, whas500_encoded_small):
        X_pd, X_pl, y = whas500_encoded_small
        est_pd = ctor()
        est_pd.fit(X_pd, y)
        pred_pd = est_pd.predict(X_pd)

        est_pl = ctor()
        est_pl.fit(X_pl, y)
        pred_pl = est_pl.predict(X_pl)

        np.testing.assert_equal(est_pd.feature_names_in_, est_pl.feature_names_in_, strict=True)

        # Iterative solvers (e.g. ecos used by Minlip / HingeLossSurvivalSVM)
        # can reach the same solution along slightly different paths when the
        # input is built through different dataframe libraries, leaving
        # convergence-level differences on a handful of elements. Allow a
        # tight tolerance instead of bit-exact equality.
        assert pred_pd.shape == (y.shape[0],)
        assert pred_pd.dtype == pred_pl.dtype
        np.testing.assert_array_almost_equal(pred_pd, pred_pl)
        assert est_pd.score(X_pd, y) == est_pl.score(X_pl, y)

        cindex_pd = concordance_index_censored(y["fstat"], y["lenfol"], pred_pd)
        cindex_pl = concordance_index_censored(y["fstat"], y["lenfol"], pred_pl)
        assert cindex_pd == cindex_pl

        for method_name in ("predict_survival_function", "predict_cumulative_hazard_function"):
            if hasattr(est_pd, method_name):
                functions_pd = getattr(est_pd, method_name)(X_pd[:10])
                functions_pl = getattr(est_pl, method_name)(X_pl.head(10))
                _assert_step_functions_equal(functions_pd, functions_pl)


@pytest.fixture(scope="module")
def whas500_pl_pd_small():
    X_pd, y = sdata.load_whas500()
    X_pl, _ = sdata.load_whas500(output_type="polars")
    return X_pd.iloc[:100], X_pl.head(100), y[:100]


class TestSklearnPipelinePolars:
    @staticmethod
    def test_pipeline_polars_matches_pandas(whas500_pl_pd_small):
        from sklearn.pipeline import Pipeline

        from sksurv.linear_model import CoxPHSurvivalAnalysis

        X_pd, X_pl, y = whas500_pl_pd_small
        pipe_pd = Pipeline([("onehot", OneHotEncoder()), ("model", CoxPHSurvivalAnalysis())]).fit(X_pd, y)
        pipe_pl = Pipeline([("onehot", OneHotEncoder()), ("model", CoxPHSurvivalAnalysis())]).fit(X_pl, y)
        pred_pd = pipe_pd.predict(X_pd)
        pred_pl = pipe_pl.predict(X_pl)
        np.testing.assert_allclose(pred_pd, pred_pl, strict=True)

    @staticmethod
    def test_cross_val_score_polars_does_not_raise(whas500_pl_pd_small):
        from sklearn.model_selection import KFold, cross_val_score

        from sksurv.linear_model import CoxPHSurvivalAnalysis

        _X_pd, X_pl, y = whas500_pl_pd_small
        X_pl_enc = OneHotEncoder().fit_transform(X_pl)
        scores = cross_val_score(CoxPHSurvivalAnalysis(), X_pl_enc, y, cv=KFold(3))
        assert scores.shape == (3,)

    @staticmethod
    def test_gridsearchcv_polars(whas500_encoded_small):
        from sklearn.model_selection import GridSearchCV

        from sksurv.linear_model import CoxPHSurvivalAnalysis

        X_pd, X_pl, y = whas500_encoded_small
        param_grid = {"alpha": [0.01, 0.1, 1.0]}
        gs_pd = GridSearchCV(CoxPHSurvivalAnalysis(), param_grid, cv=3).fit(X_pd, y)
        gs_pl = GridSearchCV(CoxPHSurvivalAnalysis(), param_grid, cv=3).fit(X_pl, y)
        assert gs_pd.best_params_ == gs_pl.best_params_


class TestMetaEstimatorsPolars:
    @staticmethod
    def test_stacking_polars_matches_pandas(whas500_pl_pd_small):
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        from sksurv.meta import Stacking

        X_pd, X_pl, y = whas500_pl_pd_small
        X_pd_enc = OneHotEncoder().fit_transform(X_pd)
        X_pl_enc = OneHotEncoder().fit_transform(X_pl)

        s_pd = Stacking(
            meta_estimator=CoxPHSurvivalAnalysis(),
            base_estimators=[
                ("cox", CoxPHSurvivalAnalysis()),
                ("rsf", RandomSurvivalForest(n_estimators=5, random_state=0)),
            ],
            probabilities=False,
        ).fit(X_pd_enc, y)
        s_pl = Stacking(
            meta_estimator=CoxPHSurvivalAnalysis(),
            base_estimators=[
                ("cox", CoxPHSurvivalAnalysis()),
                ("rsf", RandomSurvivalForest(n_estimators=5, random_state=0)),
            ],
            probabilities=False,
        ).fit(X_pl_enc, y)
        pred_pd = s_pd.predict(X_pd_enc)
        pred_pl = s_pl.predict(X_pl_enc)
        np.testing.assert_allclose(pred_pd, pred_pl, strict=True)

    @staticmethod
    def test_ensemble_selection_polars_matches_pandas(whas500_pl_pd_small):
        from sklearn.model_selection import KFold

        from sksurv.linear_model import CoxPHSurvivalAnalysis
        from sksurv.meta import EnsembleSelection

        X_pd, X_pl, y = whas500_pl_pd_small
        X_pd_enc = OneHotEncoder().fit_transform(X_pd)
        X_pl_enc = OneHotEncoder().fit_transform(X_pl)

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
        es_pl = EnsembleSelection(
            base_estimators=[
                ("cox1", CoxPHSurvivalAnalysis()),
                ("cox2", CoxPHSurvivalAnalysis(alpha=0.1)),
            ],
            cv=KFold(3),
            scorer=cindex,
        ).fit(X_pl_enc, y)
        pred_pd = es_pd.predict(X_pd_enc)
        pred_pl = es_pl.predict(X_pl_enc)
        np.testing.assert_allclose(pred_pd, pred_pl, strict=True)


class TestSurvivalEstimatorLazyFrame:
    ESTIMATORS = TestSurvivalEstimatorPolarsParity.ESTIMATORS

    @staticmethod
    @pytest.mark.parametrize("name,ctor", ESTIMATORS, ids=[t[0] for t in ESTIMATORS])
    def test_estimator_lazyframe_rejected_polars(name, ctor, whas500_encoded_small):
        _X_pd, X_pl, y = whas500_encoded_small
        # fit must reject a LazyFrame
        with pytest.raises(TypeError, match=r"polars\.LazyFrame is not supported"):
            ctor().fit(X_pl.lazy(), y)

        # predict must also reject a LazyFrame (fit on eager first)
        est = ctor().fit(X_pl, y)
        with pytest.raises(TypeError, match=r"polars\.LazyFrame is not supported"):
            est.predict(X_pl.lazy())

    @staticmethod
    def test_gb_staged_predict_lazyframe_rejected(whas500_encoded_small):
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis

        _X_pd, X_pl, y = whas500_encoded_small
        gb = GradientBoostingSurvivalAnalysis(n_estimators=3, random_state=0).fit(X_pl, y)
        with pytest.raises(TypeError, match=r"polars\.LazyFrame is not supported"):
            list(gb.staged_predict(X_pl.lazy()))
