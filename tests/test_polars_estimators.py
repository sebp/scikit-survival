"""Polars / LazyFrame input tests owned by sksurv survival estimators."""

import numpy as np
import polars as pl
import pytest

import sksurv.datasets as sdata
from sksurv.preprocessing import OneHotEncoder
from sksurv.svm import NaiveSurvivalSVM
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
        if dataframe_library == "pandas":
            import pandas as pd

            X = pd.DataFrame(data)
        elif dataframe_library == "polars":
            X = pl.DataFrame(data)
        else:
            raise ValueError(dataframe_library)
        return X, y

    return _make


class TestNaiveSurvivalSVMPolars:
    @staticmethod
    @pytest.mark.parametrize("dataframe_library", ["pandas", "polars"])
    def test_fit_predict(survival_smoke_data, dataframe_library):
        X, y = survival_smoke_data(dataframe_library)
        est = NaiveSurvivalSVM(random_state=0)
        est.fit(X, y)
        assert list(est.feature_names_in_) == [f"f{i}" for i in range(5)]
        scores = est.predict(X)
        assert scores.shape == (X.shape[0],)

    @staticmethod
    def test_polars_internal_container_preserved(survival_smoke_data):
        from sklearn.utils import check_random_state

        X, y = survival_smoke_data("polars")
        est = NaiveSurvivalSVM(random_state=0)
        rs = check_random_state(0)
        x_pairs, _ = est._get_survival_pairs(X, y, rs)
        assert isinstance(
            x_pairs, pl.DataFrame
        ), f"polars input should yield polars internal container, got {type(x_pairs)!r}"

    @staticmethod
    def test_pandas_internal_container_preserved(survival_smoke_data):
        import pandas as pd
        from sklearn.utils import check_random_state

        X, y = survival_smoke_data("pandas")
        est = NaiveSurvivalSVM(random_state=0)
        rs = check_random_state(0)
        x_pairs, _ = est._get_survival_pairs(X, y, rs)
        assert isinstance(
            x_pairs, pd.DataFrame
        ), f"pandas input should yield pandas internal container, got {type(x_pairs)!r}"


def _make_survival_estimator_constructors():
    from sksurv.ensemble import (
        ComponentwiseGradientBoostingSurvivalAnalysis,
        ExtraSurvivalTrees,
        GradientBoostingSurvivalAnalysis,
        RandomSurvivalForest,
    )
    from sksurv.linear_model import (
        CoxnetSurvivalAnalysis,
        CoxPHSurvivalAnalysis,
        IPCRidge,
    )
    from sksurv.svm import (
        FastKernelSurvivalSVM,
        FastSurvivalSVM,
        HingeLossSurvivalSVM,
        MinlipSurvivalAnalysis,
        NaiveSurvivalSVM,
    )
    from sksurv.tree import SurvivalTree

    return [
        ("CoxPHSurvivalAnalysis", CoxPHSurvivalAnalysis),
        ("IPCRidge", lambda: IPCRidge(alpha=1.0)),
        ("CoxnetSurvivalAnalysis", lambda: CoxnetSurvivalAnalysis(n_alphas=5)),
        ("RandomSurvivalForest", lambda: RandomSurvivalForest(n_estimators=5, random_state=0)),
        ("ExtraSurvivalTrees", lambda: ExtraSurvivalTrees(n_estimators=5, random_state=0)),
        ("GradientBoostingSurvivalAnalysis", lambda: GradientBoostingSurvivalAnalysis(n_estimators=5, random_state=0)),
        (
            "ComponentwiseGradientBoostingSurvivalAnalysis",
            lambda: ComponentwiseGradientBoostingSurvivalAnalysis(n_estimators=5, random_state=0),
        ),
        ("SurvivalTree", lambda: SurvivalTree(random_state=0)),
        ("NaiveSurvivalSVM", lambda: NaiveSurvivalSVM(random_state=0, max_iter=200)),
        ("FastSurvivalSVM", lambda: FastSurvivalSVM(random_state=0, max_iter=50)),
        ("FastKernelSurvivalSVM", lambda: FastKernelSurvivalSVM(random_state=0, max_iter=50)),
        ("MinlipSurvivalAnalysis", lambda: MinlipSurvivalAnalysis(solver="ecos")),
        ("HingeLossSurvivalSVM", lambda: HingeLossSurvivalSVM(solver="ecos")),
    ]


@pytest.fixture(scope="module")
def whas500_encoded_small():
    X_pd, y = sdata.load_whas500()
    X_pl, _ = sdata.load_whas500(output_type="polars")
    X_pd_enc = OneHotEncoder().fit_transform(X_pd.iloc[:100])
    X_pl_enc = OneHotEncoder().fit_transform(X_pl.head(100))
    return X_pd_enc, X_pl_enc, y[:100]


class TestSurvivalEstimatorPolarsParity:
    ESTIMATORS = _make_survival_estimator_constructors()

    @staticmethod
    @pytest.mark.parametrize("name,ctor", ESTIMATORS, ids=[t[0] for t in ESTIMATORS])
    def test_estimator_polars_matches_pandas(name, ctor, whas500_encoded_small):
        X_pd, X_pl, y = whas500_encoded_small
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est_pd = ctor()
            est_pd.fit(X_pd, y)
            pred_pd = np.asarray(est_pd.predict(X_pd), dtype=float)

            est_pl = ctor()
            est_pl.fit(X_pl, y)
            pred_pl = np.asarray(est_pl.predict(X_pl), dtype=float)

        np.testing.assert_array_equal(pred_pd, pred_pl)


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
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe_pd = Pipeline([("onehot", OneHotEncoder()), ("model", CoxPHSurvivalAnalysis())]).fit(X_pd, y)
            pipe_pl = Pipeline([("onehot", OneHotEncoder()), ("model", CoxPHSurvivalAnalysis())]).fit(X_pl, y)
            pred_pd = pipe_pd.predict(X_pd)
            pred_pl = pipe_pl.predict(X_pl)
        np.testing.assert_array_equal(pred_pd, pred_pl)

    @staticmethod
    def test_cross_val_score_polars_does_not_raise(whas500_pl_pd_small):
        from sklearn.model_selection import KFold, cross_val_score

        from sksurv.linear_model import CoxPHSurvivalAnalysis

        X_pd, X_pl, y = whas500_pl_pd_small
        X_pl_enc = OneHotEncoder().fit_transform(X_pl)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(CoxPHSurvivalAnalysis(), X_pl_enc, y, cv=KFold(3))
        assert scores.shape == (3,)


class TestMetaEstimatorsPolars:
    @staticmethod
    def test_stacking_polars_matches_pandas(whas500_pl_pd_small):
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        from sksurv.meta import Stacking

        X_pd, X_pl, y = whas500_pl_pd_small
        X_pd_enc = OneHotEncoder().fit_transform(X_pd)
        X_pl_enc = OneHotEncoder().fit_transform(X_pl)

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
        np.testing.assert_array_equal(pred_pd, pred_pl)

    @staticmethod
    def test_ensemble_selection_polars_matches_pandas(whas500_pl_pd_small):
        from sklearn.model_selection import KFold

        from sksurv.linear_model import CoxPHSurvivalAnalysis
        from sksurv.meta import EnsembleSelection
        from sksurv.metrics import concordance_index_censored

        X_pd, X_pl, y = whas500_pl_pd_small
        X_pd_enc = OneHotEncoder().fit_transform(X_pd)
        X_pl_enc = OneHotEncoder().fit_transform(X_pl)

        def cindex(est, X, y):
            return concordance_index_censored(y["fstat"], y["lenfol"], est.predict(X))[0]

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
        np.testing.assert_array_equal(pred_pd, pred_pl)


class TestNumpyOnlyAPIsPolarsPassthrough:
    @staticmethod
    def test_concordance_index_censored_with_polars_derived_risk(whas500_pl_pd_small):
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        from sksurv.metrics import concordance_index_censored

        X_pd, X_pl, y = whas500_pl_pd_small
        X_pd_enc = OneHotEncoder().fit_transform(X_pd)
        X_pl_enc = OneHotEncoder().fit_transform(X_pl)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            risk_pd = CoxPHSurvivalAnalysis().fit(X_pd_enc, y).predict(X_pd_enc)
            risk_pl = CoxPHSurvivalAnalysis().fit(X_pl_enc, y).predict(X_pl_enc)
        c_pd = concordance_index_censored(y["fstat"], y["lenfol"], risk_pd)
        c_pl = concordance_index_censored(y["fstat"], y["lenfol"], risk_pl)
        assert c_pd == c_pl

    @staticmethod
    def test_kaplan_meier_estimator_passthrough(whas500_pl_pd_small):
        from sksurv.nonparametric import kaplan_meier_estimator

        _X_pd, _X_pl, y = whas500_pl_pd_small
        t, s = kaplan_meier_estimator(y["fstat"], y["lenfol"])
        assert t.shape == s.shape
        assert len(t) > 0

    @staticmethod
    def test_nelson_aalen_estimator_passthrough(whas500_pl_pd_small):
        from sksurv.nonparametric import nelson_aalen_estimator

        _X_pd, _X_pl, y = whas500_pl_pd_small
        t, ch = nelson_aalen_estimator(y["fstat"], y["lenfol"])
        assert t.shape == ch.shape
        assert len(t) > 0

    @staticmethod
    def test_compare_survival_passthrough(whas500_pl_pd_small):
        from sksurv.compare import compare_survival

        _X_pd, _X_pl, y = whas500_pl_pd_small
        n = len(y)
        group = np.repeat([0, 1, 2], n // 3 + 1)[:n]
        chi2, pval = compare_survival(y, group)
        assert np.isfinite(chi2)
        assert 0.0 <= pval <= 1.0


class TestEstimatorPredictionApiPolarsParity:
    @staticmethod
    def _step_functions_equal(arr_pd, arr_pl):
        assert len(arr_pd) == len(arr_pl)
        for a, b in zip(arr_pd, arr_pl):
            np.testing.assert_array_equal(a.x, b.x)
            np.testing.assert_array_equal(a.y, b.y)

    @staticmethod
    def test_cox_predict_survival_function(whas500_encoded_small):
        from sksurv.linear_model import CoxPHSurvivalAnalysis

        X_pd, X_pl, y = whas500_encoded_small
        est_pd = CoxPHSurvivalAnalysis().fit(X_pd, y)
        est_pl = CoxPHSurvivalAnalysis().fit(X_pl, y)
        sf_pd = est_pd.predict_survival_function(X_pd[:10])
        sf_pl = est_pl.predict_survival_function(X_pl.head(10))
        TestEstimatorPredictionApiPolarsParity._step_functions_equal(sf_pd, sf_pl)

    @staticmethod
    def test_cox_predict_cumulative_hazard_function(whas500_encoded_small):
        from sksurv.linear_model import CoxPHSurvivalAnalysis

        X_pd, X_pl, y = whas500_encoded_small
        est_pd = CoxPHSurvivalAnalysis().fit(X_pd, y)
        est_pl = CoxPHSurvivalAnalysis().fit(X_pl, y)
        chf_pd = est_pd.predict_cumulative_hazard_function(X_pd[:10])
        chf_pl = est_pl.predict_cumulative_hazard_function(X_pl.head(10))
        TestEstimatorPredictionApiPolarsParity._step_functions_equal(chf_pd, chf_pl)

    @staticmethod
    def test_rsf_predict_survival_function(whas500_encoded_small):
        from sksurv.ensemble import RandomSurvivalForest

        X_pd, X_pl, y = whas500_encoded_small
        est_pd = RandomSurvivalForest(n_estimators=5, random_state=0).fit(X_pd, y)
        est_pl = RandomSurvivalForest(n_estimators=5, random_state=0).fit(X_pl, y)
        sf_pd = est_pd.predict_survival_function(X_pd[:10])
        sf_pl = est_pl.predict_survival_function(X_pl.head(10))
        TestEstimatorPredictionApiPolarsParity._step_functions_equal(sf_pd, sf_pl)

    @staticmethod
    def test_survival_tree_predict_survival_function(whas500_encoded_small):
        from sksurv.tree import SurvivalTree

        X_pd, X_pl, y = whas500_encoded_small
        est_pd = SurvivalTree(random_state=0).fit(X_pd, y)
        est_pl = SurvivalTree(random_state=0).fit(X_pl, y)
        sf_pd = est_pd.predict_survival_function(X_pd[:10])
        sf_pl = est_pl.predict_survival_function(X_pl.head(10))
        TestEstimatorPredictionApiPolarsParity._step_functions_equal(sf_pd, sf_pl)

    @staticmethod
    @pytest.mark.parametrize("estimator_name", ["cox", "rsf", "tree"])
    def test_score_method_parity(whas500_encoded_small, estimator_name):
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        from sksurv.tree import SurvivalTree

        ctors = {
            "cox": CoxPHSurvivalAnalysis,
            "rsf": lambda: RandomSurvivalForest(n_estimators=5, random_state=0),
            "tree": lambda: SurvivalTree(random_state=0),
        }
        ctor = ctors[estimator_name]
        X_pd, X_pl, y = whas500_encoded_small
        est_pd = ctor().fit(X_pd, y)
        est_pl = ctor().fit(X_pl, y)
        assert est_pd.score(X_pd, y) == est_pl.score(X_pl, y)

    @staticmethod
    def test_gridsearchcv_polars(whas500_encoded_small):
        from sklearn.model_selection import GridSearchCV

        from sksurv.linear_model import CoxPHSurvivalAnalysis

        X_pd, X_pl, y = whas500_encoded_small
        param_grid = {"alpha": [0.01, 0.1, 1.0]}
        gs_pd = GridSearchCV(CoxPHSurvivalAnalysis(), param_grid, cv=3).fit(X_pd, y)
        gs_pl = GridSearchCV(CoxPHSurvivalAnalysis(), param_grid, cv=3).fit(X_pl, y)
        assert gs_pd.best_params_ == gs_pl.best_params_


class TestAdditionalEstimatorPredictionApis:
    @staticmethod
    def _ctors_with_score():
        from sksurv.ensemble import ExtraSurvivalTrees
        from sksurv.linear_model import CoxnetSurvivalAnalysis, IPCRidge
        from sksurv.svm import (
            FastKernelSurvivalSVM,
            FastSurvivalSVM,
            HingeLossSurvivalSVM,
            MinlipSurvivalAnalysis,
        )

        return [
            ("IPCRidge", lambda: IPCRidge(alpha=1.0)),
            ("CoxnetSurvivalAnalysis", lambda: CoxnetSurvivalAnalysis(n_alphas=5)),
            ("FastSurvivalSVM", lambda: FastSurvivalSVM(random_state=0, max_iter=50)),
            ("FastKernelSurvivalSVM", lambda: FastKernelSurvivalSVM(random_state=0, max_iter=50)),
            ("MinlipSurvivalAnalysis", lambda: MinlipSurvivalAnalysis(solver="ecos")),
            ("HingeLossSurvivalSVM", lambda: HingeLossSurvivalSVM(solver="ecos")),
            ("ExtraSurvivalTrees", lambda: ExtraSurvivalTrees(n_estimators=5, random_state=0)),
        ]

    @staticmethod
    @pytest.mark.parametrize(
        "name,ctor",
        _ctors_with_score(),
        ids=[t[0] for t in _ctors_with_score()],
    )
    def test_score_polars_matches_pandas(name, ctor, whas500_encoded_small):
        import warnings

        X_pd, X_pl, y = whas500_encoded_small
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s_pd = ctor().fit(X_pd, y).score(X_pd, y)
            s_pl = ctor().fit(X_pl, y).score(X_pl, y)
        assert s_pd == s_pl, f"{name}: pandas={s_pd}, polars={s_pl}"

    @staticmethod
    def test_coxnet_predict_survival_function(whas500_encoded_small):
        from sksurv.linear_model import CoxnetSurvivalAnalysis

        X_pd, X_pl, y = whas500_encoded_small
        est_pd = CoxnetSurvivalAnalysis(n_alphas=5, fit_baseline_model=True).fit(X_pd, y)
        est_pl = CoxnetSurvivalAnalysis(n_alphas=5, fit_baseline_model=True).fit(X_pl, y)
        sf_pd = est_pd.predict_survival_function(X_pd[:10])
        sf_pl = est_pl.predict_survival_function(X_pl.head(10))
        TestEstimatorPredictionApiPolarsParity._step_functions_equal(sf_pd, sf_pl)

    @staticmethod
    def test_coxnet_predict_cumulative_hazard_function(whas500_encoded_small):
        from sksurv.linear_model import CoxnetSurvivalAnalysis

        X_pd, X_pl, y = whas500_encoded_small
        est_pd = CoxnetSurvivalAnalysis(n_alphas=5, fit_baseline_model=True).fit(X_pd, y)
        est_pl = CoxnetSurvivalAnalysis(n_alphas=5, fit_baseline_model=True).fit(X_pl, y)
        chf_pd = est_pd.predict_cumulative_hazard_function(X_pd[:10])
        chf_pl = est_pl.predict_cumulative_hazard_function(X_pl.head(10))
        TestEstimatorPredictionApiPolarsParity._step_functions_equal(chf_pd, chf_pl)

    @staticmethod
    def test_extra_survival_trees_predict_survival_function(whas500_encoded_small):
        from sksurv.ensemble import ExtraSurvivalTrees

        X_pd, X_pl, y = whas500_encoded_small
        est_pd = ExtraSurvivalTrees(n_estimators=5, random_state=0).fit(X_pd, y)
        est_pl = ExtraSurvivalTrees(n_estimators=5, random_state=0).fit(X_pl, y)
        sf_pd = est_pd.predict_survival_function(X_pd[:10])
        sf_pl = est_pl.predict_survival_function(X_pl.head(10))
        TestEstimatorPredictionApiPolarsParity._step_functions_equal(sf_pd, sf_pl)

    @staticmethod
    def test_extra_survival_trees_predict_cumulative_hazard_function(whas500_encoded_small):
        from sksurv.ensemble import ExtraSurvivalTrees

        X_pd, X_pl, y = whas500_encoded_small
        est_pd = ExtraSurvivalTrees(n_estimators=5, random_state=0).fit(X_pd, y)
        est_pl = ExtraSurvivalTrees(n_estimators=5, random_state=0).fit(X_pl, y)
        chf_pd = est_pd.predict_cumulative_hazard_function(X_pd[:10])
        chf_pl = est_pl.predict_cumulative_hazard_function(X_pl.head(10))
        TestEstimatorPredictionApiPolarsParity._step_functions_equal(chf_pd, chf_pl)


class TestSurvivalEstimatorLazyFrame:
    ESTIMATORS = TestSurvivalEstimatorPolarsParity.ESTIMATORS

    @staticmethod
    @pytest.mark.parametrize("name,ctor", ESTIMATORS, ids=[t[0] for t in ESTIMATORS])
    def test_estimator_lazyframe_matches_eager_polars(name, ctor, whas500_encoded_small):
        import warnings

        _X_pd, X_pl, y = whas500_encoded_small
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est_eager = ctor()
            est_eager.fit(X_pl, y)
            pred_eager = np.asarray(est_eager.predict(X_pl), dtype=float)

            est_lazy = ctor()
            est_lazy.fit(X_pl.lazy(), y)
            pred_lazy = np.asarray(est_lazy.predict(X_pl.lazy()), dtype=float)

        np.testing.assert_array_equal(pred_eager, pred_lazy)

    @staticmethod
    def test_gb_staged_predict_lazyframe(whas500_encoded_small):
        import warnings

        from sksurv.ensemble import GradientBoostingSurvivalAnalysis

        _X_pd, X_pl, y = whas500_encoded_small
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gb = GradientBoostingSurvivalAnalysis(n_estimators=3, random_state=0).fit(X_pl, y)
            staged_eager = [np.asarray(p, dtype=float) for p in gb.staged_predict(X_pl)]
            staged_lazy = [np.asarray(p, dtype=float) for p in gb.staged_predict(X_pl.lazy())]
        assert len(staged_eager) == len(staged_lazy) == 3
        for a, b in zip(staged_eager, staged_lazy):
            np.testing.assert_array_equal(a, b)
