import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest
from scipy import sparse
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from sksurv.datasets import load_breast_cancer
from sksurv.ensemble import ExtraSurvivalTrees, RandomSurvivalForest
from sksurv.preprocessing import OneHotEncoder
from sksurv.testing import assert_chf_properties, assert_cindex_almost_equal, assert_survival_function_properties
from sksurv.tree import SurvivalTree

FORESTS = [
    RandomSurvivalForest,
    ExtraSurvivalTrees,
]


@pytest.mark.parametrize(
    "forest_cls, expected_c",
    [
        (RandomSurvivalForest, (0.9009168452008676, 67703, 7446, 0, 14)),
        (ExtraSurvivalTrees, (0.8400644053813091, 63130, 12019, 0, 14)),
    ],
)
def test_fit_predict(make_whas500, forest_cls, expected_c):
    whas500 = make_whas500(to_numeric=True)

    forest = forest_cls(random_state=2)
    forest.fit(whas500.x, whas500.y)

    assert len(forest.estimators_) == 100

    pred = forest.predict(whas500.x)
    assert np.isfinite(pred).all()
    assert np.all(pred >= 0)

    assert_cindex_almost_equal(whas500.y["fstat"], whas500.y["lenfol"], pred, expected_c)


@pytest.mark.parametrize(
    "forest_cls,expected_cindex", [(ExtraSurvivalTrees, 0.7486232588273405), (RandomSurvivalForest, 0.7444120505344995)]
)
def test_fit_missing_values(make_whas500, forest_cls, expected_cindex):
    whas500 = make_whas500(to_numeric=True)

    rng = np.random.RandomState(42)
    mask = rng.binomial(n=1, p=0.15, size=whas500.x.shape)
    mask = mask.astype(bool)
    X = whas500.x.copy()
    X[mask] = np.nan

    X_train, y_train = X[:400], whas500.y[:400]
    X_test, y_test = X[400:], whas500.y[400:]

    forest = forest_cls(random_state=42)
    forest.fit(X_train, y_train)

    tags = forest.__sklearn_tags__()
    assert tags.input_tags.allow_nan

    cindex = forest.score(X_test, y_test)
    assert cindex == pytest.approx(expected_cindex)


@pytest.mark.parametrize("forst_cls", [ExtraTreesClassifier, RandomForestClassifier])
def test_sklearn_random_forest_tags(forst_cls):
    est = forst_cls()

    # https://scikit-learn.org/stable/developers/develop.html#estimator-tags
    tags = est.__sklearn_tags__()
    assert tags.target_tags.multi_output
    assert tags.requires_fit
    assert tags.target_tags.required
    assert tags.input_tags.allow_nan


@pytest.mark.parametrize("forest_cls", FORESTS)
def test_fit_int_time(make_whas500, forest_cls):
    whas500 = make_whas500(to_numeric=True)
    y = whas500.y
    y_int = np.empty(y.shape[0], dtype=[(y.dtype.names[0], bool), (y.dtype.names[1], int)])
    y_int[:] = y

    forest_f = forest_cls(oob_score=True, random_state=2).fit(whas500.x[50:], y[50:])
    forest_i = forest_cls(oob_score=True, random_state=2).fit(whas500.x[50:], y_int[50:])

    assert len(forest_f.estimators_) == len(forest_i.estimators_)
    assert forest_f.n_features_in_ == forest_i.n_features_in_
    assert forest_f.oob_score_ == forest_i.oob_score_
    assert_array_almost_equal(forest_f.unique_times_, forest_i.unique_times_)

    pred_f = forest_f.predict(whas500.x[:50])
    pred_i = forest_i.predict(whas500.x[:50])

    assert_array_almost_equal(pred_f, pred_i)


@pytest.mark.parametrize("forest_cls", FORESTS)
def test_fit_predict_chf(make_whas500, forest_cls):
    whas500 = make_whas500(to_numeric=True)

    forest = forest_cls(n_estimators=10, random_state=2)
    forest.fit(whas500.x, whas500.y)

    assert len(forest.estimators_) == 10

    chf = forest.predict_cumulative_hazard_function(whas500.x, return_array=True)
    assert chf.shape == (500, forest.unique_times_.shape[0])

    assert_chf_properties(chf)


@pytest.mark.parametrize("forest_cls", FORESTS)
def test_fit_predict_surv(make_whas500, forest_cls):
    whas500 = make_whas500(to_numeric=True)

    forest = forest_cls(n_estimators=10, random_state=2)
    forest.fit(whas500.x, whas500.y)

    assert len(forest.estimators_) == 10

    surv = forest.predict_survival_function(whas500.x, return_array=True)
    assert surv.shape == (500, forest.unique_times_.shape[0])

    assert_survival_function_properties(surv)


@pytest.mark.parametrize(
    "forest_cls, expected_oob_score", [(RandomSurvivalForest, 0.758732651), (ExtraSurvivalTrees, 0.751427165)]
)
def test_oob_score(make_whas500, forest_cls, expected_oob_score):
    whas500 = make_whas500(to_numeric=True)

    forest = forest_cls(oob_score=True, bootstrap=False, random_state=2)
    with pytest.raises(ValueError, match="Out of bag estimation only available if bootstrap=True"):
        forest.fit(whas500.x, whas500.y)

    forest.set_params(bootstrap=True)
    forest.fit(whas500.x, whas500.y)

    assert forest.oob_prediction_.shape == (whas500.x.shape[0],)
    assert forest.oob_score_ == pytest.approx(expected_oob_score)


@pytest.mark.parametrize("forest_cls", FORESTS)
@pytest.mark.parametrize("func", ["predict_survival_function", "predict_cumulative_hazard_function"])
def test_predict_step_function(make_whas500, forest_cls, func):
    whas500 = make_whas500(to_numeric=True)

    forest = forest_cls(n_estimators=10, random_state=2)
    forest.fit(whas500.x[10:], whas500.y[10:])

    pred_fn = getattr(forest, func)

    ret_array = pred_fn(whas500.x[:10], return_array=True)
    fn_array = pred_fn(whas500.x[:10], return_array=False)

    assert ret_array.shape[0] == fn_array.shape[0]

    for fn, arr in zip(fn_array, ret_array):
        assert_array_almost_equal(fn.x, forest.unique_times_)
        assert_array_almost_equal(fn.y, arr)


@pytest.mark.parametrize("forest_cls", FORESTS)
def test_oob_too_little_estimators(make_whas500, forest_cls):
    whas500 = make_whas500(to_numeric=True)

    forest = forest_cls(n_estimators=3, oob_score=True, random_state=2)
    with pytest.warns(
        UserWarning,
        match="Some inputs do not have OOB scores. "
        "This probably means too few trees were used "
        "to compute any reliable oob estimates.",
    ):
        forest.fit(whas500.x, whas500.y)


def test_fit_no_bootstrap(make_whas500):
    whas500 = make_whas500(to_numeric=True)

    forest = RandomSurvivalForest(n_estimators=10, bootstrap=False, random_state=2)
    forest.fit(whas500.x, whas500.y)

    pred = forest.predict(whas500.x)

    expected_c = (0.931881994437717, 70030, 5119, 0, 14)
    assert_cindex_almost_equal(whas500.y["fstat"], whas500.y["lenfol"], pred, expected_c)


@pytest.mark.parametrize("forest_cls", FORESTS)
def test_fit_warm_start(make_whas500, forest_cls):
    whas500 = make_whas500(to_numeric=True)

    forest = forest_cls(n_estimators=11, max_depth=2, random_state=2)
    forest.fit(whas500.x, whas500.y)

    assert len(forest.estimators_) == 11
    assert all(e.max_depth == 2 for e in forest.estimators_)

    forest.set_params(warm_start=True)
    with pytest.warns(UserWarning, match="Warm-start fitting without increasing n_estimators does not fit new trees."):
        forest.fit(whas500.x, whas500.y)

    forest.set_params(n_estimators=3)
    with pytest.raises(
        ValueError, match=r"n_estimators=3 must be larger or equal to len\(estimators_\)=11 when warm_start==True"
    ):
        forest.fit(whas500.x, whas500.y)

    forest.set_params(n_estimators=23)
    forest.fit(whas500.x, whas500.y)

    assert len(forest.estimators_) == 23
    assert all(e.max_depth == 2 for e in forest.estimators_)


@pytest.mark.parametrize("forest_cls", FORESTS)
def test_fit_with_small_max_samples(make_whas500, forest_cls):
    whas500 = make_whas500(to_numeric=True)

    # First fit with no restriction on max samples
    est1 = forest_cls(n_estimators=1, random_state=1, max_samples=None)

    # Second fit with max samples restricted to just 2
    est2 = forest_cls(n_estimators=1, random_state=1, max_samples=2)

    est1.fit(whas500.x, whas500.y)
    est2.fit(whas500.x, whas500.y)

    tree1 = est1.estimators_[0].tree_
    tree2 = est2.estimators_[0].tree_

    msg = "Tree without `max_samples` restriction should have more nodes"
    assert tree1.node_count > tree2.node_count, msg


@pytest.mark.parametrize("forest_cls", FORESTS)
def test_max_samples_without_bootstrap(make_whas500, forest_cls):
    whas500 = make_whas500(to_numeric=True)

    est = forest_cls(n_estimators=1, random_state=1, bootstrap=False, max_samples=10)
    msg = (
        r"`max_sample` cannot be set if `bootstrap=False`\. "
        r"Either switch to `bootstrap=True` or set `max_sample=None`\."
    )
    with pytest.raises(ValueError, match=msg):
        est.fit(whas500.x, whas500.y)


@pytest.mark.parametrize("forest_cls", FORESTS)
def test_estimators_samples(make_whas500, forest_cls):
    whas500 = make_whas500(to_numeric=True)

    est = forest_cls(n_estimators=10, max_samples=333, random_state=1, low_memory=True)
    est.fit(whas500.x, whas500.y)

    n_samples = [len(np.unique(arr)) for arr in est.estimators_samples_]
    expected = np.array([255, 227, 245, 247, 246, 239, 254, 252, 245, 248])
    assert_array_equal(n_samples, expected)


@pytest.mark.parametrize("forest_cls", FORESTS)
@pytest.mark.parametrize("func", ["predict_survival_function", "predict_cumulative_hazard_function"])
def test_pipeline_predict(breast_cancer, forest_cls, func):
    X_str, _ = load_breast_cancer()
    X_num, y = breast_cancer

    est = forest_cls(n_estimators=10, random_state=1)
    est.fit(X_num[10:], y[10:])

    pipe = make_pipeline(OneHotEncoder(), forest_cls(n_estimators=10, random_state=1))
    pipe.fit(X_str[10:], y[10:])

    tree_pred = getattr(est, func)(X_num[:10], return_array=True)
    pipe_pred = getattr(pipe, func)(X_str[:10], return_array=True)

    assert_array_almost_equal(tree_pred, pipe_pred)


@pytest.mark.parametrize("forest_cls", FORESTS)
@pytest.mark.parametrize(
    "max_samples, exc_type, exc_msg, with_prefix",
    [
        (int(1e9), ValueError, "`max_samples` must be <= n_samples=500 but got value 1000000000", False),
        (1.0 + 1e-7, ValueError, r"Got 1\.0000001 instead", True),
        (2.0, ValueError, r"Got 2\.0 instead", True),
        (0.0, ValueError, r"Got 0\.0 instead", True),
        (np.nan, ValueError, "Got nan instead", True),
        (np.inf, ValueError, r"Got inf instead", True),
        ("str max_samples?!", TypeError, r"Got 'str max_samples\?!' instead", True),
        (np.ones(2), TypeError, r"Got array\(\[1\., 1\.\]\) instead", True),
        (0, ValueError, r"Got 0 instead", True),
    ],
)
def test_fit_max_samples(make_whas500, forest_cls, max_samples, exc_type, exc_msg, with_prefix):
    whas500 = make_whas500(to_numeric=True)
    forest = forest_cls(max_samples=max_samples)
    prefix = (
        f"The 'max_samples' parameter of {forest_cls.__name__} must be None, "
        r"a float in the range \(0\.0, 1\.0] or an int in the range \[1, inf\)\. "
    )
    if with_prefix:
        msg = prefix + exc_msg
    else:
        msg = exc_msg
    with pytest.raises(exc_type, match=msg):
        forest.fit(whas500.x, whas500.y)


@pytest.mark.parametrize("forest_cls", FORESTS)
@pytest.mark.parametrize("max_features", [0, 0.0, 3.0, "", "None", "sqrt_", "log10", "car"])
def test_fit_max_features(make_whas500, forest_cls, max_features):
    whas500 = make_whas500(to_numeric=True)
    forest = forest_cls(max_features=max_features)

    msg = (
        f"The 'max_features' parameter of {forest_cls.__name__} must be "
        r"an int in the range \[1, inf\), a float in the range \(0\.0, 1\.0\], "
        r"a str among {.+} or None\."
    )
    with pytest.raises(ValueError, match=msg):
        forest.fit(whas500.x, whas500.y)


@pytest.mark.parametrize("forest_cls", FORESTS)
def test_apply(make_whas500, forest_cls):
    whas500 = make_whas500(to_numeric=True)

    forest = forest_cls()
    forest.fit(whas500.x, whas500.y)

    x_trans = forest.apply(whas500.x)

    assert x_trans.shape[0] == whas500.x.shape[0]
    assert x_trans.shape[1] == forest.n_estimators

    x_path, _ = forest.decision_path(whas500.x)

    assert x_path.toarray().shape[0] == whas500.x.shape[0]


@pytest.mark.parametrize("forest_cls", FORESTS)
def test_apply_sparse(make_whas500, forest_cls):
    whas500 = make_whas500(to_numeric=True)

    forest = forest_cls()
    X, y = whas500.x, whas500.y
    X_csr = sparse.csr_matrix(X)
    forest.fit(X_csr, y)

    X_trans = forest.apply(X_csr)

    assert X_trans.shape[0] == X.shape[0]
    assert X_trans.shape[1] == forest.n_estimators

    X_path, _ = forest.decision_path(X_csr)

    assert X_path.toarray().shape[0] == X.shape[0]


@pytest.mark.parametrize("forest_cls", FORESTS)
def test_predict_sparse(make_whas500, forest_cls):
    seed = 42
    whas500 = make_whas500(to_numeric=True)
    X, y = whas500.x, whas500.y
    X = np.random.RandomState(seed).binomial(n=5, p=0.1, size=X.shape)

    X_train, X_test, y_train, _ = train_test_split(X, y, random_state=seed)

    forest = forest_cls(random_state=seed)
    forest.fit(X_train, y_train)
    y_pred = forest.predict(X_test)
    y_cum_h = forest.predict_cumulative_hazard_function(X_test)
    y_surv = forest.predict_survival_function(X_test)

    X_train_csr = sparse.csr_matrix(X_train)
    X_test_csr = sparse.csr_matrix(X_test)

    forest_csr = forest_cls(random_state=seed)
    forest_csr.fit(X_train_csr, y_train)
    y_pred_csr = forest_csr.predict(X_test_csr)
    y_cum_h_csr = forest_csr.predict_cumulative_hazard_function(X_test_csr)
    y_surv_csr = forest_csr.predict_survival_function(X_test_csr)

    assert y_pred.shape[0] == X_test.shape[0]
    assert y_pred_csr.shape[0] == X_test.shape[0]

    assert_array_equal(y_pred, y_pred_csr)
    assert_array_equal(y_cum_h_csr, y_cum_h)
    assert_array_equal(y_surv, y_surv_csr)


@pytest.mark.parametrize(
    "est_cls,params",
    [
        (SurvivalTree, {"min_samples_leaf": 10, "random_state": 42}),
        (RandomSurvivalForest, {"n_estimators": 10, "min_samples_leaf": 10, "random_state": 42}),
        (ExtraSurvivalTrees, {"n_estimators": 10, "min_samples_leaf": 10, "random_state": 42}),
    ],
)
def test_predict_low_memory(make_whas500, est_cls, params):
    whas500 = make_whas500(to_numeric=True)
    X, y = whas500.x, whas500.y

    X_train, X_test, y_train, _ = train_test_split(X, y, random_state=params["random_state"])

    est_high = est_cls(**params)
    est_high.set_params(low_memory=False)
    est_high.fit(X_train, y_train)
    pred_high = est_high.predict(X_test)

    est_low = est_cls(**params)
    est_low.set_params(low_memory=True)
    est_low.fit(X_train, y_train)
    pred_low = est_low.predict(X_test)

    assert pred_high.shape[0] == X_test.shape[0]
    assert pred_low.shape[0] == X_test.shape[0]

    assert_array_almost_equal(pred_high, pred_low)

    msg = (
        "predict_cumulative_hazard_function is not implemented in low memory mode."
        " run fit with low_memory=False to disable low memory mode."
    )
    with pytest.raises(NotImplementedError, match=msg):
        est_low.predict_cumulative_hazard_function(X_test)

    msg = (
        "predict_survival_function is not implemented in low memory mode."
        " run fit with low_memory=False to disable low memory mode."
    )
    with pytest.raises(NotImplementedError, match=msg):
        est_low.predict_survival_function(X_test)
