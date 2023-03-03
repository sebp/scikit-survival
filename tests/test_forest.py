import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from sksurv.datasets import load_breast_cancer
from sksurv.ensemble import ExtraSurvivalTrees, RandomSurvivalForest
from sksurv.preprocessing import OneHotEncoder
from sksurv.testing import assert_cindex_almost_equal

FORESTS = [
    RandomSurvivalForest,
    ExtraSurvivalTrees,
]


@pytest.mark.parametrize(
    'forest_cls, expected_c',
    [(RandomSurvivalForest, (0.9026201280123488, 67831, 7318, 0, 14)),
     (ExtraSurvivalTrees, (0.8389200122423452, 63044, 12105, 0, 14))]
)
def test_fit_predict(make_whas500, forest_cls, expected_c):
    whas500 = make_whas500(to_numeric=True)

    forest = forest_cls(random_state=2)
    forest.fit(whas500.x, whas500.y)

    assert len(forest.estimators_) == 100

    pred = forest.predict(whas500.x)
    assert np.isfinite(pred).all()
    assert np.all(pred >= 0)

    assert_cindex_almost_equal(
        whas500.y["fstat"], whas500.y["lenfol"], pred, expected_c)


@pytest.mark.parametrize('forest_cls', FORESTS)
def test_fit_int_time(make_whas500, forest_cls):
    whas500 = make_whas500(to_numeric=True)
    y = whas500.y
    y_int = np.empty(
        y.shape[0], dtype=[(y.dtype.names[0], bool), (y.dtype.names[1], int)]
    )
    y_int[:] = y

    forest_f = forest_cls(oob_score=True, random_state=2).fit(whas500.x[50:], y[50:])
    forest_i = forest_cls(oob_score=True, random_state=2).fit(whas500.x[50:], y_int[50:])

    assert len(forest_f.estimators_) == len(forest_i.estimators_)
    assert forest_f.n_features_in_ == forest_i.n_features_in_
    assert forest_f.oob_score_ == forest_i.oob_score_
    assert_array_almost_equal(forest_f.event_times_, forest_i.event_times_)

    pred_f = forest_f.predict(whas500.x[:50])
    pred_i = forest_i.predict(whas500.x[:50])

    assert_array_almost_equal(pred_f, pred_i)


@pytest.mark.parametrize('forest_cls', FORESTS)
def test_fit_predict_chf(make_whas500, forest_cls):
    whas500 = make_whas500(to_numeric=True)

    forest = forest_cls(n_estimators=10, random_state=2)
    forest.fit(whas500.x, whas500.y)

    assert len(forest.estimators_) == 10

    chf = forest.predict_cumulative_hazard_function(whas500.x, return_array=True)
    assert chf.shape == (500, forest.event_times_.shape[0])

    assert np.isfinite(chf).all()
    assert np.all(chf >= 0.0)

    vals, counts = np.unique(chf[:, 0], return_counts=True)
    assert vals[0] == 0.0
    assert np.max(counts) == counts[0]

    d = np.apply_along_axis(np.diff, 1, chf)
    assert (d >= 0).all()


@pytest.mark.parametrize('forest_cls', FORESTS)
def test_fit_predict_surv(make_whas500, forest_cls):
    whas500 = make_whas500(to_numeric=True)

    forest = forest_cls(n_estimators=10, random_state=2)
    forest.fit(whas500.x, whas500.y)

    assert len(forest.estimators_) == 10

    surv = forest.predict_survival_function(whas500.x, return_array=True)
    assert surv.shape == (500, forest.event_times_.shape[0])

    assert np.isfinite(surv).all()
    assert np.all(surv >= 0.0)
    assert np.all(surv <= 1.0)

    vals, counts = np.unique(surv[:, 0], return_counts=True)
    assert vals[-1] == 1.0
    assert np.max(counts) == counts[-1]

    d = np.apply_along_axis(np.diff, 1, surv)
    assert (d <= 0).all()


@pytest.mark.parametrize(
    'forest_cls, expected_oob_score',
    [(RandomSurvivalForest, 0.753010685),
     (ExtraSurvivalTrees, 0.752092510)]
)
def test_oob_score(make_whas500, forest_cls, expected_oob_score):
    whas500 = make_whas500(to_numeric=True)

    forest = forest_cls(oob_score=True, bootstrap=False, random_state=2)
    with pytest.raises(ValueError, match="Out of bag estimation only available "
                                         "if bootstrap=True"):
        forest.fit(whas500.x, whas500.y)

    forest.set_params(bootstrap=True)
    forest.fit(whas500.x, whas500.y)

    assert forest.oob_prediction_.shape == (whas500.x.shape[0],)
    assert round(abs(forest.oob_score_ - expected_oob_score), 6) == 0.0


@pytest.mark.parametrize('forest_cls', FORESTS)
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
        assert_array_almost_equal(fn.x, forest.event_times_)
        assert_array_almost_equal(fn.y, arr)


@pytest.mark.parametrize('forest_cls', FORESTS)
def test_oob_too_little_estimators(make_whas500, forest_cls):
    whas500 = make_whas500(to_numeric=True)

    forest = forest_cls(n_estimators=3, oob_score=True, random_state=2)
    with pytest.warns(UserWarning, match="Some inputs do not have OOB scores. "
                                         "This probably means too few trees were used "
                                         "to compute any reliable oob estimates."):
        forest.fit(whas500.x, whas500.y)


def test_fit_no_bootstrap(make_whas500):
    whas500 = make_whas500(to_numeric=True)

    forest = RandomSurvivalForest(n_estimators=10, bootstrap=False, random_state=2)
    forest.fit(whas500.x, whas500.y)

    pred = forest.predict(whas500.x)

    expected_c = (0.931881994437717, 70030, 5119, 0, 14)
    assert_cindex_almost_equal(
        whas500.y["fstat"], whas500.y["lenfol"], pred, expected_c)


@pytest.mark.parametrize('forest_cls', FORESTS)
def test_fit_warm_start(make_whas500, forest_cls):
    whas500 = make_whas500(to_numeric=True)

    forest = forest_cls(n_estimators=11, max_depth=2, random_state=2)
    forest.fit(whas500.x, whas500.y)

    assert len(forest.estimators_) == 11
    assert all((e.max_depth == 2 for e in forest.estimators_))

    forest.set_params(warm_start=True)
    with pytest.warns(UserWarning, match="Warm-start fitting without increasing "
                                         "n_estimators does not fit new trees."):
        forest.fit(whas500.x, whas500.y)

    forest.set_params(n_estimators=3)
    with pytest.raises(ValueError, match="n_estimators=3 must be larger or equal to "
                                         r"len\(estimators_\)=11 when warm_start==True"):
        forest.fit(whas500.x, whas500.y)

    forest.set_params(n_estimators=23)
    forest.fit(whas500.x, whas500.y)

    assert len(forest.estimators_) == 23
    assert all((e.max_depth == 2 for e in forest.estimators_))


@pytest.mark.parametrize('forest_cls', FORESTS)
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


@pytest.mark.parametrize('forest_cls', FORESTS)
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


@pytest.mark.parametrize('forest_cls', FORESTS)
@pytest.mark.parametrize(
    'max_samples, exc_type, exc_msg, with_prefix',
    [(int(1e9), ValueError,
      "`max_samples` must be <= n_samples=500 but got value 1000000000", False),
     (1.0 + 1e-7, ValueError,
      r"Got 1\.0000001 instead", True),
     (2.0, ValueError,
      r"Got 2\.0 instead", True),
     (0.0, ValueError,
      r"Got 0\.0 instead", True),
     (np.nan, ValueError,
      "Got nan instead", True),
     (np.inf, ValueError,
      r"Got inf instead", True),
     ('str max_samples?!', TypeError,
      r"Got 'str max_samples\?!' instead", True),
     (np.ones(2), TypeError,
      r"Got array\(\[1\., 1\.\]\) instead", True),
     (0, ValueError, r"Got 0 instead", True)]
)
def test_fit_max_samples(make_whas500, forest_cls, max_samples, exc_type, exc_msg, with_prefix):
    whas500 = make_whas500(to_numeric=True)
    forest = forest_cls(max_samples=max_samples)
    prefix = f"The 'max_samples' parameter of {forest_cls.__name__} must be None, " \
             r"a float in the range \(0\.0, 1\.0] or an int in the range \[1, inf\)\. "
    if with_prefix:
        msg = prefix + exc_msg
    else:
        msg = exc_msg
    with pytest.raises(exc_type, match=msg):
        forest.fit(whas500.x, whas500.y)


@pytest.mark.parametrize('forest_cls', FORESTS)
@pytest.mark.parametrize('max_features', [0, 0.0, 3.0, "", "None", "sqrt_", "log10", "car"])
def test_fit_max_features(make_whas500, forest_cls, max_features):
    whas500 = make_whas500(to_numeric=True)
    forest = forest_cls(max_features=max_features)

    msg = f"The 'max_features' parameter of {forest_cls.__name__} must be " \
          r"an int in the range \[1, inf\), a float in the range \(0\.0, 1\.0\], " \
          r"a str among {.+} or None\."
    with pytest.raises(ValueError, match=msg):
        forest.fit(whas500.x, whas500.y)


@pytest.mark.parametrize('forest_cls', FORESTS)
def test_apply(make_whas500, forest_cls):
    whas500 = make_whas500(to_numeric=True)

    forest = forest_cls()
    forest.fit(whas500.x, whas500.y)

    x_trans = forest.apply(whas500.x)

    assert x_trans.shape[0] == whas500.x.shape[0]
    assert x_trans.shape[1] == forest.n_estimators

    x_path, _ = forest.decision_path(whas500.x)

    assert x_path.toarray().shape[0] == whas500.x.shape[0]


@pytest.mark.parametrize('forest_cls', FORESTS)
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


@pytest.mark.parametrize('forest_cls', FORESTS)
def test_predict_sparse(make_whas500, forest_cls):
    seed = 42
    whas500 = make_whas500(to_numeric=True)
    X, y = whas500.x, whas500.y
    X = np.random.RandomState(seed).binomial(n=5, p=.1, size=X.shape)

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
