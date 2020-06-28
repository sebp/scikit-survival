import numpy
from numpy.testing import assert_array_almost_equal
import pytest
from sklearn.pipeline import make_pipeline

from sksurv.datasets import load_breast_cancer
from sksurv.ensemble import RandomSurvivalForest
from sksurv.preprocessing import OneHotEncoder
from sksurv.testing import assert_cindex_almost_equal


def test_fit_predict(make_whas500):
    whas500 = make_whas500(to_numeric=True)

    forest = RandomSurvivalForest(random_state=2)
    forest.fit(whas500.x, whas500.y)

    assert len(forest.estimators_) == 100

    pred = forest.predict(whas500.x)
    assert numpy.isfinite(pred).all()
    assert numpy.all(pred >= 0)

    expected_c = (0.9026201280123488, 67831, 7318, 0, 14)
    assert_cindex_almost_equal(
        whas500.y["fstat"], whas500.y["lenfol"], pred, expected_c)


def test_fit_predict_chf(make_whas500):
    whas500 = make_whas500(to_numeric=True)

    forest = RandomSurvivalForest(n_estimators=10, random_state=2)
    forest.fit(whas500.x, whas500.y)

    assert len(forest.estimators_) == 10

    chf = forest.predict_cumulative_hazard_function(whas500.x, return_array=True)
    assert chf.shape == (500, forest.event_times_.shape[0])

    assert numpy.isfinite(chf).all()
    assert numpy.all(chf >= 0.0)

    vals, counts = numpy.unique(chf[:, 0], return_counts=True)
    assert vals[0] == 0.0
    assert numpy.max(counts) == counts[0]

    d = numpy.apply_along_axis(numpy.diff, 1, chf)
    assert (d >= 0).all()


def test_fit_predict_surv(make_whas500):
    whas500 = make_whas500(to_numeric=True)

    forest = RandomSurvivalForest(n_estimators=10, random_state=2)
    forest.fit(whas500.x, whas500.y)

    assert len(forest.estimators_) == 10

    surv = forest.predict_survival_function(whas500.x, return_array=True)
    assert surv.shape == (500, forest.event_times_.shape[0])

    assert numpy.isfinite(surv).all()
    assert numpy.all(surv >= 0.0)
    assert numpy.all(surv <= 1.0)

    vals, counts = numpy.unique(surv[:, 0], return_counts=True)
    assert vals[-1] == 1.0
    assert numpy.max(counts) == counts[-1]

    d = numpy.apply_along_axis(numpy.diff, 1, surv)
    assert (d <= 0).all()


def test_oob_score(make_whas500):
    whas500 = make_whas500(to_numeric=True)

    forest = RandomSurvivalForest(oob_score=True, bootstrap=False, random_state=2)
    with pytest.raises(ValueError, match="Out of bag estimation only available "
                                         "if bootstrap=True"):
        forest.fit(whas500.x, whas500.y)

    forest.set_params(bootstrap=True)
    forest.fit(whas500.x, whas500.y)

    assert forest.oob_prediction_.shape == (whas500.x.shape[0],)
    assert round(abs(forest.oob_score_ - 0.753010685), 6) == 0.0


@pytest.mark.parametrize("func", ("predict_survival_function", "predict_cumulative_hazard_function"))
def test_predict_step_function(make_whas500, func):
    whas500 = make_whas500(to_numeric=True)

    forest = RandomSurvivalForest(n_estimators=10, random_state=2)
    forest.fit(whas500.x[10:], whas500.y[10:])

    pred_fn = getattr(forest, func)

    ret_array = pred_fn(whas500.x[:10], return_array=True)
    fn_array = pred_fn(whas500.x[:10], return_array=False)

    assert ret_array.shape[0] == fn_array.shape[0]

    for fn, arr in zip(fn_array, ret_array):
        assert_array_almost_equal(fn.x, forest.event_times_)
        assert_array_almost_equal(fn.y, arr)


@pytest.mark.parametrize("func", ("predict_survival_function", "predict_cumulative_hazard_function"))
def test_predict_step_function_warning(make_whas500, func):
    whas500 = make_whas500(to_numeric=True)

    forest = RandomSurvivalForest(n_estimators=3, oob_score=True, random_state=2)
    forest.fit(whas500.x, whas500.y)

    pred_fn = getattr(forest, func)

    with pytest.warns(FutureWarning,
                      match="{} will return an array of StepFunction instances in 0.14".format(func)):
        pred_fn(whas500.x)


def test_oob_too_little_estimators(make_whas500):
    whas500 = make_whas500(to_numeric=True)

    forest = RandomSurvivalForest(n_estimators=3, oob_score=True, random_state=2)
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


def test_fit_warm_start(make_whas500):
    whas500 = make_whas500(to_numeric=True)

    forest = RandomSurvivalForest(n_estimators=11, max_depth=2, random_state=2)
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


def test_fit_with_small_max_samples(make_whas500):
    whas500 = make_whas500(to_numeric=True)

    # First fit with no restriction on max samples
    est1 = RandomSurvivalForest(
        n_estimators=1,
        random_state=1,
        max_samples=None,
    )

    # Second fit with max samples restricted to just 2
    est2 = RandomSurvivalForest(
        n_estimators=1,
        random_state=1,
        max_samples=2,
    )

    est1.fit(whas500.x, whas500.y)
    est2.fit(whas500.x, whas500.y)

    tree1 = est1.estimators_[0].tree_
    tree2 = est2.estimators_[0].tree_

    msg = "Tree without `max_samples` restriction should have more nodes"
    assert tree1.node_count > tree2.node_count, msg


@pytest.mark.parametrize("func", ("predict_survival_function", "predict_cumulative_hazard_function"))
def test_pipeline_predict(breast_cancer, func):
    X_str, _ = load_breast_cancer()
    X_num, y = breast_cancer

    est = RandomSurvivalForest(n_estimators=10, random_state=1)
    est.fit(X_num[10:], y[10:])

    pipe = make_pipeline(OneHotEncoder(), RandomSurvivalForest(n_estimators=10, random_state=1))
    pipe.fit(X_str[10:], y[10:])

    tree_pred = getattr(est, func)(X_num[:10])
    pipe_pred = getattr(pipe, func)(X_str[:10])

    assert_array_almost_equal(tree_pred, pipe_pred)


@pytest.mark.parametrize(
    'max_samples, exc_type, exc_msg',
    [(int(1e9), ValueError,
      "`max_samples` must be in range 1 to 500 but got value 1000000000"),
     (1.0, ValueError,
      r"`max_samples` must be in range \(0, 1\) but got value 1.0"),
     (2.0, ValueError,
      r"`max_samples` must be in range \(0, 1\) but got value 2.0"),
     (0.0, ValueError,
      r"`max_samples` must be in range \(0, 1\) but got value 0.0"),
     (numpy.nan, ValueError,
      r"`max_samples` must be in range \(0, 1\) but got value nan"),
     (numpy.inf, ValueError,
      r"`max_samples` must be in range \(0, 1\) but got value inf"),
     ('str max_samples?!', TypeError,
      r"`max_samples` should be int or float, but got "
      r"type '\<class 'str'\>'"),
     (numpy.ones(2), TypeError,
      r"`max_samples` should be int or float, but got type "
      r"'\<class 'numpy.ndarray'\>'")]
)
def test_fit_max_samples(make_whas500, max_samples, exc_type, exc_msg):
    whas500 = make_whas500(to_numeric=True)
    forest = RandomSurvivalForest(max_samples=max_samples)
    with pytest.raises(exc_type, match=exc_msg):
        forest.fit(whas500.x, whas500.y)
