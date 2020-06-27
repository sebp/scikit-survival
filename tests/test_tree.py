from queue import LifoQueue
import numpy
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pandas
import pytest
from sklearn.tree._tree import TREE_UNDEFINED
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder

from sksurv.compare import compare_survival
from sksurv.datasets import load_veterans_lung_cancer, load_breast_cancer
from sksurv.nonparametric import kaplan_meier_estimator, nelson_aalen_estimator
from sksurv.tree import SurvivalTree


@pytest.fixture()
def veterans():
    return load_veterans_lung_cancer()


@pytest.fixture()
def breast_cancer():
    X, y = load_breast_cancer()
    X.loc[:, "er"] = X.loc[:, "er"].replace({"negative": 0, "positive": 1})
    X.loc[:, "grade"] = X.loc[:, "grade"].replace(
        {"intermediate": 0,
         "poorly differentiated": 1,
         "unkown": 2,
         "well differentiated": 3}
    )
    return X, y


@pytest.fixture()
def toy_data():
    rnd = numpy.random.RandomState(1)
    n_samples = 500
    X = numpy.empty((n_samples, 4), dtype=float)
    X[:, :2] = rnd.normal(scale=2, size=(n_samples, 2))
    X[:, 2] = rnd.uniform(40, 80, size=n_samples)
    X[:, 3] = rnd.binomial(1, 0.4, size=n_samples)

    time = numpy.zeros(n_samples, dtype=float)
    groups = [
        (X[:, 0] < 0.15) & (X[:, 2] > 66),
        (X[:, 0] < 0.15) & (X[:, 2] <= 66),
        (X[:, 0] >= 0.15) & (X[:, 0] < 0.65) & (X[:, 1] >= 0.5),
        (X[:, 0] >= 0.15) & (X[:, 0] < 0.65) & (X[:, 1] < 0.5),
        (X[:, 0] >= 0.65) & (X[:, 3] == 1),
        (X[:, 0] >= 0.65) & (X[:, 3] == 0),
    ]
    scales = [3, 1, 8, 5, 9, 7]
    for g, s in zip(groups, scales):
        assert g.sum() > 0
        time[g] = 1 + rnd.lognormal(mean=s, sigma=3, size=g.sum()).astype(int)

    event = numpy.ones(n_samples, dtype=numpy.bool)
    event[rnd.binomial(1, 0.333, size=n_samples).astype(bool)] = False

    y = numpy.fromiter(zip(event, time),
                       dtype=[("status", numpy.bool), ("time", numpy.float)])
    return X, y


def assert_curve_almost_equal(x, y):
    jumps_x = numpy.diff(x) != 0
    jumps_y = numpy.diff(y) != 0

    assert_array_almost_equal(x[:1], y[:1])
    assert_array_almost_equal(x[1:][jumps_x], y[1:][jumps_y])


class LogrankTreeBuilder:

    def __init__(self, max_depth=4, min_leaf=20):
        self.max_depth = max_depth
        self.min_leaf = min_leaf

    def build(self, X, y):
        val, feat, stat = self._get_best_split(X, y)
        splits = LifoQueue()
        splits.put((val, feat, stat, 0, numpy.arange(X.shape[0])))

        node_stats = []
        while splits.qsize() > 0:
            val, feat, stat, lvl, idx = splits.get()
            s = {"feature": feat,
                 "threshold": val,
                 "n_node_samples": idx.shape[0],
                 "statistic": stat,
                 "depth": lvl}
            node_stats.append(s)

            if val == TREE_UNDEFINED:
                continue

            left = X[idx, feat] <= val
            right = idx[~left]
            left = idx[left]

            if lvl == self.max_depth - 1:
                splits.put([TREE_UNDEFINED, TREE_UNDEFINED,
                            -numpy.infty, lvl + 1, right])
                splits.put([TREE_UNDEFINED, TREE_UNDEFINED,
                            -numpy.infty, lvl + 1, left])
                continue

            X_right = X[right, :]
            y_right = y[right]
            s_right = self._get_best_split(X_right, y_right)
            splits.put(list(s_right) + [lvl + 1, right])

            X_left = X[left, :]
            y_left = y[left]
            s_left = self._get_best_split(X_left, y_left)
            splits.put(list(s_left) + [lvl + 1, left])

        return pandas.DataFrame.from_dict(
            dict(zip(range(len(node_stats)), node_stats)),
            orient="index")

    def _get_best_split(self, X, y):
        min_leaf = self.min_leaf
        best_val = TREE_UNDEFINED
        best_feat = TREE_UNDEFINED
        best_stat = -numpy.infty

        if y[y.dtype.names[0]].sum() == 0:
            return best_val, best_feat, best_stat
        for j in range(X.shape[1]):
            vals = X[:, j]
            values = numpy.unique(vals)
            if len(values) < 2:
                continue

            for i, v in enumerate(values[:-1]):
                t = (v + values[i + 1]) * 0.5
                groups = (vals <= t).astype(int)
                if groups.sum() >= min_leaf and (X.shape[0] - groups.sum()) >= min_leaf:
                    s, _ = compare_survival(y, groups)
                    if s > best_stat:
                        best_feat = j
                        best_val = t
                        best_stat = s
        return best_val, best_feat, best_stat


def test_tree_one_split(veterans):
    X, y = veterans
    X = X.loc[:, "Karnofsky_score"].values[:, numpy.newaxis]

    tree = SurvivalTree(max_depth=1)
    tree.fit(X, y)

    stats = LogrankTreeBuilder(max_depth=1).build(X, y)

    assert tree.tree_.capacity == stats.shape[0]
    assert_array_equal(tree.tree_.feature, stats.loc[:, "feature"].values)
    assert_array_equal(tree.tree_.n_node_samples, stats.loc[:, "n_node_samples"].values)
    assert_array_almost_equal(tree.tree_.threshold, stats.loc[:, "threshold"].values)

    expected_time = numpy.array([
        1, 2, 3, 4, 7, 8, 10, 11, 12, 13, 15, 16, 18, 19, 20,
        21, 22, 24, 25, 27, 29, 30, 31, 33, 35, 36, 42, 43, 44,
        45, 48, 49, 51, 52, 53, 54, 56, 59, 61, 63, 72, 73, 80,
        82, 84, 87, 90, 92, 95, 99, 100, 103, 105, 110, 111, 112,
        117, 118, 122, 126, 132, 133, 139, 140, 143, 144, 151, 153,
        156, 162, 164, 177, 186, 200, 201, 216, 228, 231, 242, 250,
        260, 278, 283, 287, 314, 340, 357, 378, 384, 389, 392, 411,
        467, 553, 587, 991, 999], dtype=float)
    assert_array_equal(tree.event_times_, expected_time)

    threshold = stats.loc[0, "threshold"]
    m = X[:, 0] <= threshold
    y_left = y[m]
    _, chf_left = nelson_aalen_estimator(
        y_left["Status"], y_left["Survival_in_days"])

    y_right = y[~m]
    _, chf_right = nelson_aalen_estimator(
        y_right["Status"], y_right["Survival_in_days"])

    X_pred = numpy.array([[threshold - 10], [threshold + 10]])
    chf_pred = tree.predict_cumulative_hazard_function(
        X_pred, return_array=True)

    assert_curve_almost_equal(chf_pred[0], chf_left)
    assert_curve_almost_equal(chf_pred[1], chf_right)

    mrt_pred = tree.predict(X_pred)
    assert_array_almost_equal(mrt_pred, numpy.array([196.55878, 86.14939]))

    _, surv_left = kaplan_meier_estimator(
        y_left["Status"], y_left["Survival_in_days"])
    _, surv_right = kaplan_meier_estimator(
        y_right["Status"], y_right["Survival_in_days"])

    surv_pred = tree.predict_survival_function(
        X_pred, return_array=True)

    assert_curve_almost_equal(surv_pred[0], surv_left)
    assert_curve_almost_equal(surv_pred[1], surv_right)


def test_tree_two_split(veterans):
    X, y = veterans
    X = X.loc[:, "Karnofsky_score"].values[:, numpy.newaxis]

    tree = SurvivalTree(max_depth=2, max_features=1)
    tree.fit(X, y)

    assert tree.tree_.capacity == 7
    assert_array_equal(
        tree.tree_.threshold, numpy.array(
            [45., 25., TREE_UNDEFINED, TREE_UNDEFINED,
             87.5, TREE_UNDEFINED, TREE_UNDEFINED]))
    expected_size = numpy.array([X.shape[0], 38, 8, 30, 99, 91, 8])
    assert_array_equal(tree.tree_.n_node_samples, expected_size)

    X_pred = numpy.array([66.05, 87.91, 45.62, 40.18, 50.65, 71.24,
                          96.21, 33.33, 11.57, 94.28]).reshape(-1, 1)
    mrt_pred = tree.predict(X_pred)
    expected_risk = numpy.array([96.7044629620645, 19.6309523809524, 96.7044629620645,
                                 179.264571990757, 96.7044629620645, 96.7044629620645,
                                 19.6309523809524, 179.264571990757, 214.027380952381,
                                 19.6309523809524])
    assert_array_almost_equal(mrt_pred, expected_risk)

    chf_pred = tree.predict_cumulative_hazard_function(
        X_pred, return_array=True)
    assert numpy.all(numpy.diff(chf_pred) >= 0)

    surv_pred = tree.predict_survival_function(
        X_pred, return_array=True)
    assert numpy.all(numpy.diff(surv_pred) <= 0)


def test_tree_split_all_censored(veterans):
    X, y = veterans
    X = X.loc[:, "Karnofsky_score"].values[:, numpy.newaxis]
    y["Status"][X[:, 0] > 45.] = False

    tree = SurvivalTree(max_depth=2, max_features=1)
    tree.fit(X, y)

    assert tree.tree_.capacity == 5
    assert_array_equal(
        tree.tree_.threshold, numpy.array(
            [45., 25., TREE_UNDEFINED, TREE_UNDEFINED, TREE_UNDEFINED]))
    expected_size = numpy.array([X.shape[0], 38, 8, 30, 99])
    assert_array_equal(tree.tree_.n_node_samples, expected_size)


@pytest.mark.slow
def test_toy_data(toy_data):
    X, y = toy_data
    tree = SurvivalTree(max_depth=4, max_features=1.0, min_samples_leaf=20)
    tree.fit(X, y)

    stats = LogrankTreeBuilder(max_depth=4, min_leaf=20).build(X, y)

    assert tree.tree_.capacity == stats.shape[0]
    assert_array_equal(tree.tree_.feature, stats.loc[:, "feature"].values)
    assert_array_equal(tree.tree_.n_node_samples, stats.loc[:, "n_node_samples"].values)
    assert_array_almost_equal(tree.tree_.threshold, stats.loc[:, "threshold"].values, 5)


def test_breast_cancer_1(breast_cancer):
    X, y = breast_cancer

    tree = SurvivalTree(max_features="auto",
                        max_depth=5,
                        max_leaf_nodes=10,
                        min_samples_split=0.06,
                        min_samples_leaf=0.03,
                        random_state=6)
    tree.fit(X.values, y)

    assert tree.tree_.capacity == 19
    assert_array_equal(tree.tree_.feature, numpy.array(
        [61, 29, 5, TREE_UNDEFINED, 40, 65, TREE_UNDEFINED, 10, 12, 4,
         TREE_UNDEFINED, TREE_UNDEFINED, TREE_UNDEFINED, TREE_UNDEFINED,
         TREE_UNDEFINED, TREE_UNDEFINED, 10, TREE_UNDEFINED, TREE_UNDEFINED]))
    assert_array_equal(tree.tree_.n_node_samples, numpy.array(
        [198, 170, 28, 8, 20, 164, 6, 59, 105, 74, 31, 9, 65,
         13, 7, 39, 20, 7, 13]))
    assert_array_almost_equal(tree.tree_.threshold, numpy.array(
        [10.97448, 11.10251, 11.34859, TREE_UNDEFINED, 10.53533, 8.08848,
         TREE_UNDEFINED, 10.86403, 10.14138, 11.49171, TREE_UNDEFINED,
         TREE_UNDEFINED, TREE_UNDEFINED, TREE_UNDEFINED, TREE_UNDEFINED,
         TREE_UNDEFINED, 11.01874, TREE_UNDEFINED, TREE_UNDEFINED]), 5)


def test_breast_cancer_2(breast_cancer):
    X, y = breast_cancer

    tree = SurvivalTree(max_features="log2",
                        splitter="random",
                        max_depth=5,
                        min_samples_split=30,
                        min_samples_leaf=15,
                        random_state=6)
    tree.fit(X.values, y)

    assert tree.tree_.capacity == 11
    assert_array_equal(tree.tree_.feature, numpy.array(
        [55, 14, TREE_UNDEFINED, 60, 23, TREE_UNDEFINED, TREE_UNDEFINED, 31,
         TREE_UNDEFINED, TREE_UNDEFINED, TREE_UNDEFINED]))
    assert_array_equal(tree.tree_.n_node_samples, numpy.array(
        [198, 153, 76, 77, 46, 16, 30, 31, 16, 15, 45]))
    assert_array_almost_equal(tree.tree_.threshold, numpy.array(
        [11.3019, 9.0768, TREE_UNDEFINED, 8.6903, 6.83564, TREE_UNDEFINED,
         TREE_UNDEFINED, 10.66262, TREE_UNDEFINED, TREE_UNDEFINED,
         TREE_UNDEFINED]), 5)


@pytest.mark.parametrize("func", ("predict_survival_function", "predict_cumulative_hazard_function"))
def test_predict_step_function(breast_cancer, func):
    X, y = breast_cancer

    tree = SurvivalTree(max_features="log2",
                        splitter="random",
                        max_depth=5,
                        min_samples_split=30,
                        min_samples_leaf=15,
                        random_state=6)
    tree.fit(X.iloc[10:], y[10:])

    pred_fn = getattr(tree, func)

    ret_array = pred_fn(X.iloc[:10], return_array=True)
    fn_array = pred_fn(X.iloc[:10], return_array=False)

    assert ret_array.shape[0] == fn_array.shape[0]

    for fn, arr in zip(fn_array, ret_array):
        assert_array_almost_equal(fn.x, tree.event_times_)
        assert_array_almost_equal(fn.y, arr)


@pytest.mark.parametrize("func", ("predict_survival_function", "predict_cumulative_hazard_function"))
def test_predict_step_function_warning(toy_data, func):
    X, y = toy_data
    tree = SurvivalTree(max_depth=1)
    tree.fit(X, y)

    pred_fn = getattr(tree, func)

    with pytest.warns(FutureWarning,
                      match="{} will return an array of StepFunction instances in 0.14".format(func)):
        pred_fn(X)


@pytest.mark.parametrize("func", ("predict_survival_function", "predict_cumulative_hazard_function"))
def test_pipeline_predict(breast_cancer, func):
    X_num, y = breast_cancer
    X_num = X_num.loc[:, ["er", "grade"]].values

    tree = SurvivalTree().fit(X_num[10:], y[10:])

    X_str, _ = load_breast_cancer()
    X_str = X_str.loc[:, ["er", "grade"]].values

    pipe = make_pipeline(OrdinalEncoder(), SurvivalTree())
    pipe.fit(X_str[10:], y[10:])

    tree_pred = getattr(tree, func)(X_num[:10], return_array=True)
    pipe_pred = getattr(pipe, func)(X_str[:10], return_array=True)

    assert_array_almost_equal(tree_pred, pipe_pred)


@pytest.mark.parametrize("n_features", [1, 3, 5, 10])
def test_predict_wrong_features(toy_data, n_features):
    X, y = toy_data
    tree = SurvivalTree(max_depth=1)
    tree.fit(X, y)

    with pytest.raises(ValueError, match="Number of features of the model must "
                                         "match the input. Model n_features is 4 and "
                                         "input n_features is {}.".format(n_features)):
        X_new = numpy.random.randn(12, n_features)
        tree.predict(X_new)


@pytest.mark.parametrize("val", [0, 0.0, -1, -1e-6, -1512])
def test_max_depth(fake_data, val):
    X, y = fake_data
    tree = SurvivalTree(max_depth=val)

    with pytest.raises(ValueError,
                       match="max_depth must be greater than zero."):
        tree.fit(X, y)


@pytest.mark.parametrize("val", [0, 0.0, -1, -1e-6, -1512, 10.0, 0.51, 1.0, numpy.nan, numpy.infty])
def test_min_samples_leaf(fake_data, val):
    X, y = fake_data
    tree = SurvivalTree(min_samples_leaf=val)

    with pytest.raises(ValueError,
                       match=r"min_samples_leaf must be at least 1 "
                             r"or in \(0, 0\.5\], got"):
        tree.fit(X, y)


@pytest.mark.parametrize("val", [0, 0.0, 1, -1, -1e-6, -1512, 10.0, 1.000001, numpy.nan, -numpy.infty, numpy.infty])
def test_min_samples_split(fake_data, val):
    X, y = fake_data
    tree = SurvivalTree(min_samples_split=val)

    with pytest.raises(ValueError,
                       match="min_samples_split must be an integer "
                             r"greater than 1 or a float in \(0\.0, 1\.0\]; "
                             "got "):
        tree.fit(X, y)


@pytest.mark.parametrize("val", [1, -1, -1e-6, -1512, 0.500001, numpy.nan, -numpy.infty, numpy.infty])
def test_min_weight_fraction_leaf(fake_data, val):
    X, y = fake_data
    tree = SurvivalTree(min_weight_fraction_leaf=val)

    with pytest.raises(ValueError,
                       match=r"min_weight_fraction_leaf must in \[0, 0\.5\]"):
        tree.fit(X, y)


@pytest.mark.parametrize("val", ["", "None", "sqrt_", "log10", "car"])
def test_max_features_invalid(fake_data, val):
    X, y = fake_data
    tree = SurvivalTree(max_features=val)

    with pytest.raises(ValueError,
                       match='Invalid value for max_features. Allowed string '
                             'values are "auto", "sqrt" or "log2".'):
        tree.fit(X, y)


@pytest.mark.parametrize("val", [0, 0.0, 12, 13, 100, 865411])
def test_max_features_too_large(fake_data, val):
    X, y = fake_data
    tree = SurvivalTree(max_features=val)

    with pytest.raises(ValueError,
                       match=r"max_features must be in \(0, n_features\]"):
        tree.fit(X, y)


@pytest.mark.parametrize("val", [12., 13.1, 1.11, numpy.nan, numpy.infty])
def test_max_leaf_nodes_no_int(fake_data, val):
    X, y = fake_data
    tree = SurvivalTree(max_leaf_nodes=val)

    with pytest.raises(ValueError,
                       match="max_leaf_nodes must be integral number but was "):
        tree.fit(X, y)


@pytest.mark.parametrize("val", [0, 1])
def test_max_leaf_nodes_too_small(fake_data, val):
    X, y = fake_data
    tree = SurvivalTree(max_leaf_nodes=val)

    with pytest.raises(ValueError,
                       match="max_leaf_nodes {} must be either None "
                             "or larger than 1".format(val)):
        tree.fit(X, y)


@pytest.mark.parametrize("val", [-1, "False", "True", "", numpy.nan])
def test_presort(fake_data, val):
    X, y = fake_data
    tree = SurvivalTree(presort=val)

    with pytest.warns(DeprecationWarning,
                      match="The parameter 'presort' is deprecated "):
        tree.fit(X, y)
