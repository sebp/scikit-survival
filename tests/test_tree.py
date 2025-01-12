from itertools import product
from queue import LifoQueue

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pandas as pd
import pytest
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree._tree import TREE_UNDEFINED

from sksurv.compare import compare_survival
from sksurv.datasets import load_breast_cancer, load_veterans_lung_cancer
from sksurv.nonparametric import kaplan_meier_estimator, nelson_aalen_estimator
from sksurv.tree import SurvivalTree
from sksurv.util import Surv


@pytest.fixture()
def veterans():
    return load_veterans_lung_cancer()


@pytest.fixture()
def breast_cancer():
    X, y = load_breast_cancer()
    X["er"] = X.loc[:, "er"].map({"negative": 0, "positive": 1})
    X["grade"] = X.loc[:, "grade"].map(
        {
            "intermediate": 0,
            "poorly differentiated": 1,
            "unkown": 2,
            "well differentiated": 3,
        }
    )
    return X, y


@pytest.fixture()
def toy_data():
    rnd = np.random.RandomState(1)
    n_samples = 500
    X = np.empty((n_samples, 4), dtype=float)
    X[:, :2] = rnd.normal(scale=2, size=(n_samples, 2))
    X[:, 2] = rnd.uniform(40, 80, size=n_samples)
    X[:, 3] = rnd.binomial(1, 0.4, size=n_samples)

    time = np.zeros(n_samples, dtype=float)
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

    event = np.ones(n_samples, dtype=bool)
    event[rnd.binomial(1, 0.333, size=n_samples).astype(bool)] = False

    y = np.fromiter(zip(event, time), dtype=[("status", bool), ("time", float)])
    return X, y


def supported_float_dtypes():
    names = (
        "float16",
        "float32",
        "float64",
        "float128",
    )
    return [i for i in names if hasattr(np, i)]


def assert_curve_almost_equal(x, y):
    jumps_x = np.diff(x) != 0
    jumps_y = np.diff(y) != 0

    pytest.approx(y[0], x[0])
    assert_array_almost_equal(x[:1], y[:1])
    assert_array_almost_equal(x[1:][jumps_x], y[1:][jumps_y])


class LogrankTreeBuilder:
    def __init__(self, max_depth=4, min_leaf=20):
        self.max_depth = max_depth
        self.min_leaf = min_leaf

    def build(self, X, y):
        val, feat, stat = self._get_best_split(X, y)
        splits = LifoQueue()
        splits.put((val, feat, stat, 0, np.arange(X.shape[0])))

        node_stats = []
        while splits.qsize() > 0:
            val, feat, stat, lvl, idx = splits.get()
            s = {"feature": feat, "threshold": val, "n_node_samples": idx.shape[0], "statistic": stat, "depth": lvl}
            node_stats.append(s)

            if val == TREE_UNDEFINED:
                continue

            left = X[idx, feat] <= val
            right = idx[~left]
            left = idx[left]

            if lvl == self.max_depth - 1:
                splits.put([TREE_UNDEFINED, TREE_UNDEFINED, -np.inf, lvl + 1, right])
                splits.put([TREE_UNDEFINED, TREE_UNDEFINED, -np.inf, lvl + 1, left])
                continue

            X_right = X[right, :]
            y_right = y[right]
            s_right = self._get_best_split(X_right, y_right)
            splits.put(list(s_right) + [lvl + 1, right])

            X_left = X[left, :]
            y_left = y[left]
            s_left = self._get_best_split(X_left, y_left)
            splits.put(list(s_left) + [lvl + 1, left])

        return pd.DataFrame.from_dict(dict(zip(range(len(node_stats)), node_stats)), orient="index")

    def _get_best_split(self, X, y):
        min_leaf = self.min_leaf
        best_val = TREE_UNDEFINED
        best_feat = TREE_UNDEFINED
        best_stat = -np.inf

        if y[y.dtype.names[0]].sum() == 0:
            return best_val, best_feat, best_stat
        for j in range(X.shape[1]):
            vals = X[:, j]
            values = np.unique(vals)
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
    X = X.loc[:, "Karnofsky_score"].values[:, np.newaxis]

    tree = SurvivalTree(max_depth=1)
    tree.fit(X, y)

    stats = LogrankTreeBuilder(max_depth=1).build(X, y)

    assert tree.tree_.capacity == stats.shape[0]
    assert_array_equal(tree.tree_.feature, stats.loc[:, "feature"].values)
    assert_array_equal(tree.tree_.n_node_samples, stats.loc[:, "n_node_samples"].values)
    assert_array_almost_equal(tree.tree_.threshold, stats.loc[:, "threshold"].values)

    expected_time = np.array(
        [
            1,
            2,
            3,
            4,
            7,
            8,
            10,
            11,
            12,
            13,
            15,
            16,
            18,
            19,
            20,
            21,
            22,
            24,
            25,
            27,
            29,
            30,
            31,
            33,
            35,
            36,
            42,
            43,
            44,
            45,
            48,
            49,
            51,
            52,
            53,
            54,
            56,
            59,
            61,
            63,
            72,
            73,
            80,
            82,
            83,
            84,
            87,
            90,
            92,
            95,
            97,
            99,
            100,
            103,
            105,
            110,
            111,
            112,
            117,
            118,
            122,
            123,
            126,
            132,
            133,
            139,
            140,
            143,
            144,
            151,
            153,
            156,
            162,
            164,
            177,
            182,
            186,
            200,
            201,
            216,
            228,
            231,
            242,
            250,
            260,
            278,
            283,
            287,
            314,
            340,
            357,
            378,
            384,
            389,
            392,
            411,
            467,
            553,
            587,
            991,
            999,
        ],
        dtype=float,
    )
    assert_array_equal(tree.unique_times_, expected_time)
    assert_array_equal(tree.unique_times_[~tree.is_event_time_], np.array([83, 97, 123, 182], dtype=float))

    threshold = stats.loc[0, "threshold"]
    m = X[:, 0] <= threshold
    y_left = y[m]
    _, chf_left = nelson_aalen_estimator(y_left["Status"], y_left["Survival_in_days"])

    y_right = y[~m]
    _, chf_right = nelson_aalen_estimator(y_right["Status"], y_right["Survival_in_days"])

    X_pred = np.array([[threshold - 10], [threshold + 10]])
    chf_pred = tree.predict_cumulative_hazard_function(X_pred, return_array=True)

    assert_curve_almost_equal(chf_pred[0], chf_left)
    assert_curve_almost_equal(chf_pred[1], chf_right)

    mrt_pred = tree.predict(X_pred)
    assert_array_almost_equal(mrt_pred, np.array([196.55878, 86.14939]))

    _, surv_left = kaplan_meier_estimator(y_left["Status"], y_left["Survival_in_days"])
    _, surv_right = kaplan_meier_estimator(y_right["Status"], y_right["Survival_in_days"])

    surv_pred = tree.predict_survival_function(X_pred, return_array=True)

    assert_curve_almost_equal(surv_pred[0], surv_left)
    assert_curve_almost_equal(surv_pred[1], surv_right)


def test_tree_two_split(veterans):
    X, y = veterans
    X = X.loc[:, "Karnofsky_score"].values[:, np.newaxis]

    tree = SurvivalTree(max_depth=2, max_features=1)
    tree.fit(X, y)

    assert tree.tree_.capacity == 7
    assert_array_equal(
        tree.tree_.threshold,
        np.array([45.0, 25.0, TREE_UNDEFINED, TREE_UNDEFINED, 87.5, TREE_UNDEFINED, TREE_UNDEFINED]),
    )
    expected_size = np.array([X.shape[0], 38, 8, 30, 99, 91, 8])
    assert_array_equal(tree.tree_.n_node_samples, expected_size)

    X_pred = np.array([66.05, 87.91, 45.62, 40.18, 50.65, 71.24, 96.21, 33.33, 11.57, 94.28]).reshape(-1, 1)
    mrt_pred = tree.predict(X_pred)
    expected_risk = np.array(
        [
            96.7044629620645,
            19.6309523809524,
            96.7044629620645,
            179.264571990757,
            96.7044629620645,
            96.7044629620645,
            19.6309523809524,
            179.264571990757,
            214.027380952381,
            19.6309523809524,
        ]
    )
    assert_array_almost_equal(mrt_pred, expected_risk)

    chf_pred = tree.predict_cumulative_hazard_function(X_pred, return_array=True)
    assert np.all(np.diff(chf_pred) >= 0)

    surv_pred = tree.predict_survival_function(X_pred, return_array=True)
    assert np.all(np.diff(surv_pred) <= 0)


def test_tree_split_all_censored(veterans):
    X, y = veterans
    X = X.loc[:, "Karnofsky_score"].values[:, np.newaxis]
    y["Status"][X[:, 0] > 45.0] = False

    tree = SurvivalTree(max_depth=2, max_features=1)
    tree.fit(X, y)

    assert tree.tree_.capacity == 5
    assert_array_equal(tree.tree_.threshold, np.array([45.0, 25.0, TREE_UNDEFINED, TREE_UNDEFINED, TREE_UNDEFINED]))
    expected_size = np.array([X.shape[0], 38, 8, 30, 99])
    assert_array_equal(tree.tree_.n_node_samples, expected_size)


@pytest.mark.slow()
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

    tree = SurvivalTree(
        max_features="sqrt",
        max_depth=5,
        max_leaf_nodes=10,
        min_samples_split=0.06,
        min_samples_leaf=0.03,
        random_state=6,
    )
    tree.fit(X.values, y)

    assert tree.tree_.capacity == 19
    assert_array_equal(
        tree.tree_.feature,
        np.array(
            [
                61,
                29,
                5,
                TREE_UNDEFINED,
                40,
                65,
                TREE_UNDEFINED,
                10,
                12,
                4,
                TREE_UNDEFINED,
                TREE_UNDEFINED,
                TREE_UNDEFINED,
                TREE_UNDEFINED,
                TREE_UNDEFINED,
                TREE_UNDEFINED,
                10,
                TREE_UNDEFINED,
                TREE_UNDEFINED,
            ]
        ),
    )
    assert_array_equal(
        tree.tree_.n_node_samples,
        np.array(
            [
                198,
                170,
                28,
                8,
                20,
                164,
                6,
                59,
                105,
                74,
                31,
                9,
                65,
                13,
                7,
                39,
                20,
                7,
                13,
            ]
        ),
    )
    assert_array_almost_equal(
        tree.tree_.threshold,
        np.array(
            [
                10.97448,
                11.10251,
                11.34859,
                TREE_UNDEFINED,
                10.53533,
                8.08848,
                TREE_UNDEFINED,
                10.86403,
                10.14138,
                11.49171,
                TREE_UNDEFINED,
                TREE_UNDEFINED,
                TREE_UNDEFINED,
                TREE_UNDEFINED,
                TREE_UNDEFINED,
                TREE_UNDEFINED,
                11.01874,
                TREE_UNDEFINED,
                TREE_UNDEFINED,
            ]
        ),
        5,
    )


def test_breast_cancer_2(breast_cancer):
    X, y = breast_cancer

    tree = SurvivalTree(
        max_features="log2", splitter="random", max_depth=5, min_samples_split=30, min_samples_leaf=15, random_state=6
    )
    tree.fit(X.values, y)

    assert tree.tree_.capacity == 11
    assert_array_equal(
        tree.tree_.feature,
        np.array(
            [
                55,
                14,
                TREE_UNDEFINED,
                60,
                23,
                TREE_UNDEFINED,
                TREE_UNDEFINED,
                31,
                TREE_UNDEFINED,
                TREE_UNDEFINED,
                TREE_UNDEFINED,
            ]
        ),
    )
    assert_array_equal(
        tree.tree_.n_node_samples,
        np.array(
            [
                198,
                153,
                76,
                77,
                46,
                16,
                30,
                31,
                16,
                15,
                45,
            ]
        ),
    )
    assert_array_almost_equal(
        tree.tree_.threshold,
        np.array(
            [
                11.3019,
                9.0768,
                TREE_UNDEFINED,
                8.6903,
                6.83564,
                TREE_UNDEFINED,
                TREE_UNDEFINED,
                10.66262,
                TREE_UNDEFINED,
                TREE_UNDEFINED,
                TREE_UNDEFINED,
            ]
        ),
        5,
    )


def test_fit_int_time(breast_cancer):
    X, y = breast_cancer
    y_int = np.empty(y.shape[0], dtype=[(y.dtype.names[0], bool), (y.dtype.names[1], int)])
    y_int[:] = y

    tree_f = SurvivalTree(
        max_features="log2",
        splitter="random",
        max_depth=5,
        min_samples_split=30,
        min_samples_leaf=15,
        random_state=6,
    ).fit(X, y)

    tree_i = SurvivalTree(
        max_features="log2",
        splitter="random",
        max_depth=5,
        min_samples_split=30,
        min_samples_leaf=15,
        random_state=6,
    ).fit(X, y_int)

    assert_array_almost_equal(tree_f.unique_times_, tree_i.unique_times_)
    assert_array_equal(tree_f.tree_.feature, tree_i.tree_.feature)
    assert_array_equal(tree_f.tree_.n_node_samples, tree_i.tree_.n_node_samples)
    assert_array_almost_equal(tree_f.tree_.threshold, tree_i.tree_.threshold)


@pytest.mark.parametrize("dtype,missing", product(supported_float_dtypes(), [False, True]))
def test_fit_dtype(toy_data, dtype, missing):
    X, y = toy_data
    if missing:
        X[:23, 0] = np.nan
    X = X.astype(dtype)

    tree = SurvivalTree()
    tree.fit(X, y)
    assert hasattr(tree, "tree_")

    pred = tree.predict(X)
    assert pred.shape[0] == X.shape[0]


@pytest.mark.parametrize("func", ["predict_survival_function", "predict_cumulative_hazard_function"])
def test_predict_step_function(breast_cancer, func):
    X, y = breast_cancer

    tree = SurvivalTree(
        max_features="log2",
        splitter="random",
        max_depth=5,
        min_samples_split=30,
        min_samples_leaf=15,
        random_state=6,
    )
    tree.fit(X.iloc[10:], y[10:])

    pred_fn = getattr(tree, func)

    ret_array = pred_fn(X.iloc[:10], return_array=True)
    fn_array = pred_fn(X.iloc[:10], return_array=False)

    assert ret_array.shape[0] == fn_array.shape[0]

    for fn, arr in zip(fn_array, ret_array):
        assert_array_almost_equal(fn.x, tree.unique_times_)
        assert_array_almost_equal(fn.y, arr)


@pytest.mark.parametrize("func", ["predict_survival_function", "predict_cumulative_hazard_function"])
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

    X_new = np.random.randn(12, n_features)
    with pytest.raises(
        ValueError, match=f"X has {n_features} features, but SurvivalTree is expecting 4 features as input."
    ):
        tree.predict(X_new)


@pytest.mark.parametrize("val", [0, 0.0, -1, -1e-6, -1512])
def test_max_depth(fake_data, val):
    X, y = fake_data
    tree = SurvivalTree(max_depth=val)

    msg = (
        r"The 'max_depth' parameter of SurvivalTree must be "
        rf"an int in the range \[1, inf\) or None\. Got {val!r} instead\."
    )
    with pytest.raises(ValueError, match=msg):
        tree.fit(X, y)


@pytest.mark.parametrize("val", [0, 0.0, -1, -1e-6, -1512, 10.0, 0.51, 1.0, np.nan, np.inf])
def test_min_samples_leaf(fake_data, val):
    X, y = fake_data
    tree = SurvivalTree(min_samples_leaf=val)

    msg = (
        r"The 'min_samples_leaf' parameter of SurvivalTree must be an int in the range \[1, inf\) or "
        rf"a float in the range \(0\.0, 0\.5\]\. Got {val!r} instead\."
    )
    with pytest.raises(ValueError, match=msg):
        tree.fit(X, y)


@pytest.mark.parametrize("val", [0, 0.0, 1, 1.0, -1, -1e-6, -1512, 10.0, 1.000001, np.nan, -np.inf, np.inf])
def test_min_samples_split(fake_data, val):
    X, y = fake_data
    tree = SurvivalTree(min_samples_split=val)

    msg = (
        r"The 'min_samples_split' parameter of SurvivalTree must be an int in the range \[2, inf\) or "
        rf"a float in the range \(0\.0, 1\.0\)\. Got {val!r} instead\."
    )
    with pytest.raises(ValueError, match=msg):
        tree.fit(X, y)


@pytest.mark.parametrize("val", [1, -1, -1e-6, -1512, 0.500001, np.nan, -np.inf, np.inf])
def test_min_weight_fraction_leaf(fake_data, val):
    X, y = fake_data
    tree = SurvivalTree(min_weight_fraction_leaf=val)

    msg = (
        r"The 'min_weight_fraction_leaf' parameter of SurvivalTree must be "
        rf"a float in the range \[0\.0, 0\.5\]\. Got {val!r} instead\."
    )
    with pytest.raises(ValueError, match=msg):
        tree.fit(X, y)


@pytest.mark.parametrize("val", [0, 0.0, "", "None", "sqrt_", "log10", "car"])
def test_max_features_invalid(fake_data, val):
    X, y = fake_data
    tree = SurvivalTree(max_features=val)

    msg = (
        r"The 'max_features' parameter of SurvivalTree must be "
        r"an int in the range \[1, inf\), a float in the range \(0\.0, 1\.0\], "
        r"a str among {.+} or None\."
    )
    with pytest.raises(ValueError, match=msg):
        tree.fit(X, y)


@pytest.mark.parametrize("val", [12, 13, 100, 865411])
def test_max_features_too_large(fake_data, val):
    X, y = fake_data
    tree = SurvivalTree(max_features=val)

    with pytest.raises(ValueError, match=r"max_features must be in \(0, n_features\]"):
        tree.fit(X, y)


@pytest.mark.parametrize("val", [12.0, 13.1, 1.11, np.nan, np.inf])
def test_max_leaf_nodes_no_int(fake_data, val):
    X, y = fake_data
    tree = SurvivalTree(max_leaf_nodes=val)

    msg = r"The 'max_leaf_nodes' parameter of SurvivalTree must be an int in the range \[2, inf\) or None\."
    with pytest.raises(ValueError, match=msg):
        tree.fit(X, y)


@pytest.mark.parametrize("val", [0, 1])
def test_max_leaf_nodes_too_small(fake_data, val):
    X, y = fake_data
    tree = SurvivalTree(max_leaf_nodes=val)

    msg = r"The 'max_leaf_nodes' parameter of SurvivalTree must be an int in the range \[2, inf\) or None\."
    with pytest.raises(ValueError, match=msg):
        tree.fit(X, y)


def test_apply(veterans):
    X, y = veterans
    X = X.loc[:, "Karnofsky_score"].values[:, np.newaxis].astype(np.float32)

    tree = SurvivalTree(max_depth=2, max_features=1)
    tree.fit(X, y)

    X_trans = tree.apply(X)

    assert X_trans.shape[0] == X.shape[0]
    assert all(X_trans >= 0)
    assert all(X_trans < tree.tree_.node_count)

    X_path = tree.decision_path(X).toarray()

    assert X_path.shape[0] == X.shape[0]
    assert X_path.shape[1] == tree.tree_.node_count

    ones = X_path[np.arange(X.shape[0]), X_trans]
    assert_array_equal(ones, np.ones(X.shape[0]))


def test_apply_sparse(veterans):
    X, y = veterans
    X = X.loc[:, "Karnofsky_score"].values[:, np.newaxis].astype(np.float32)
    X_sparse = sparse.csr_matrix(X)
    tree = SurvivalTree(max_depth=2, max_features=1)
    tree.fit(X_sparse, y)

    X_trans = tree.apply(X_sparse)

    assert X_trans.shape[0] == X.shape[0]
    assert all(X_trans >= 0)
    assert all(X_trans < tree.tree_.node_count)

    X_path = tree.decision_path(X_sparse).toarray()

    assert X_path.shape[0] == X.shape[0]
    assert X_path.shape[1] == tree.tree_.node_count

    ones = X_path[np.arange(X.shape[0]), X_trans]
    assert_array_equal(ones, np.ones(X.shape[0]))


def test_predict_sparse(make_whas500):
    seed = 42
    whas500 = make_whas500(to_numeric=True)
    X, y = whas500.x, whas500.y
    # Duplicates values in whas500 leads to assert errors because of
    # tie resolution during tree fitting.
    # Using a synthetic dataset resolves this issue.
    X = np.random.RandomState(seed).binomial(n=5, p=0.1, size=X.shape)

    X_train, X_test, y_train, _ = train_test_split(X, y, random_state=seed)

    tree = SurvivalTree(min_samples_leaf=10, random_state=seed)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    y_cum_h = tree.predict_cumulative_hazard_function(X_test)
    y_surv = tree.predict_survival_function(X_test)

    X_train_csr = sparse.csr_matrix(X_train)
    X_test_csr = sparse.csr_matrix(X_test)

    tree_csr = SurvivalTree(min_samples_leaf=10, random_state=seed)
    tree_csr.fit(X_train_csr, y_train)
    y_pred_csr = tree_csr.predict(X_test_csr)
    y_cum_h_csr = tree_csr.predict_cumulative_hazard_function(X_test_csr)
    y_surv_csr = tree_csr.predict_survival_function(X_test_csr)

    assert y_pred.shape[0] == X_test.shape[0]
    assert y_pred_csr.shape[0] == X_test.shape[0]

    assert_array_equal(y_pred, y_pred_csr)
    assert_array_equal(y_cum_h, y_cum_h_csr)
    assert_array_equal(y_surv, y_surv_csr)


def test_missing_values_best_splitter_to_max_samples(veterans):
    X, y = veterans
    X = X.loc[:, "Karnofsky_score"].values[:, np.newaxis].astype(np.float32)

    tree = SurvivalTree(max_depth=1)
    tree.fit(X, y)

    y_pred_chf = tree.predict_cumulative_hazard_function([[np.nan]], return_array=True)
    y_pred_surv = tree.predict_survival_function([[np.nan]], return_array=True)
    y_pred = np.column_stack((y_pred_chf[0], y_pred_surv[0]))

    # missing values go to node with the most samples
    assert np.argmax(tree.tree_.n_node_samples[1:]) == 1
    y_expected = tree.tree_.value[2]
    assert_array_almost_equal(y_pred, y_expected)


def test_missing_values_best_splitter_to_right():
    X = np.array([[np.nan] * 8 + list(range(7))], dtype=np.float32).T
    y = Surv.from_arrays(time=np.concatenate((np.arange(8) + 10, np.arange(6, 13))), event=np.ones(15, dtype=bool))

    tree = SurvivalTree(max_depth=2)
    tree.fit(X, y)

    y_pred_chf = tree.predict_cumulative_hazard_function([[np.nan]], return_array=True)
    y_pred_surv = tree.predict_survival_function([[np.nan]], return_array=True)
    y_pred = np.column_stack((y_pred_chf[0], y_pred_surv[0]))

    # missing values go to the right
    y_expected = tree.tree_.value[4]
    assert_array_almost_equal(y_pred, y_expected)
