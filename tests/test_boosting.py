from contextlib import nullcontext as does_not_raise
from os.path import dirname, join

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pandas as pd
import pytest
from sklearn.metrics import mean_squared_error

from sksurv.column import categorical_to_numeric, standardize
from sksurv.datasets import load_whas500
from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis, GradientBoostingSurvivalAnalysis
from sksurv.testing import FixtureParameterFactory, assert_cindex_almost_equal
from sksurv.util import Surv

CGBOOST_CUMHAZ_FILE = join(dirname(__file__), 'data', 'compnentwise-gradient-boosting-coxph-cumhazard.csv')
CGBOOST_SURV_FILE = join(dirname(__file__), 'data', 'compnentwise-gradient-boosting-coxph-surv.csv')
GBOOST_CUMHAZ_FILE = join(dirname(__file__), 'data', 'gradient-boosting-coxph-cumhazard.csv')
GBOOST_SURV_FILE = join(dirname(__file__), 'data', 'gradient-boosting-coxph-surv.csv')


def early_stopping_monitor(i, est, locals_):  # pylint: disable=unused-argument
    """Returns True on the 10th iteration. """
    return i == 9


class MaxFeaturesCases(FixtureParameterFactory):

    _prefix = (
        "The 'max_features' parameter of GradientBoostingSurvivalAnalysis must be "
        r"an int in the range \[1, inf\), a float in the range \(0\.0, 1\.0\], a str "
        r"among {.+} or None\. "
    )

    @property
    def n_features(self):
        return 14

    def data_1p0(self):
        return 1.0, self.n_features, does_not_raise()

    def data_sqrt(self):
        return "sqrt", np.sqrt(self.n_features), does_not_raise()

    def data_log2(self):
        return "log2", np.log2(self.n_features), does_not_raise()

    def data_0p25(self):
        return 0.25, 0.25 * self.n_features, does_not_raise()

    def data_5(self):
        return 5, 5, does_not_raise()

    def data_m1(self):
        return -1, None, pytest.raises(ValueError, match=MaxFeaturesCases._prefix + r"Got -1 instead\.")

    def data_m1p125(self):
        return -1.125, None, pytest.raises(ValueError, match=MaxFeaturesCases._prefix + r"Got -1\.125 instead\.")

    def data_fail_me(self):
        return "fail_me", None, pytest.raises(
            ValueError,
            match=MaxFeaturesCases._prefix + r"Got 'fail_me' instead\."
        )


class TestGradientBoosting:
    @property
    def data(self):
        x, y = load_whas500()
        x = categorical_to_numeric(standardize(x, with_std=False))
        return x.values, y

    def assert_fit_and_predict(self, expected_cindex, mask_test=None, **kwargs):
        X, y = self.data
        if mask_test is None:
            X_train, y_train, X_test, y_test = X, y, X, y
        else:
            X_train = X[~mask_test]
            y_train = y[~mask_test]
            X_test = X[mask_test]
            y_test = y[mask_test]

        kwargs.setdefault("random_state", 0)
        model = GradientBoostingSurvivalAnalysis(**kwargs)
        model.fit(X_train, y_train)

        if expected_cindex is not None:
            p = model.predict(X_test)

            assert_cindex_almost_equal(
                y_test['fstat'], y_test['lenfol'], p, expected_cindex,
            )

        return model

    def test_fit(self):
        model = self.assert_fit_and_predict(
            expected_cindex=(0.86272605091218779, 64826, 10309, 14, 14),
            n_estimators=100, max_depth=3, min_samples_split=10,
        )

        with pytest.warns(FutureWarning):
            assert model.loss_.__class__.__name__ == "CoxPH"

        assert model.max_features_ == 14
        assert not hasattr(model, "oob_improvement_")

        assert (100,) == model.train_score_.shape

    def test_predict_incorrect_features(self):
        model = self.assert_fit_and_predict(expected_cindex=None)
        with pytest.raises(
            ValueError,
            match="X has 2 features, but GradientBoostingSurvivalAnalysis is expecting 14 features as input.",
        ):
            model.predict(np.random.randn(10, 2))

    def test_fit_subsample(self):
        idx = np.array([
            229, 337, 327, 416, 306, 131, 5, 431, 434, 134, 276, 264, 34, 274, 489,
            11, 378, 467, 425, 241, 93, 310, 189, 277, 138, 332, 313, 59, 212, 171,
            375, 85, 266, 36, 272, 287, 185, 338, 33, 343, 363, 9, 204, 472, 48,
            280, 15, 184, 24, 82,
        ], dtype=int)
        incl_mask = np.zeros(500, dtype=bool)
        incl_mask[idx] = True

        model = self.assert_fit_and_predict(
            expected_cindex=(0.7754172989377845, 511, 148, 0, 0),
            mask_test=incl_mask,
            n_estimators=50,
            learning_rate=0.2,
            min_samples_split=10,
            max_features=1.0,
            subsample=0.637,
        )

        assert model.max_features_ == 14
        assert hasattr(model, "oob_improvement_")

        assert (50,) == model.train_score_.shape
        assert (50,) == model.oob_improvement_.shape

    @pytest.mark.slow()
    def test_fit_dropout(self):
        model = self.assert_fit_and_predict(
            expected_cindex=(0.9094333, 68343, 6806, 0, 14),
            n_estimators=100, max_features=8, learning_rate=1.0, dropout_rate=0.03,
        )

        assert not hasattr(model, "oob_improvement_")
        assert model.max_features_ == 8

    @pytest.mark.parametrize(
        "parameter,value",
        [
            ("max_depth", 4.0),
            ("n_estimators", 100.0),
            ("max_leaf_nodes", 15.0),
            ("min_samples_split", 10.0),
        ]
    )
    def test_fit_int_param_as_float(self, parameter, value):
        kwargs = {parameter: value}
        model = GradientBoostingSurvivalAnalysis(**kwargs)

        msg = f"The '{parameter}' parameter of GradientBoostingSurvivalAnalysis " \
              "must be an int in the range "

        X, y = self.data
        with pytest.raises(ValueError, match=msg):
            model.fit(X, y)

    @pytest.mark.parametrize("fn,expected_file",
                             [("predict_survival_function", GBOOST_SURV_FILE),
                              ("predict_cumulative_hazard_function", GBOOST_CUMHAZ_FILE)])
    def test_predict_function(self, fn, expected_file):
        X, y = self.data

        train_y = y[10:]
        test_x = X[:10]
        test_mask = np.zeros(500, dtype=bool)
        test_mask[:10] = True

        model = self.assert_fit_and_predict(
            expected_cindex=None,
            mask_test=test_mask,
            n_estimators=100, max_depth=2,
        )

        surv_fn = getattr(model, fn)(test_x)

        times = np.unique(train_y["lenfol"][train_y["fstat"]])
        actual = np.row_stack([fn_gb(times) for fn_gb in surv_fn])

        expected = np.loadtxt(expected_file, delimiter=",")

        assert_array_almost_equal(actual, expected)

    @pytest.mark.parametrize("max_features,expected_features,expected_error", MaxFeaturesCases().get_cases())
    def test_max_features(self, max_features, expected_features, expected_error):
        X, y = self.data

        model = GradientBoostingSurvivalAnalysis(
            n_estimators=10, max_features=max_features, max_depth=3, random_state=0,
        )

        with expected_error:
            model.fit(X, y)

        if expected_features is not None:
            assert round(abs(model.max_features_ - int(expected_features)), 7) == 0

    def test_ccp_alpha(self):
        est_full = self.assert_fit_and_predict(
            expected_cindex=None,
            n_estimators=10, max_leaf_nodes=20, random_state=1,
        )

        est_pruned = self.assert_fit_and_predict(
            expected_cindex=None,
            n_estimators=10, max_leaf_nodes=20, ccp_alpha=10.0, random_state=1,
        )

        tree = est_full.estimators_[0, 0].tree_
        subtree = est_pruned.estimators_[0, 0].tree_
        assert tree.node_count > subtree.node_count
        assert tree.max_depth > subtree.max_depth

    @staticmethod
    def test_negative_ccp_alpha(make_whas500):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        clf = GradientBoostingSurvivalAnalysis()
        msg = "The 'ccp_alpha' parameter of GradientBoostingSurvivalAnalysis must be a float in the range " \
              r"\[0\.0, inf\). Got -1\.0 instead\."

        clf.set_params(ccp_alpha=-1.0)
        with pytest.raises(ValueError, match=msg):
            clf.fit(whas500_data.x, whas500_data.y)

    def test_fit_verbose(self):
        self.assert_fit_and_predict(expected_cindex=None, n_estimators=10, verbose=1)

    def test_ipcwls_loss(self):
        model = self.assert_fit_and_predict(
            expected_cindex=None,
            loss="ipcwls", n_estimators=100, max_depth=3,
        )

        with pytest.warns(FutureWarning):
            assert model.loss_.__class__.__name__ == "IPCWLeastSquaresError"

        X, y = self.data
        time_predicted = model.predict(X)
        time_true = y["lenfol"]
        event_true = y["fstat"]

        rmse_all = np.sqrt(mean_squared_error(time_true, time_predicted))
        assert round(abs(rmse_all - 590.5441693629117), 7) == 0

        rmse_uncensored = np.sqrt(mean_squared_error(time_true[event_true], time_predicted[event_true]))
        assert round(abs(rmse_uncensored - 392.97741487479743), 7) == 0

        cindex = model.score(X, y)
        assert round(abs(cindex - 0.8979161399), 7) == 0

        with pytest.raises(ValueError, match="`fit` must be called with the loss option set to 'coxph'"):
            model.predict_survival_function(X)

        with pytest.raises(ValueError, match="`fit` must be called with the loss option set to 'coxph'"):
            model.predict_cumulative_hazard_function(X)

    def test_squared_loss(self):
        model = self.assert_fit_and_predict(
            expected_cindex=None,
            loss="squared", n_estimators=100, max_depth=3,
        )

        with pytest.warns(FutureWarning):
            assert model.loss_.__class__.__name__ == "CensoredSquaredLoss"

        X, y = self.data
        time_predicted = model.predict(X)
        time_true = y["lenfol"]
        event_true = y["fstat"]

        rmse_all = np.sqrt(mean_squared_error(time_true, time_predicted))
        assert round(abs(rmse_all - 580.23345259002951), 7) == 0

        rmse_uncensored = np.sqrt(mean_squared_error(time_true[event_true], time_predicted[event_true]))
        assert round(abs(rmse_uncensored - 383.10639243317951), 7) == 0

        cindex = model.score(X, y)
        assert round(abs(cindex - 0.9021810004), 7) == 0

        with pytest.raises(ValueError, match="`fit` must be called with the loss option set to 'coxph'"):
            model.predict_survival_function(X)

        with pytest.raises(ValueError, match="`fit` must be called with the loss option set to 'coxph'"):
            model.predict_cumulative_hazard_function(X)

    def assert_staged_predict(self, model, X):
        """test if prediction for last stage equals ``predict``"""
        y_pred = model.predict(X)

        # test if prediction for last stage equals ``predict``
        y_staged = None
        for y_staged in model.staged_predict(X):
            assert y_staged.shape == y_pred.shape

        assert_array_equal(y_pred, y_staged)

    @pytest.mark.parametrize("dropout_rate", [0.0, 0.03])
    @pytest.mark.parametrize("loss", ["ipcwls", "squared"])
    def test_loss_staged_predict(self, loss, dropout_rate):
        # Test whether staged decision function eventually gives
        # the same prediction.
        model = self.assert_fit_and_predict(
            expected_cindex=None,
            loss=loss, dropout_rate=dropout_rate, n_estimators=100, max_depth=3,
        )

        X, _ = self.data
        self.assert_staged_predict(model, X)

    def test_monitor_early_stopping(self):
        X, y = self.data

        est = GradientBoostingSurvivalAnalysis(
            loss="ipcwls", n_estimators=50, max_depth=1, subsample=0.5, random_state=0,
        )
        est.fit(X, y, monitor=early_stopping_monitor)

        assert est.n_estimators == 50  # this is not altered
        assert est.estimators_.shape[0] == 10
        assert est.train_score_.shape[0] == 10
        assert est.oob_improvement_.shape[0] == 10


class TestSparseGradientBoosting:
    def assert_fit_and_predict(self, data, **kwargs):
        model = GradientBoostingSurvivalAnalysis(random_state=0, **kwargs)
        model.fit(data.x_sparse, data.y)

        assert model.estimators_.shape[0] == 100
        assert model.train_score_.shape == (100,)

        sparse_predict = model.predict(data.x_sparse)

        model.fit(data.x_sparse, data.y)
        dense_predict = model.predict(data.x_dense.values)

        assert_array_almost_equal(sparse_predict, dense_predict)

        return model

    @pytest.mark.parametrize('loss', ['coxph', 'squared', 'ipcwls'])
    def test_fit(self, whas500_sparse_data, loss):
        model = self.assert_fit_and_predict(
            whas500_sparse_data,
            loss=loss, n_estimators=100, max_depth=1, min_samples_split=10, subsample=0.5,
        )
        assert model.oob_improvement_.shape == (100,)

    @pytest.mark.parametrize('loss', ['coxph', 'squared', 'ipcwls'])
    @pytest.mark.slow()
    def test_dropout(self, whas500_sparse_data, loss):
        self.assert_fit_and_predict(
            whas500_sparse_data,
            loss=loss, n_estimators=100, max_depth=1, min_samples_split=10, dropout_rate=0.03,
        )


class TestComponentwiseGradientBoosting:

    @staticmethod
    def test_fit(make_whas500):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        model = ComponentwiseGradientBoostingSurvivalAnalysis(n_estimators=100)
        model.fit(whas500_data.x, whas500_data.y)

        assert model.loss_.__class__.__name__ == "CoxPH"

        p = model.predict(whas500_data.x)

        assert_cindex_almost_equal(whas500_data.y['fstat'], whas500_data.y['lenfol'], p,
                                   (0.7755659, 58283, 16866, 0, 14))

        expected_coef = pd.Series(np.zeros(15, dtype=float), index=whas500_data.names)
        expected_coef['age'] = 0.040919
        expected_coef['hr'] = 0.004977
        expected_coef['diasbp'] = -0.003407
        expected_coef['bmi'] = -0.017938
        expected_coef['sho'] = 0.429904
        expected_coef['chf'] = 0.508211

        assert_array_almost_equal(expected_coef.values, model.coef_)

        assert (100,) == model.train_score_.shape

        with pytest.raises(ValueError, match="X has 2 features, but ComponentwiseGradientBoostingSurvivalAnalysis is "
                                             "expecting 14 features as input."):
            model.predict(whas500_data.x[:, :2])

    @staticmethod
    def test_fit_subsample(make_whas500):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        model = ComponentwiseGradientBoostingSurvivalAnalysis(n_estimators=100, subsample=0.6, random_state=0)
        model.fit(whas500_data.x, whas500_data.y)
        p = model.predict(whas500_data.x)

        assert_cindex_almost_equal(whas500_data.y['fstat'], whas500_data.y['lenfol'], p,
                                   (0.7750602, 58245, 16904, 0, 14))

        expected_coef = pd.Series(np.zeros(15, dtype=float), index=whas500_data.names)
        expected_coef['age'] = 0.041299
        expected_coef['hr'] = 0.00487
        expected_coef['diasbp'] = -0.003381
        expected_coef['bmi'] = -0.017018
        expected_coef['sho'] = 0.433685
        expected_coef['chf'] = 0.510277

        assert_array_almost_equal(expected_coef.values, model.coef_)

        assert (100,) == model.train_score_.shape
        assert (100,) == model.oob_improvement_.shape

        with pytest.raises(ValueError, match="X has 2 features, but ComponentwiseGradientBoostingSurvivalAnalysis is "
                                             "expecting 14 features as input."):
            model.predict(whas500_data.x[:, :2])

    @staticmethod
    def test_fit_dropout(make_whas500):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        model = ComponentwiseGradientBoostingSurvivalAnalysis(n_estimators=100, learning_rate=1.0,
                                                              dropout_rate=0.03, random_state=0)
        model.fit(whas500_data.x, whas500_data.y)
        p = model.predict(whas500_data.x)

        assert_cindex_almost_equal(whas500_data.y['fstat'], whas500_data.y['lenfol'], p,
                                   (0.7772425, 58409, 16740, 0, 14))

        expected_coef = pd.Series(np.zeros(15, dtype=float), index=whas500_data.names)
        expected_coef['age'] = 0.275537
        expected_coef['hr'] = 0.040048
        expected_coef['diasbp'] = -0.029998
        expected_coef['bmi'] = -0.138909
        expected_coef['sho'] = 3.318941
        expected_coef['chf'] = 2.851386
        expected_coef['mitype'] = -0.075817

        assert_array_almost_equal(expected_coef.values, model.coef_)

    @staticmethod
    @pytest.mark.parametrize("fn,expected_file",
                             [("predict_survival_function", CGBOOST_SURV_FILE),
                              ("predict_cumulative_hazard_function", CGBOOST_CUMHAZ_FILE)])
    def test_predict_function(make_whas500, fn, expected_file):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        model = ComponentwiseGradientBoostingSurvivalAnalysis(n_estimators=100, random_state=0)
        train_x, train_y = whas500_data.x[10:], whas500_data.y[10:]
        model.fit(train_x, train_y)

        test_x = whas500_data.x[:10]
        surv_fn = getattr(model, fn)(test_x)

        times = np.unique(train_y["lenfol"][train_y["fstat"]])
        actual = np.row_stack([fn_gb(times) for fn_gb in surv_fn])

        expected = np.loadtxt(expected_file, delimiter=",")

        assert_array_almost_equal(actual, expected)

    @staticmethod
    def test_feature_importances(make_whas500):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        model = ComponentwiseGradientBoostingSurvivalAnalysis(n_estimators=100, random_state=0)
        model.fit(whas500_data.x, whas500_data.y)

        assert whas500_data.x.shape[1] + 1 == len(model.feature_importances_)

    @staticmethod
    def test_fit_verbose(make_whas500):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        model = ComponentwiseGradientBoostingSurvivalAnalysis(n_estimators=10, verbose=1, random_state=0)
        model.fit(whas500_data.x, whas500_data.y)

    @staticmethod
    def test_ipcwls_loss(make_whas500):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        model = ComponentwiseGradientBoostingSurvivalAnalysis(loss="ipcwls", n_estimators=100, random_state=0)
        model.fit(whas500_data.x, whas500_data.y)

        assert model.loss_.__class__.__name__ == "IPCWLeastSquaresError"

        time_predicted = model.predict(whas500_data.x)
        time_true = whas500_data.y["lenfol"]
        event_true = whas500_data.y["fstat"]

        rmse_all = np.sqrt(mean_squared_error(time_true, time_predicted))
        assert round(abs(rmse_all - 806.283308322), 7) == 0

        rmse_uncensored = np.sqrt(mean_squared_error(time_true[event_true], time_predicted[event_true]))
        assert round(abs(rmse_uncensored - 542.884585289), 7) == 0

        cindex = model.score(whas500_data.x, whas500_data.y)
        assert round(abs(cindex - 0.7773356931), 7) == 0

        with pytest.raises(ValueError, match="`fit` must be called with the loss option set to 'coxph'"):
            model.predict_survival_function(whas500_data.x)

        with pytest.raises(ValueError, match="`fit` must be called with the loss option set to 'coxph'"):
            model.predict_cumulative_hazard_function(whas500_data.x)

    @staticmethod
    def test_squared_loss(make_whas500):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        model = ComponentwiseGradientBoostingSurvivalAnalysis(loss="squared", n_estimators=100, random_state=0)
        model.fit(whas500_data.x, whas500_data.y)

        assert model.loss_.__class__.__name__ == "CensoredSquaredLoss"

        time_predicted = model.predict(whas500_data.x)
        time_true = whas500_data.y["lenfol"]
        event_true = whas500_data.y["fstat"]

        rmse_all = np.sqrt(mean_squared_error(time_true, time_predicted))
        assert round(abs(rmse_all - 793.6256945839657), 7) == 0

        rmse_uncensored = np.sqrt(mean_squared_error(time_true[event_true], time_predicted[event_true]))
        assert round(abs(rmse_uncensored - 542.83358120153525), 7) == 0

        cindex = model.score(whas500_data.x, whas500_data.y)
        assert round(abs(cindex - 0.7777082862), 7) == 0

        with pytest.raises(ValueError, match="`fit` must be called with the loss option set to 'coxph'"):
            model.predict_survival_function(whas500_data.x)

        with pytest.raises(ValueError, match="`fit` must be called with the loss option set to 'coxph'"):
            model.predict_cumulative_hazard_function(whas500_data.x)


@pytest.fixture(params=[GradientBoostingSurvivalAnalysis, ComponentwiseGradientBoostingSurvivalAnalysis])
def sample_gb_class(request):
    x = np.arange(100).reshape(5, 20)
    y = Surv.from_arrays([False, False, True, True, False], [12, 14, 6, 9, 1])

    return request.param, x, y


@pytest.mark.parametrize("n_estimators", [0, -1])
def test_param_n_estimators(sample_gb_class, n_estimators):
    est_cls, x, y = sample_gb_class
    model = est_cls(n_estimators=n_estimators)

    msg = f"The 'n_estimators' parameter of {est_cls.__name__} must be an int " \
          r"in the range \[1, inf\)\. " \
          f"Got {n_estimators} instead\\."

    with pytest.raises(ValueError, match=msg):
        model.fit(x, y)


@pytest.mark.parametrize("learning_rate", [-np.finfo(float).eps, -1])
def test_param_learning_rate(sample_gb_class, learning_rate):
    est_cls, x, y = sample_gb_class
    model = est_cls(learning_rate=learning_rate)

    msg = f"The 'learning_rate' parameter of {est_cls.__name__} must be a float " \
          r"in the range \[0\.0, inf\)\. " \
          f"Got {learning_rate} instead\\."

    with pytest.raises(ValueError, match=msg):
        model.fit(x, y)


@pytest.mark.parametrize("subsample", [0, 1.2])
def test_param_subsample(sample_gb_class, subsample):
    est_cls, x, y = sample_gb_class
    model = est_cls(subsample=subsample)

    msg = f"The 'subsample' parameter of {est_cls.__name__} must be a float " \
          r"in the range \(0\.0, 1\.0\]\. " \
          f"Got {subsample} instead\\."

    with pytest.raises(ValueError, match=msg):
        model.fit(x, y)


@pytest.mark.parametrize("dropout_rate", [-0.1, 1.0, 1.2])
def test_param_dropout_rate(sample_gb_class, dropout_rate):
    est_cls, x, y = sample_gb_class
    model = est_cls(dropout_rate=dropout_rate)

    msg = f"The 'dropout_rate' parameter of {est_cls.__name__} must be a float " \
          r"in the range \[0\.0, 1\.0\)\. " \
          f"Got {dropout_rate} instead\\."

    with pytest.raises(ValueError, match=msg):
        model.fit(x, y)


def test_param_sample_weight(sample_gb_class):
    est_cls, x, y = sample_gb_class
    model = est_cls()

    with pytest.raises(ValueError, match=r"Found input variables with inconsistent numbers of samples: \[5, 3\]"):
        model.fit(x, y, [2, 3, 4])

    model.set_params(dropout_rate=1.2)
    with pytest.raises(ValueError, match=r"Found input variables with inconsistent numbers of samples: \[5, 8\]"):
        model.fit(x, y, [2, 4, 5, 6, 7, 1, 2, 7])


@pytest.mark.parametrize("loss", ["", "unknown", None])
def test_param_loss(sample_gb_class, loss):
    est_cls, x, y = sample_gb_class
    model = est_cls(loss=loss)

    msg = f"The 'loss' parameter of {est_cls.__name__} must be a str among " \
          r"{.+}\. " \
          f"Got {loss!r} instead\\."

    with pytest.raises(ValueError, match=msg):
        model.fit(x, y)
