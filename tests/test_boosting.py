import numpy
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pandas
import pytest
from sklearn.metrics import mean_squared_error

from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis, GradientBoostingSurvivalAnalysis
from sksurv.testing import assert_cindex_almost_equal
from sksurv.util import Surv


def early_stopping_monitor(i, est, locals_):
    """Returns True on the 10th iteration. """
    return i == 9


class TestGradientBoosting(object):

    @staticmethod
    def test_fit(make_whas500):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        model = GradientBoostingSurvivalAnalysis(n_estimators=100, max_depth=3, min_samples_split=10,
                                                 random_state=0)
        model.fit(whas500_data.x, whas500_data.y)

        assert model.max_features_ == 14
        assert not hasattr(model, "oob_improvement_")

        p = model.predict(whas500_data.x)

        assert_cindex_almost_equal(whas500_data.y['fstat'], whas500_data.y['lenfol'], p,
                                   (0.86272605091218779, 64826, 10309, 14, 14))

        assert (100,) == model.train_score_.shape

        with pytest.raises(ValueError, match="Number of features of the model must match the input. "
                                             "Model n_features is 14 and input n_features is 2 "):
            model.predict(whas500_data.x[:, :2])

    @staticmethod
    def test_fit_subsample(make_whas500):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        model = GradientBoostingSurvivalAnalysis(n_estimators=50, max_features=8, subsample=0.6,
                                                 random_state=0)
        model.fit(whas500_data.x, whas500_data.y)

        assert model.max_features_ == 8
        assert hasattr(model, "oob_improvement_")

        incl_mask = numpy.ones(whas500_data.x.shape[0], dtype=bool)
        incl_mask[[35, 111, 174, 206, 236, 268, 497]] = False
        x_test = whas500_data.x[incl_mask]
        y_test = whas500_data.y[incl_mask]

        p = model.predict(x_test)

        assert_cindex_almost_equal(y_test['fstat'], y_test['lenfol'], p,
                                   (0.8330510326740247, 60985, 12221, 2, 14))

        assert (50,) == model.train_score_.shape
        assert (50,) == model.oob_improvement_.shape

        with pytest.raises(ValueError, match="Number of features of the model must match the input. "
                                             "Model n_features is 14 and input n_features is 2 "):
            model.predict(whas500_data.x[:, :2])

    @staticmethod
    @pytest.mark.slow
    def test_fit_dropout(make_whas500):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        model = GradientBoostingSurvivalAnalysis(n_estimators=100, max_features=8,
                                                 learning_rate=1.0, dropout_rate=0.03,
                                                 random_state=0)
        model.fit(whas500_data.x, whas500_data.y)

        assert not hasattr(model, "oob_improvement_")
        assert model.max_features_ == 8

        p = model.predict(whas500_data.x)

        assert_cindex_almost_equal(whas500_data.y['fstat'], whas500_data.y['lenfol'], p,
                                   (0.9094333, 68343, 6806, 0, 14))

    @staticmethod
    def test_fit_int_param_as_float(make_whas500):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        # Account for https://github.com/scikit-learn/scikit-learn/pull/12344
        max_depth = 4

        model = GradientBoostingSurvivalAnalysis(
            n_estimators=100.0,
            max_depth=float(max_depth),
            min_samples_split=10.0,
            random_state=0)
        params = model.get_params()
        assert 100 == params["n_estimators"]
        assert max_depth == params["max_depth"]
        assert 10 == params["min_samples_split"]

        model.set_params(max_leaf_nodes=15.0)
        assert 15 == model.get_params()["max_leaf_nodes"]

        model.fit(whas500_data.x, whas500_data.y)
        p = model.predict(whas500_data.x)

        assert_cindex_almost_equal(whas500_data.y['fstat'], whas500_data.y['lenfol'], p,
                                   (0.90256690042449006, 67826, 7321, 2, 14))

    @staticmethod
    def test_max_features(make_whas500):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        model = GradientBoostingSurvivalAnalysis(n_estimators=10, max_features="auto", max_depth=3, random_state=0)
        model.fit(whas500_data.x, whas500_data.y)

        assert model.max_features_ == whas500_data.x.shape[1]

        model.set_params(max_features="sqrt")
        model.fit(whas500_data.x, whas500_data.y)
        assert round(abs(model.max_features_ - int(numpy.sqrt(whas500_data.x.shape[1]))), 7) == 0

        model.set_params(max_features="log2")
        model.fit(whas500_data.x, whas500_data.y)
        assert round(abs(model.max_features_ - int(numpy.log2(whas500_data.x.shape[1]))), 7) == 0

        model.set_params(max_features=0.25)
        model.fit(whas500_data.x, whas500_data.y)
        assert round(abs(model.max_features_ - int(0.25 * whas500_data.x.shape[1])), 7) == 0

        model.set_params(max_features=5)
        model.fit(whas500_data.x, whas500_data.y)
        assert round(abs(model.max_features_ - 5), 7) == 0

        model.set_params(max_features=-1)
        with pytest.raises(ValueError,
                           match=r"max_features must be in \(0, n_features\]"):
            model.fit(whas500_data.x, whas500_data.y)

        model.set_params(max_features=-1.125)
        with pytest.raises(ValueError,
                           match=r"max_features must be in \(0, 1.0\]"):
            model.fit(whas500_data.x, whas500_data.y)

        model.set_params(max_features="fail_me")
        with pytest.raises(ValueError,
                           match="Invalid value for max_features: 'fail_me'. "
                                 "Allowed string values are 'auto', 'sqrt' "
                                 "or 'log2'"):
            model.fit(whas500_data.x, whas500_data.y)

    @staticmethod
    @pytest.mark.parametrize('presort', ['auto', True, False, None])
    def test_presort(make_whas500, presort):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        model = GradientBoostingSurvivalAnalysis(n_estimators=10, presort=presort, random_state=0)

        with pytest.warns(DeprecationWarning,
                          match="The parameter 'presort' is deprecated "):
            model.fit(whas500_data.x, whas500_data.y)

    @staticmethod
    def test_ccp_alpha(make_whas500):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        est_full = GradientBoostingSurvivalAnalysis(
            n_estimators=10,
            max_leaf_nodes=20,
            random_state=1)
        est_full.fit(whas500_data.x, whas500_data.y)

        est_pruned = GradientBoostingSurvivalAnalysis(
            n_estimators=10,
            max_leaf_nodes=20,
            ccp_alpha=10.0,
            random_state=1)
        est_pruned.fit(whas500_data.x, whas500_data.y)

        tree = est_full.estimators_[0, 0].tree_
        subtree = est_pruned.estimators_[0, 0].tree_
        assert tree.node_count > subtree.node_count
        assert tree.max_depth > subtree.max_depth

    @staticmethod
    def test_negative_ccp_alpha(make_whas500):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        clf = GradientBoostingSurvivalAnalysis()
        msg = "ccp_alpha must be greater than or equal to 0"

        with pytest.raises(ValueError, match=msg):
            clf.set_params(ccp_alpha=-1.0)
            clf.fit(whas500_data.x, whas500_data.y)

    @staticmethod
    def test_fit_verbose(make_whas500):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        model = GradientBoostingSurvivalAnalysis(n_estimators=10, verbose=1, random_state=0)
        model.fit(whas500_data.x, whas500_data.y)

    @staticmethod
    def test_ipcwls_loss(make_whas500):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        model = GradientBoostingSurvivalAnalysis(loss="ipcwls", n_estimators=100, max_depth=3, random_state=0)
        model.fit(whas500_data.x, whas500_data.y)

        time_predicted = model.predict(whas500_data.x)
        time_true = whas500_data.y["lenfol"]
        event_true = whas500_data.y["fstat"]

        rmse_all = numpy.sqrt(mean_squared_error(time_true, time_predicted))
        assert round(abs(rmse_all - 590.5441693629117), 7) == 0

        rmse_uncensored = numpy.sqrt(mean_squared_error(time_true[event_true], time_predicted[event_true]))
        assert round(abs(rmse_uncensored - 392.97741487479743), 7) == 0

    @staticmethod
    def test_squared_loss(make_whas500):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        model = GradientBoostingSurvivalAnalysis(loss="squared", n_estimators=100, max_depth=3, random_state=0)
        model.fit(whas500_data.x, whas500_data.y)

        time_predicted = model.predict(whas500_data.x)
        time_true = whas500_data.y["lenfol"]
        event_true = whas500_data.y["fstat"]

        rmse_all = numpy.sqrt(mean_squared_error(time_true, time_predicted))
        assert round(abs(rmse_all - 580.23345259002951), 7) == 0

        rmse_uncensored = numpy.sqrt(mean_squared_error(time_true[event_true], time_predicted[event_true]))
        assert round(abs(rmse_uncensored - 383.10639243317951), 7) == 0

    @staticmethod
    def test_ipcw_loss_staged_predict(make_whas500):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        # Test whether staged decision function eventually gives
        # the same prediction.
        model = GradientBoostingSurvivalAnalysis(loss="ipcwls", n_estimators=100, max_depth=3, random_state=0)
        model.fit(whas500_data.x, whas500_data.y)

        y_pred = model.predict(whas500_data.x)

        # test if prediction for last stage equals ``predict``
        for y in model.staged_predict(whas500_data.x):
            assert y.shape == y_pred.shape

        assert_array_equal(y_pred, y)

        model.set_params(dropout_rate=0.03)
        model.fit(whas500_data.x, whas500_data.y)

        y_pred = model.predict(whas500_data.x)

        # test if prediction for last stage equals ``predict``
        for y in model.staged_predict(whas500_data.x):
            assert y.shape == y_pred.shape

        assert_array_equal(y_pred, y)

    @staticmethod
    def test_squared_loss_staged_predict(make_whas500):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        # Test whether staged decision function eventually gives
        # the same prediction.
        model = GradientBoostingSurvivalAnalysis(loss="squared", n_estimators=100, max_depth=3, random_state=0)
        model.fit(whas500_data.x, whas500_data.y)

        y_pred = model.predict(whas500_data.x)

        # test if prediction for last stage equals ``predict``
        for y in model.staged_predict(whas500_data.x):
            assert y.shape == y_pred.shape

        assert_array_equal(y_pred, y)

        model.set_params(dropout_rate=0.03)
        model.fit(whas500_data.x, whas500_data.y)

        y_pred = model.predict(whas500_data.x)

        # test if prediction for last stage equals ``predict``
        for y in model.staged_predict(whas500_data.x):
            assert y.shape == y_pred.shape

        assert_array_equal(y_pred, y)

    @staticmethod
    def test_monitor_early_stopping(make_whas500):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        est = GradientBoostingSurvivalAnalysis(loss="ipcwls", n_estimators=50, max_depth=1,
                                               subsample=0.5,
                                               random_state=0)
        est.fit(whas500_data.x, whas500_data.y, monitor=early_stopping_monitor)

        assert est.n_estimators == 50  # this is not altered
        assert est.estimators_.shape[0] == 10
        assert est.train_score_.shape[0] == 10
        assert est.oob_improvement_.shape[0] == 10


class TestSparseGradientBoosting(object):

    @staticmethod
    @pytest.mark.parametrize('loss', ['coxph', 'squared', 'ipcwls'])
    def test_fit(whas500_sparse_data, loss):
        model = GradientBoostingSurvivalAnalysis(loss=loss, n_estimators=100, max_depth=1, min_samples_split=10,
                                                 subsample=0.5, random_state=0)
        model.fit(whas500_sparse_data.x_sparse, whas500_sparse_data.y)

        assert model.estimators_.shape[0] == 100
        assert model.train_score_.shape == (100,)
        assert model.oob_improvement_.shape == (100,)

        sparse_predict = model.predict(whas500_sparse_data.x_dense)

        model.fit(whas500_sparse_data.x_sparse, whas500_sparse_data.y)
        dense_predict = model.predict(whas500_sparse_data.x_dense)

        assert_array_almost_equal(sparse_predict, dense_predict)

    @staticmethod
    @pytest.mark.parametrize('loss', ['coxph', 'squared', 'ipcwls'])
    @pytest.mark.slow
    def test_dropout(whas500_sparse_data, loss):
        model = GradientBoostingSurvivalAnalysis(loss=loss, n_estimators=100, max_depth=1, min_samples_split=10,
                                                 dropout_rate=0.03, random_state=0)
        model.fit(whas500_sparse_data.x_sparse, whas500_sparse_data.y)

        assert model.estimators_.shape[0] == 100
        assert model.train_score_.shape == (100,)

        sparse_predict = model.predict(whas500_sparse_data.x_dense)

        model.fit(whas500_sparse_data.x_dense, whas500_sparse_data.y)
        dense_predict = model.predict(whas500_sparse_data.x_dense)

        assert_array_almost_equal(sparse_predict, dense_predict)


class TestComponentwiseGradientBoosting(object):

    @staticmethod
    def test_fit(make_whas500):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        model = ComponentwiseGradientBoostingSurvivalAnalysis(n_estimators=100)
        model.fit(whas500_data.x, whas500_data.y)
        p = model.predict(whas500_data.x)

        assert_cindex_almost_equal(whas500_data.y['fstat'], whas500_data.y['lenfol'], p,
                                   (0.7755659, 58283, 16866, 0, 14))

        expected_coef = pandas.Series(numpy.zeros(15, dtype=float), index=whas500_data.names)
        expected_coef['age'] = 0.040919
        expected_coef['hr'] = 0.004977
        expected_coef['diasbp'] = -0.003407
        expected_coef['bmi'] = -0.017938
        expected_coef['sho'] = 0.429904
        expected_coef['chf'] = 0.508211

        assert_array_almost_equal(expected_coef.values, model.coef_)

        assert (100,) == model.train_score_.shape

        with pytest.raises(ValueError, match='Dimensions of X are inconsistent with training data: '
                                             'expected 14 features, but got 2'):
            model.predict(whas500_data.x[:, :2])

    @staticmethod
    def test_fit_subsample(make_whas500):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        model = ComponentwiseGradientBoostingSurvivalAnalysis(n_estimators=100, subsample=0.6, random_state=0)
        model.fit(whas500_data.x, whas500_data.y)
        p = model.predict(whas500_data.x)

        assert_cindex_almost_equal(whas500_data.y['fstat'], whas500_data.y['lenfol'], p,
                                   (0.7750602, 58245, 16904, 0, 14))

        expected_coef = pandas.Series(numpy.zeros(15, dtype=float), index=whas500_data.names)
        expected_coef['age'] = 0.041299
        expected_coef['hr'] = 0.00487
        expected_coef['diasbp'] = -0.003381
        expected_coef['bmi'] = -0.017018
        expected_coef['sho'] = 0.433685
        expected_coef['chf'] = 0.510277

        assert_array_almost_equal(expected_coef.values, model.coef_)

        assert (100,) == model.train_score_.shape
        assert (100,) == model.oob_improvement_.shape

        with pytest.raises(ValueError, match='Dimensions of X are inconsistent with training data: '
                                             'expected 14 features, but got 2'):
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

        expected_coef = pandas.Series(numpy.zeros(15, dtype=float), index=whas500_data.names)
        expected_coef['age'] = 0.275537
        expected_coef['hr'] = 0.040048
        expected_coef['diasbp'] = -0.029998
        expected_coef['bmi'] = -0.138909
        expected_coef['sho'] = 3.318941
        expected_coef['chf'] = 2.851386
        expected_coef['mitype'] = -0.075817

        assert_array_almost_equal(expected_coef.values, model.coef_)

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

        time_predicted = model.predict(whas500_data.x)
        time_true = whas500_data.y["lenfol"]
        event_true = whas500_data.y["fstat"]

        rmse_all = numpy.sqrt(mean_squared_error(time_true, time_predicted))
        assert round(abs(rmse_all - 806.283308322), 7) == 0

        rmse_uncensored = numpy.sqrt(mean_squared_error(time_true[event_true], time_predicted[event_true]))
        assert round(abs(rmse_uncensored - 542.884585289), 7) == 0

    @staticmethod
    def test_squared_loss(make_whas500):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        model = ComponentwiseGradientBoostingSurvivalAnalysis(loss="squared", n_estimators=100, random_state=0)
        model.fit(whas500_data.x, whas500_data.y)

        time_predicted = model.predict(whas500_data.x)
        time_true = whas500_data.y["lenfol"]
        event_true = whas500_data.y["fstat"]

        rmse_all = numpy.sqrt(mean_squared_error(time_true, time_predicted))
        assert round(abs(rmse_all - 793.6256945839657), 7) == 0

        rmse_uncensored = numpy.sqrt(mean_squared_error(time_true[event_true], time_predicted[event_true]))
        assert round(abs(rmse_uncensored - 542.83358120153525), 7) == 0


@pytest.fixture(params=[GradientBoostingSurvivalAnalysis, ComponentwiseGradientBoostingSurvivalAnalysis])
def sample_gb_class(request):
    x = numpy.arange(100).reshape(5, 20)
    y = Surv.from_arrays([False, False, True, True, False], [12, 14, 6, 9, 1])

    return request.param, x, y


def test_param_n_estimators(sample_gb_class):
    est_cls, x, y = sample_gb_class
    model = est_cls(n_estimators=0)

    with pytest.raises(ValueError, match="n_estimators must be greater than 0 but was 0"):
        model.fit(x, y)

    model.set_params(n_estimators=-1)
    with pytest.raises(ValueError, match="n_estimators must be greater than 0 but was -1"):
        model.fit(x, y)


def test_param_learning_rate(sample_gb_class):
    est_cls, x, y = sample_gb_class
    model = est_cls(learning_rate=0)

    with pytest.raises(ValueError, match="learning_rate must be within ]0; 1] but was 0"):
        model.fit(x, y)

    model.set_params(learning_rate=1.2)
    with pytest.raises(ValueError, match="learning_rate must be within ]0; 1] but was 1.2"):
        model.fit(x, y)


def test_param_subsample(sample_gb_class):
    est_cls, x, y = sample_gb_class
    model = est_cls(subsample=0)

    with pytest.raises(ValueError, match="subsample must be in ]0; 1] but was 0"):
        model.fit(x, y)

    model.set_params(subsample=1.2)
    with pytest.raises(ValueError, match="subsample must be in ]0; 1] but was 1.2"):
        model.fit(x, y)


def test_param_dropout_rate(sample_gb_class):
    est_cls, x, y = sample_gb_class
    model = est_cls(dropout_rate=-0.1)

    with pytest.raises(ValueError, match=r"dropout_rate must be within \[0; 1\[, but was -0.1"):
        model.fit(x, y)

    model.set_params(dropout_rate=1.2)
    with pytest.raises(ValueError, match=r"dropout_rate must be within \[0; 1\[, but was 1.2"):
        model.fit(x, y)


def test_param_sample_weight(sample_gb_class):
    est_cls, x, y = sample_gb_class
    model = est_cls()

    with pytest.raises(ValueError, match=r"Found input variables with inconsistent numbers of samples: \[5, 3\]"):
        model.fit(x, y, [2, 3, 4])

    model.set_params(dropout_rate=1.2)
    with pytest.raises(ValueError, match=r"Found input variables with inconsistent numbers of samples: \[5, 8\]"):
        model.fit(x, y, [2, 4, 5, 6, 7, 1, 2, 7])


def test_param_loss(sample_gb_class):
    est_cls, x, y = sample_gb_class
    model = est_cls(loss="")

    with pytest.raises(ValueError, match="Loss '' not supported"):
        model.fit(x, y)

    model.set_params(loss="unknown")
    with pytest.raises(ValueError, match="Loss 'unknown' not supported"):
        model.fit(x, y)

    model.set_params(loss=None)
    with pytest.raises(ValueError, match="Loss None not supported"):
        model.fit(x, y)
