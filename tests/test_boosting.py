from os.path import join, dirname

import numpy
from numpy.testing import TestCase, run_module_suite, assert_array_equal, assert_array_almost_equal
import pandas
from scipy.sparse import coo_matrix
from sklearn.metrics import mean_squared_error

from sksurv.datasets import load_whas500
from sksurv.metrics import concordance_index_censored
from sksurv import column
from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis, GradientBoostingSurvivalAnalysis


def early_stopping_monitor(i, est, locals):
    """Returns True on the 10th iteration. """
    if i == 9:
        return True
    else:
        return False


class TestGradientBoosting(TestCase):
    def setUp(self):
        x, self.y = load_whas500()

        x = column.categorical_to_numeric(column.standardize(x, with_std=False))
        self.x = x.values
        self.columns = x.columns.tolist()

    def test_fit(self):
        model = GradientBoostingSurvivalAnalysis(n_estimators=100, max_depth=3, min_samples_split=10,
                                                 random_state=0)
        model.fit(self.x, self.y)

        self.assertEquals(model.max_features_, 14)
        self.assertFalse(hasattr(model, "oob_improvement_"))

        p = model.predict(self.x)

        expected_cindex = numpy.array([0.86272605091218779, 64826, 10309, 14, 119])
        result = concordance_index_censored(self.y['fstat'], self.y['lenfol'], p)
        assert_array_almost_equal(expected_cindex, numpy.array(result))

        self.assertTupleEqual((100,), model.train_score_.shape)

        self.assertRaisesRegex(ValueError, "Number of features of the model must match the input. "
                                           "Model n_features is 14 and input n_features is 2 ",
                               model.predict, self.x[:, :2])

    def test_fit_subsample(self):
        model = GradientBoostingSurvivalAnalysis(n_estimators=100, max_features=8, subsample=0.6,
                                                 random_state=0)
        model.fit(self.x, self.y)

        self.assertEquals(model.max_features_, 8)
        self.assertTrue(hasattr(model, "oob_improvement_"))

        p = model.predict(self.x)

        expected_cindex = numpy.array([0.8610760, 64709, 10440, 0, 119])
        result = concordance_index_censored(self.y['fstat'], self.y['lenfol'], p)
        assert_array_almost_equal(expected_cindex, numpy.array(result))

        self.assertTupleEqual((100,), model.train_score_.shape)
        self.assertTupleEqual((100,), model.oob_improvement_.shape)

        self.assertRaisesRegex(ValueError, "Number of features of the model must match the input. "
                                           "Model n_features is 14 and input n_features is 2 ",
                               model.predict, self.x[:, :2])

    def test_fit_dropout(self):
        model = GradientBoostingSurvivalAnalysis(n_estimators=100, max_features=8,
                                                 learning_rate=1.0, dropout_rate=0.03,
                                                 random_state=0)
        model.fit(self.x, self.y)

        self.assertFalse(hasattr(model, "oob_improvement_"))
        self.assertEquals(model.max_features_, 8)

        p = model.predict(self.x)

        expected_cindex = numpy.array([0.9094333, 68343, 6806, 0, 119])
        result = concordance_index_censored(self.y['fstat'], self.y['lenfol'], p)
        assert_array_almost_equal(expected_cindex, numpy.array(result))

    def test_fit_int_param_as_float(self):
        model = GradientBoostingSurvivalAnalysis(n_estimators=100.0, max_depth=3.0, min_samples_split=10.0,
                                                 random_state=0)
        params = model.get_params()
        self.assertEqual(100, params["n_estimators"])
        self.assertEqual(3, params["max_depth"])
        self.assertEqual(10, params["min_samples_split"])

        model.set_params(max_leaf_nodes=15.0)
        self.assertEqual(15, model.get_params()["max_leaf_nodes"])

        model.fit(self.x, self.y)
        p = model.predict(self.x)

        expected_cindex = numpy.array([0.90256690042449006, 67826, 7321, 2, 119])
        result = concordance_index_censored(self.y['fstat'], self.y['lenfol'], p)
        assert_array_almost_equal(expected_cindex, numpy.array(result))

    def test_max_features(self):
        model = GradientBoostingSurvivalAnalysis(n_estimators=10, max_features="auto", max_depth=3, random_state=0)
        model.fit(self.x, self.y)

        self.assertEqual(model.max_features_, self.x.shape[1])

        model.set_params(max_features="sqrt")
        model.fit(self.x, self.y)
        self.assertAlmostEqual(model.max_features_, int(numpy.sqrt(self.x.shape[1])))

        model.set_params(max_features="log2")
        model.fit(self.x, self.y)
        self.assertAlmostEqual(model.max_features_, int(numpy.log2(self.x.shape[1])))

        model.set_params(max_features=0.25)
        model.fit(self.x, self.y)
        self.assertAlmostEqual(model.max_features_, int(0.25 * self.x.shape[1]))

        model.set_params(max_features=5)
        model.fit(self.x, self.y)
        self.assertAlmostEqual(model.max_features_, 5)

        model.set_params(max_features=-1)
        self.assertRaisesRegex(ValueError,
                               "max_features must be in \(0, n_features\]",
                               model.fit, self.x, self.y)

        model.set_params(max_features=-1.125)
        self.assertRaisesRegex(ValueError,
                               "max_features must be in \(0, 1.0\]",
                               model.fit, self.x, self.y)

        model.set_params(max_features="fail_me")
        self.assertRaisesRegex(ValueError,
                               "Invalid value for max_features: 'fail_me'. "
                               "Allowed string values are 'auto', 'sqrt' "
                               "or 'log2'",
                               model.fit, self.x, self.y)

    def test_fit_verbose(self):
        model = GradientBoostingSurvivalAnalysis(n_estimators=10, verbose=1, random_state=0)
        model.fit(self.x, self.y)

    def test_ipcwls_loss(self):
        model = GradientBoostingSurvivalAnalysis(loss="ipcwls", n_estimators=100, max_depth=3, random_state=0)
        model.fit(self.x, self.y)

        time_predicted = model.predict(self.x)
        time_true = self.y["lenfol"]
        event_true = self.y["fstat"]

        rmse_all = numpy.sqrt(mean_squared_error(time_true, time_predicted))
        self.assertAlmostEqual(rmse_all, 590.5441693629117)

        rmse_uncensored = numpy.sqrt(mean_squared_error(time_true[event_true], time_predicted[event_true]))
        self.assertAlmostEqual(rmse_uncensored, 392.97741487479743)

    def test_squared_loss(self):
        model = GradientBoostingSurvivalAnalysis(loss="squared", n_estimators=100, max_depth=3, random_state=0)
        model.fit(self.x, self.y)

        time_predicted = model.predict(self.x)
        time_true = self.y["lenfol"]
        event_true = self.y["fstat"]

        rmse_all = numpy.sqrt(mean_squared_error(time_true, time_predicted))
        self.assertAlmostEqual(rmse_all, 580.23345259002951)

        rmse_uncensored = numpy.sqrt(mean_squared_error(time_true[event_true], time_predicted[event_true]))
        self.assertAlmostEqual(rmse_uncensored, 383.10639243317951)

    def test_ipcw_loss_staged_predict(self):
        # Test whether staged decision function eventually gives
        # the same prediction.
        model = GradientBoostingSurvivalAnalysis(loss="ipcwls", n_estimators=100, max_depth=3, random_state=0)
        model.fit(self.x, self.y)

        y_pred = model.predict(self.x)

        # test if prediction for last stage equals ``predict``
        for y in model.staged_predict(self.x):
            self.assertTupleEqual(y.shape, y_pred.shape)

        assert_array_equal(y_pred, y)

        model.set_params(dropout_rate=0.03)
        model.fit(self.x, self.y)

        y_pred = model.predict(self.x)

        # test if prediction for last stage equals ``predict``
        for y in model.staged_predict(self.x):
            self.assertTupleEqual(y.shape, y_pred.shape)

        assert_array_equal(y_pred, y)

    def test_squared_loss_staged_predict(self):
        # Test whether staged decision function eventually gives
        # the same prediction.
        model = GradientBoostingSurvivalAnalysis(loss="squared", n_estimators=100, max_depth=3, random_state=0)
        model.fit(self.x, self.y)

        y_pred = model.predict(self.x)

        # test if prediction for last stage equals ``predict``
        for y in model.staged_predict(self.x):
            self.assertTupleEqual(y.shape, y_pred.shape)

        assert_array_equal(y_pred, y)

        model.set_params(dropout_rate=0.03)
        model.fit(self.x, self.y)

        y_pred = model.predict(self.x)

        # test if prediction for last stage equals ``predict``
        for y in model.staged_predict(self.x):
            self.assertTupleEqual(y.shape, y_pred.shape)

        assert_array_equal(y_pred, y)

    def test_monitor_early_stopping(self):
        est = GradientBoostingSurvivalAnalysis(loss="ipcwls", n_estimators=50, max_depth=1,
                                               subsample=0.5,
                                               random_state=0)
        est.fit(self.x, self.y, monitor=early_stopping_monitor)

        self.assertEqual(est.n_estimators, 50)  # this is not altered
        self.assertEqual(est.estimators_.shape[0], 10)
        self.assertEqual(est.train_score_.shape[0], 10)
        self.assertEqual(est.oob_improvement_.shape[0], 10)


class TestSparseGradientBoosting(TestCase):

    def setUp(self):
        x, self.y = load_whas500()
        self.x_dense = column.categorical_to_numeric(x.select_dtypes(exclude=[numpy.float_]))

        data = []
        index_i = []
        index_j = []
        for j, (_, col) in enumerate(self.x_dense.iteritems()):
            idx = numpy.flatnonzero(col.values)
            data.extend([1] * len(idx))
            index_i.extend(idx)
            index_j.extend([j] * len(idx))

        self.x_sparse = coo_matrix((data, (index_i, index_j)))
        assert_array_equal(self.x_dense.values, self.x_sparse.toarray())

    def test_fit(self):
        for loss in ('coxph', 'squared', 'ipcwls'):
            model = GradientBoostingSurvivalAnalysis(loss=loss, n_estimators=100, max_depth=1, min_samples_split=10,
                                                     subsample=0.5, random_state=0)
            model.fit(self.x_sparse, self.y)

            self.assertEqual(model.estimators_.shape[0], 100)
            self.assertTupleEqual(model.train_score_.shape, (100,))
            self.assertTupleEqual(model.oob_improvement_.shape, (100,))

            sparse_predict = model.predict(self.x_dense)

            model.fit(self.x_dense, self.y)
            dense_predict = model.predict(self.x_dense)

            assert_array_almost_equal(sparse_predict, dense_predict)

    def test_dropout(self):
        for loss in ('coxph', 'squared', 'ipcwls'):
            model = GradientBoostingSurvivalAnalysis(loss=loss, n_estimators=100, max_depth=1, min_samples_split=10,
                                                     dropout_rate=0.03, random_state=0)
            model.fit(self.x_sparse, self.y)

            self.assertEqual(model.estimators_.shape[0], 100)
            self.assertTupleEqual(model.train_score_.shape, (100,))

            sparse_predict = model.predict(self.x_dense)

            model.fit(self.x_dense, self.y)
            dense_predict = model.predict(self.x_dense)

            assert_array_almost_equal(sparse_predict, dense_predict)


class TestComponentwiseGradientBoosting(TestCase):
    def setUp(self):
        x, self.y = load_whas500()

        x = column.categorical_to_numeric(column.standardize(x, with_std=False))
        self.x = x.values
        self.columns = x.columns.tolist()

    def test_fit(self):
        model = ComponentwiseGradientBoostingSurvivalAnalysis(n_estimators=100)
        model.fit(self.x, self.y)
        p = model.predict(self.x)

        result = concordance_index_censored(self.y['fstat'], self.y['lenfol'], p)
        expected_cindex = numpy.array([0.7755659, 58283, 16866, 0, 119])
        assert_array_almost_equal(expected_cindex, numpy.array(result))

        coef_index = ['(Intercept)'] + self.columns

        expected_coef = pandas.Series(numpy.zeros(15, dtype=float), index=coef_index)
        expected_coef['age'] = 0.040919
        expected_coef['hr'] = 0.004977
        expected_coef['diasbp'] = -0.003407
        expected_coef['bmi'] = -0.017938
        expected_coef['sho'] = 0.429904
        expected_coef['chf'] = 0.508211

        assert_array_almost_equal(expected_coef.values, model.coef_)

        self.assertTupleEqual((100,), model.train_score_.shape)

        self.assertRaisesRegex(ValueError, 'Dimensions of X are inconsistent with training data: '
                                           'expected 14 features, but got 2',
                               model.predict, self.x[:, :2])

    def test_fit_subsample(self):
        model = ComponentwiseGradientBoostingSurvivalAnalysis(n_estimators=100, subsample=0.6, random_state=0)
        model.fit(self.x, self.y)
        p = model.predict(self.x)

        result = concordance_index_censored(self.y['fstat'], self.y['lenfol'], p)
        expected_cindex = numpy.array([0.7750602, 58245, 16904, 0, 119])
        assert_array_almost_equal(expected_cindex, numpy.array(result))

        coef_index = ['(Intercept)'] + self.columns

        expected_coef = pandas.Series(numpy.zeros(15, dtype=float), index=coef_index)
        expected_coef['age'] = 0.041299
        expected_coef['hr'] = 0.00487
        expected_coef['diasbp'] = -0.003381
        expected_coef['bmi'] = -0.017018
        expected_coef['sho'] = 0.433685
        expected_coef['chf'] = 0.510277

        assert_array_almost_equal(expected_coef.values, model.coef_)

        self.assertTupleEqual((100,), model.train_score_.shape)
        self.assertTupleEqual((100,), model.oob_improvement_.shape)

        self.assertRaisesRegex(ValueError, 'Dimensions of X are inconsistent with training data: '
                                           'expected 14 features, but got 2',
                               model.predict, self.x[:, :2])

    def test_fit_dropout(self):
        model = ComponentwiseGradientBoostingSurvivalAnalysis(n_estimators=100, learning_rate=1.0,
                                                              dropout_rate=0.03, random_state=0)
        model.fit(self.x, self.y)
        p = model.predict(self.x)

        result = concordance_index_censored(self.y['fstat'], self.y['lenfol'], p)
        expected_cindex = numpy.array([0.7772425, 58409, 16740, 0, 119])
        assert_array_almost_equal(expected_cindex, numpy.array(result))

        coef_index = ['(Intercept)'] + self.columns

        expected_coef = pandas.Series(numpy.zeros(15, dtype=float), index=coef_index)
        expected_coef['age'] = 0.275537
        expected_coef['hr'] = 0.040048
        expected_coef['diasbp'] = -0.029998
        expected_coef['bmi'] = -0.138909
        expected_coef['sho'] = 3.318941
        expected_coef['chf'] = 2.851386
        expected_coef['mitype'] = -0.075817

        assert_array_almost_equal(expected_coef.values, model.coef_)

    def test_feature_importances(self):
        model = ComponentwiseGradientBoostingSurvivalAnalysis(n_estimators=100, random_state=0)
        model.fit(self.x, self.y)

        self.assertEqual(self.x.shape[1] + 1, len(model.feature_importances_))

    def test_fit_verbose(self):
        model = ComponentwiseGradientBoostingSurvivalAnalysis(n_estimators=10, verbose=1, random_state=0)
        model.fit(self.x, self.y)

    def test_ipcwls_loss(self):
        model = ComponentwiseGradientBoostingSurvivalAnalysis(loss="ipcwls", n_estimators=100, random_state=0)
        model.fit(self.x, self.y)

        time_predicted = model.predict(self.x)
        time_true = self.y["lenfol"]
        event_true = self.y["fstat"]

        rmse_all = numpy.sqrt(mean_squared_error(time_true, time_predicted))
        self.assertAlmostEqual(rmse_all, 806.283308322)

        rmse_uncensored = numpy.sqrt(mean_squared_error(time_true[event_true], time_predicted[event_true]))
        self.assertAlmostEqual(rmse_uncensored, 542.884585289)

    def test_squared_loss(self):
        model = ComponentwiseGradientBoostingSurvivalAnalysis(loss="squared", n_estimators=100, random_state=0)
        model.fit(self.x, self.y)

        time_predicted = model.predict(self.x)
        time_true = self.y["lenfol"]
        event_true = self.y["fstat"]

        rmse_all = numpy.sqrt(mean_squared_error(time_true, time_predicted))
        self.assertAlmostEqual(rmse_all, 793.6256945839657)

        rmse_uncensored = numpy.sqrt(mean_squared_error(time_true[event_true], time_predicted[event_true]))
        self.assertAlmostEqual(rmse_uncensored, 542.83358120153525)


class ExceptionCases:
    def test_n_estimators(self):
        model = self.ESTIMATOR(n_estimators=0)

        x = numpy.arange(100).reshape(5, 20)
        y = numpy.empty(dtype=[('event', bool), ('time', float)], shape=5)
        y['time'] = [12, 14, 6, 9, 1]
        y['event'] = [False, False, True, True, False]

        self.assertRaisesRegex(ValueError, "n_estimators must be greater than 0 but was 0",
                               model.fit, x, y)

        model.set_params(n_estimators=-1)
        self.assertRaisesRegex(ValueError, "n_estimators must be greater than 0 but was -1",
                               model.fit, x, y)

    def test_learning_rate(self):
        model = self.ESTIMATOR(learning_rate=0)

        x = numpy.arange(100).reshape(5, 20)
        y = numpy.empty(dtype=[('event', bool), ('time', float)], shape=5)
        y['time'] = [12, 14, 6, 9, 1]
        y['event'] = [False, False, True, True, False]

        self.assertRaisesRegex(ValueError, "learning_rate must be within ]0; 1] but was 0",
                               model.fit, x, y)

        model.set_params(learning_rate=1.2)
        self.assertRaisesRegex(ValueError, "learning_rate must be within ]0; 1] but was 1.2",
                               model.fit, x, y)

    def test_subsample(self):
        model = self.ESTIMATOR(subsample=0)

        x = numpy.arange(100).reshape(5, 20)
        y = numpy.empty(dtype=[('event', bool), ('time', float)], shape=5)
        y['time'] = [12, 14, 6, 9, 1]
        y['event'] = [False, False, True, True, False]

        self.assertRaisesRegex(ValueError, "subsample must be in ]0; 1] but was 0",
                               model.fit, x, y)

        model.set_params(subsample=1.2)
        self.assertRaisesRegex(ValueError, "subsample must be in ]0; 1] but was 1.2",
                               model.fit, x, y)

    def test_dropout_rate(self):
        model = self.ESTIMATOR(dropout_rate=-0.1)

        x = numpy.arange(100).reshape(5, 20)
        y = numpy.empty(dtype=[('event', bool), ('time', float)], shape=5)
        y['time'] = [12, 14, 6, 9, 1]
        y['event'] = [False, False, True, True, False]

        self.assertRaisesRegex(ValueError, "dropout_rate must be within \[0; 1\[, but was -0.1",
                               model.fit, x, y)

        model.set_params(dropout_rate=1.2)
        self.assertRaisesRegex(ValueError, "dropout_rate must be within \[0; 1\[, but was 1.2",
                               model.fit, x, y)

    def test_sample_weight(self):
        model = self.ESTIMATOR()

        x = numpy.arange(100).reshape(5, 20)
        y = numpy.empty(dtype=[('event', bool), ('time', float)], shape=5)
        y['time'] = [12, 14, 6, 9, 1]
        y['event'] = [False, False, True, True, False]

        self.assertRaisesRegex(ValueError, "Found input variables with inconsistent numbers of samples: \[5, 3\]",
                               model.fit, x, y, [2, 3, 4])

        model.set_params(dropout_rate=1.2)
        self.assertRaisesRegex(ValueError, "Found input variables with inconsistent numbers of samples: \[5, 8\]",
                               model.fit, x, y, [2, 4, 5, 6, 7, 1, 2, 7])



class TestGradientBoostingSurvivalAnalysisExceptions(TestCase, ExceptionCases):
    ESTIMATOR = GradientBoostingSurvivalAnalysis


class TestComponentwiseGradientBoostingSurvivalAnalysisExceptions(TestCase, ExceptionCases):
    ESTIMATOR = ComponentwiseGradientBoostingSurvivalAnalysis


if __name__ == '__main__':
    run_module_suite()
