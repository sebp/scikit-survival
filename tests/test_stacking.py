from os.path import join, dirname

import numpy
from numpy.testing import TestCase, run_module_suite, assert_array_almost_equal
from sklearn.base import BaseEstimator
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.svm import SVC

from sksurv.column import categorical_to_numeric
from sksurv.datasets import load_whas500
from sksurv.meta import Stacking, MeanEstimator
from sksurv.metrics import concordance_index_censored
from sksurv.svm import FastSurvivalSVM
from sksurv.linear_model import CoxPHSurvivalAnalysis


class _NoFitEstimator(BaseEstimator):
    pass


class _NoPredictDummy(BaseEstimator):
    def fit(self, X, y):
        pass


class _PredictDummy(BaseEstimator):
    def fit(self, X, y):
        pass

    def predict(self, X):
        pass


class _PredictProbaDummy(BaseEstimator):
    def fit(self, X, y):
        pass

    def predict_proba(self, X):
        pass


class TestStackingClassifier(TestCase):
    def test_base_no_fit(self):
        self.assertRaisesRegex(TypeError,
                               "All base estimators should implement fit and predict/predict_proba (.+) doesn't",
                               Stacking, _PredictDummy, [('m1', _NoFitEstimator)])

    def test_base_no_predict(self):
        self.assertRaisesRegex(TypeError,
                               "All base estimators should implement fit and predict/predict_proba (.+) doesn't",
                               Stacking, _PredictDummy, [('m1', _NoPredictDummy)])

    def test_meta_no_fit(self):
        self.assertRaisesRegex(TypeError,
                               "meta estimator should implement fit (.+) doesn't",
                               Stacking, _NoFitEstimator, [('m1', _PredictDummy)])

    def test_names_not_unique(self):
        self.assertRaisesRegex(ValueError,
                               "Names provided are not unique: \('m1', 'm2', 'm1'\)",
                               Stacking, _NoFitEstimator,
                               [('m1', _PredictDummy), ('m2', _PredictDummy), ('m1', _PredictDummy)])

    def test_fit(self):
        data = load_iris()
        x = data["data"]
        y = data["target"]

        meta = Stacking(LogisticRegression(), [('tree', DecisionTreeClassifier(max_depth=1, random_state=0)),
                                               ('svm', SVC(probability=True, random_state=0))])
        self.assertEqual(2, len(meta))
        meta.fit(x, y)

        p = meta._predict_estimators(x)
        self.assertTupleEqual((x.shape[0], 3 * 2), p.shape)

        self.assertTupleEqual((3, 3 * 2), meta.meta_estimator.coef_.shape)

    def test_fit_sample_weights(self):
        data = load_iris()
        x = data["data"]
        y = data["target"]

        meta = Stacking(LogisticRegression(), [('tree', DecisionTreeClassifier(max_depth=1, random_state=0)),
                                               ('svm', SVC(probability=True, random_state=0))])

        sample_weight = numpy.random.RandomState(0).uniform(size=x.shape[0])
        meta.fit(x, y, tree__sample_weight=sample_weight, svm__sample_weight=sample_weight)

    def test_set_params(self):
        meta = Stacking(LogisticRegression(), [('tree', DecisionTreeClassifier(max_depth=1, random_state=0)),
                                               ('svm', SVC(probability=True, random_state=0))],
                        probabilities=True)
        self.assertEqual(2, len(meta))
        meta.set_params(tree__min_samples_split=7, svm__C=0.05)

        self.assertEqual(7, meta.get_params()["tree__min_samples_split"])
        self.assertEqual(0.05, meta.get_params()["svm__C"])
        self.assertIsInstance(meta.get_params()["meta_estimator"], LogisticRegression)
        self.assertTrue(meta.get_params()["probabilities"])

        meta.set_params(meta_estimator=DecisionTreeClassifier(), probabilities=False)
        self.assertIsInstance(meta.get_params()["meta_estimator"], DecisionTreeClassifier)
        self.assertFalse(meta.get_params()["probabilities"])

    def test_predict(self):
        data = load_iris()
        x = data["data"]
        y = data["target"]

        meta = Stacking(LogisticRegression(multi_class='multinomial', solver='lbfgs'),
                        [('tree', DecisionTreeClassifier(max_depth=1, random_state=0)),
                         ('svm', SVC(probability=True, random_state=0))])
        self.assertEqual(2, len(meta))
        meta.fit(x, y)
        p = meta.predict(x)
        acc = accuracy_score(y, p)

        self.assertEqual(0.98, acc)

    def test_predict_proba(self):
        data = load_iris()
        x = data["data"]
        y = data["target"]

        meta = Stacking(LogisticRegression(multi_class='multinomial', solver='lbfgs'),
                        [('tree', DecisionTreeClassifier(max_depth=1, random_state=0)),
                         ('svm', SVC(probability=True, random_state=0))])
        meta.fit(x, y)
        p = meta.predict_proba(x)

        scores = numpy.empty(3)
        for i, c in enumerate(meta.meta_estimator.classes_):
            scores[i] = roc_auc_score(numpy.asarray(y == c, dtype=int), p[:, i])

        assert_array_almost_equal(numpy.array([1.0, 0.9986, 0.9986]), scores)

    def test_predict_log_proba(self):
        data = load_iris()
        x = data["data"]
        y = data["target"]

        meta = Stacking(LogisticRegression(multi_class='multinomial', solver='lbfgs'),
                        [('tree', DecisionTreeClassifier(max_depth=1, random_state=0)),
                         ('svm', SVC(probability=True, random_state=0))])
        meta.fit(x, y)
        p = meta.predict_log_proba(x)

        scores = numpy.empty(3)
        for i, c in enumerate(meta.meta_estimator.classes_):
            scores[i] = roc_auc_score(numpy.asarray(y == c, dtype=int), p[:, i])

        assert_array_almost_equal(numpy.array([1.0, 0.9986, 0.9986]), scores)


class TestStackingSurvivalAnalysis(TestCase):
    def setUp(self):
        x, self.y = load_whas500()
        self.x = categorical_to_numeric(x)

    def test_fit(self):
        meta = Stacking(MeanEstimator(),
                        [('coxph', CoxPHSurvivalAnalysis()),
                         ('svm', FastSurvivalSVM(random_state=0))],
                        probabilities=False)
        self.assertEqual(2, len(meta))
        meta.fit(self.x.values, self.y)

        p = meta._predict_estimators(self.x.values)
        self.assertTupleEqual((self.x.shape[0], 2), p.shape)

    def test_set_params(self):
        meta = Stacking(_PredictDummy(),
                        [('coxph', CoxPHSurvivalAnalysis()),
                         ('svm', FastSurvivalSVM(random_state=0))],
                        probabilities=False)

        meta.set_params(coxph__alpha=1.0, svm__alpha=0.4132)

        self.assertEqual(1.0, meta.get_params()["coxph__alpha"])
        self.assertEqual(0.4132, meta.get_params()["svm__alpha"])

    def test_predict(self):
        meta = Stacking(MeanEstimator(),
                        [('coxph', CoxPHSurvivalAnalysis()),
                         ('svm', FastSurvivalSVM(random_state=0))],
                        probabilities=False)

        meta.fit(self.x.values, self.y)

        # result is different if randomForestSRC has not been compiled with OpenMP support
        p = meta.predict(self.x.values)
        actual_cindex = concordance_index_censored(self.y['fstat'], self.y['lenfol'], p)

        expected_cindex = numpy.array([0.7848807, 58983, 16166, 0, 119])
        assert_array_almost_equal(expected_cindex, actual_cindex)

    def test_predict_proba(self):
        meta = Stacking(_PredictDummy(),
                        [('coxph', CoxPHSurvivalAnalysis()),
                         ('svm', FastSurvivalSVM(random_state=0))],
                        probabilities=False)

        self.assertRaisesRegex(AttributeError,
                               "'_PredictDummy' object has no attribute 'predict_proba'",
                               getattr, meta, "predict_proba")


if __name__ == '__main__':
    run_module_suite()
