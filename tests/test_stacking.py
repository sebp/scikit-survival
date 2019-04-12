import numpy
from numpy.testing import assert_array_almost_equal
import pytest
from sklearn.base import BaseEstimator
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.svm import SVC

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.meta import Stacking, MeanEstimator
from sksurv.testing import assert_cindex_almost_equal
from sksurv.svm import FastSurvivalSVM


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


class TestStackingClassifier(object):
    @staticmethod
    @pytest.mark.parametrize('estimator', [_NoFitEstimator, _NoPredictDummy])
    def test_base_estimator(estimator):
        with pytest.raises(TypeError,
                           match=r"All base estimators should implement fit and predict/predict_proba (.+) doesn't"):
            Stacking(_PredictDummy, [('m1', estimator)])

    @staticmethod
    def test_meta_no_fit():
        with pytest.raises(TypeError,
                           match=r"meta estimator should implement fit (.+) doesn't"):
            Stacking(_NoFitEstimator, [('m1', _PredictDummy)])

    @staticmethod
    def test_names_not_unique():
        with pytest.raises(ValueError,
                           match=r"Names provided are not unique: \('m1', 'm2', 'm1'\)"):
            Stacking(_NoFitEstimator,
                     [('m1', _PredictDummy), ('m2', _PredictDummy), ('m1', _PredictDummy)])

    @staticmethod
    def test_fit():
        data = load_iris()
        x = data["data"]
        y = data["target"]

        meta = Stacking(LogisticRegression(solver='liblinear', multi_class='ovr'),
                        [('tree', DecisionTreeClassifier(max_depth=1, random_state=0)),
                         ('svm', SVC(probability=True, gamma='auto', random_state=0))])
        assert 2 == len(meta)
        meta.fit(x, y)

        p = meta._predict_estimators(x)
        assert (x.shape[0], 3 * 2) == p.shape

        assert (3, 3 * 2) == meta.meta_estimator.coef_.shape

    @staticmethod
    def test_fit_sample_weights():
        data = load_iris()
        x = data["data"]
        y = data["target"]

        meta = Stacking(LogisticRegression(solver='liblinear', multi_class='ovr'),
                        [('tree', DecisionTreeClassifier(max_depth=1, random_state=0)),
                         ('svm', SVC(probability=True, gamma='auto', random_state=0))])

        sample_weight = numpy.random.RandomState(0).uniform(size=x.shape[0])
        meta.fit(x, y, tree__sample_weight=sample_weight, svm__sample_weight=sample_weight)

    @staticmethod
    def test_set_params():
        meta = Stacking(LogisticRegression(), [('tree', DecisionTreeClassifier(max_depth=1, random_state=0)),
                                               ('svm', SVC(probability=True, random_state=0))],
                        probabilities=True)
        assert 2 == len(meta)
        meta.set_params(tree__min_samples_split=7, svm__C=0.05)

        assert 7 == meta.get_params()["tree__min_samples_split"]
        assert 0.05 == meta.get_params()["svm__C"]
        assert isinstance(meta.get_params()["meta_estimator"], LogisticRegression)
        assert meta.get_params()["probabilities"]

        meta.set_params(meta_estimator=DecisionTreeClassifier(), probabilities=False)
        assert isinstance(meta.get_params()["meta_estimator"], DecisionTreeClassifier)
        assert not meta.get_params()["probabilities"]

        p = meta.get_params(deep=False)
        assert set(p.keys()) == {"meta_estimator", "base_estimators", "probabilities"}

    @staticmethod
    def test_predict():
        data = load_iris()
        x = data["data"]
        y = data["target"]

        meta = Stacking(LogisticRegression(multi_class='multinomial', solver='lbfgs'),
                        [('tree', DecisionTreeClassifier(max_depth=1, random_state=0)),
                         ('svm', SVC(probability=True, gamma='auto', random_state=0))])
        assert 2 == len(meta)
        meta.fit(x, y)
        p = meta.predict(x)
        acc = accuracy_score(y, p)

        assert acc >= 0.98

    @staticmethod
    def test_predict_proba():
        data = load_iris()
        x = data["data"]
        y = data["target"]

        meta = Stacking(LogisticRegression(multi_class='multinomial', solver='lbfgs'),
                        [('tree', DecisionTreeClassifier(max_depth=1, random_state=0)),
                         ('svm', SVC(probability=True, gamma='auto', random_state=0))])
        meta.fit(x, y)
        p = meta.predict_proba(x)

        scores = numpy.empty(3)
        for i, c in enumerate(meta.meta_estimator.classes_):
            scores[i] = roc_auc_score(numpy.asarray(y == c, dtype=int), p[:, i])

        assert_array_almost_equal(numpy.array([1.0, 0.9986, 0.9986]), scores)

    @staticmethod
    def test_predict_log_proba():
        data = load_iris()
        x = data["data"]
        y = data["target"]

        meta = Stacking(LogisticRegression(multi_class='multinomial', solver='lbfgs'),
                        [('tree', DecisionTreeClassifier(max_depth=1, random_state=0)),
                         ('svm', SVC(probability=True, gamma='auto', random_state=0))])
        meta.fit(x, y)
        p = meta.predict_log_proba(x)

        scores = numpy.empty(3)
        for i, c in enumerate(meta.meta_estimator.classes_):
            scores[i] = roc_auc_score(numpy.asarray(y == c, dtype=int), p[:, i])

        assert_array_almost_equal(numpy.array([1.0, 0.9986, 0.9986]), scores)


class TestStackingSurvivalAnalysis(object):
    @staticmethod
    def test_fit(make_whas500):
        whas500 = make_whas500(with_mean=False, with_std=False, to_numeric=True)

        meta = Stacking(MeanEstimator(),
                        [('coxph', CoxPHSurvivalAnalysis()),
                         ('svm', FastSurvivalSVM(random_state=0))],
                        probabilities=False)
        assert 2 == len(meta)
        meta.fit(whas500.x, whas500.y)

        p = meta._predict_estimators(whas500.x)
        assert (whas500.x.shape[0], 2) == p.shape

    @staticmethod
    def test_set_params():
        meta = Stacking(_PredictDummy(),
                        [('coxph', CoxPHSurvivalAnalysis()),
                         ('svm', FastSurvivalSVM(random_state=0))],
                        probabilities=False)

        meta.set_params(coxph__alpha=1.0, svm__alpha=0.4132)

        assert 1.0 == meta.get_params()["coxph__alpha"]
        assert 0.4132 == meta.get_params()["svm__alpha"]

    @staticmethod
    def test_predict(make_whas500):
        whas500 = make_whas500(with_mean=False, with_std=False, to_numeric=True)

        meta = Stacking(MeanEstimator(),
                        [('coxph', CoxPHSurvivalAnalysis()),
                         ('svm', FastSurvivalSVM(random_state=0))],
                        probabilities=False)

        meta.fit(whas500.x, whas500.y)

        # result is different if randomForestSRC has not been compiled with OpenMP support
        p = meta.predict(whas500.x)
        assert_cindex_almost_equal(whas500.y['fstat'], whas500.y['lenfol'], p,
                                   (0.7848807, 58983, 16166, 0, 14))

    @staticmethod
    def test_predict_proba():
        meta = Stacking(_PredictDummy(),
                        [('coxph', CoxPHSurvivalAnalysis()),
                         ('svm', FastSurvivalSVM(random_state=0))],
                        probabilities=False)

        with pytest.raises(AttributeError,
                           match="'_PredictDummy' object has no attribute 'predict_proba'"):
            getattr(meta, "predict_proba")
