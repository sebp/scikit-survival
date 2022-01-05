import numpy
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pandas
import pytest
from sklearn.base import BaseEstimator
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.meta import MeanEstimator, Stacking
from sksurv.svm import FastSurvivalSVM
from sksurv.testing import assert_cindex_almost_equal


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


@pytest.fixture()
def dummy_data():
    X = numpy.zeros((3, 2))
    y = numpy.zeros(3)
    return X, y


@pytest.fixture()
def iris_data_with_estimator():
    def _make_estimator(**params):
        data = load_iris()
        x = data["data"]
        y = data["target"]

        meta = Stacking(
            LogisticRegression(**params),
            [
                ("tree", DecisionTreeClassifier(max_depth=1, random_state=0)),
                ("svm", SVC(probability=True, gamma="auto", random_state=0))
            ]
        )
        return x, y, meta
    return _make_estimator


@pytest.fixture()
def whas_data_with_estimator(make_whas500):
    whas500 = make_whas500(with_mean=False, with_std=False, to_numeric=True)

    meta = Stacking(
        MeanEstimator(),
        [("coxph", CoxPHSurvivalAnalysis()), ("svm", FastSurvivalSVM(random_state=0))],
        probabilities=False
    )
    return whas500.x, whas500.y, meta


class TestStackingClassifier:
    @staticmethod
    @pytest.mark.parametrize("estimator", [_NoFitEstimator(), _NoPredictDummy()])
    def test_base_estimator(estimator, dummy_data):
        est = Stacking(_PredictDummy(), [("m1", estimator)])
        X, y = dummy_data
        with pytest.raises(TypeError,
                           match=r"All base estimators should implement fit and predict/predict_proba (.+) doesn't"):
            est.fit(X, y)

    @staticmethod
    def test_meta_no_fit(dummy_data):
        est = Stacking(_NoFitEstimator(), [("m1", _PredictDummy())])
        X, y = dummy_data
        with pytest.raises(TypeError,
                           match=r"meta estimator should implement fit (.+) doesn't"):
            est.fit(X, y)

    @staticmethod
    def test_names_not_unique(dummy_data):
        est = Stacking(
            _NoFitEstimator(),
            [("m1", _PredictDummy()), ("m2", _PredictDummy()), ("m1", _PredictDummy())]
        )
        X, y = dummy_data
        with pytest.raises(ValueError,
                           match=r"Names provided are not unique: \('m1', 'm2', 'm1'\)"):
            est.fit(X, y)

    @staticmethod
    def test_fit(iris_data_with_estimator):
        x, y, meta = iris_data_with_estimator(solver="liblinear", multi_class="ovr")
        assert 2 == len(meta)
        meta.fit(x, y)

        p = meta._predict_estimators(x)
        assert (x.shape[0], 3 * 2) == p.shape

        assert (3, 3 * 2) == meta.meta_estimator.coef_.shape

    @staticmethod
    def test_fit_sample_weights(iris_data_with_estimator):
        x, y, meta = iris_data_with_estimator(solver="liblinear", multi_class="ovr")

        sample_weight = numpy.random.RandomState(0).uniform(size=x.shape[0])
        meta.fit(x, y, tree__sample_weight=sample_weight, svm__sample_weight=sample_weight)

    @staticmethod
    def test_set_params():
        meta = Stacking(
            LogisticRegression(), [
                ("tree", DecisionTreeClassifier(max_depth=1, random_state=0)),
                ("svm", SVC(probability=True, random_state=0))
            ],
            probabilities=True
        )
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
    def test_predict(iris_data_with_estimator):
        x, y, meta = iris_data_with_estimator(multi_class="multinomial", solver="lbfgs")
        assert 2 == len(meta)
        meta.fit(x, y)
        p = meta.predict(x)
        acc = accuracy_score(y, p)

        assert acc >= 0.98

    @staticmethod
    @pytest.mark.parametrize("method", ["predict_proba", "predict_log_proba"])
    def test_predict_proba(iris_data_with_estimator, method):
        x, y, meta = iris_data_with_estimator(multi_class="multinomial", solver="lbfgs")
        meta.fit(x, y)
        p = getattr(meta, method)(x)

        scores = numpy.empty(3)
        for i, c in enumerate(meta.meta_estimator.classes_):
            scores[i] = roc_auc_score(numpy.asarray(y == c, dtype=int), p[:, i])

        assert_array_almost_equal(numpy.array([1.0, 0.9986, 0.9986]), scores)

    @staticmethod
    def test_feature_names_in():
        data = load_iris()
        x = pandas.DataFrame(data["data"], columns=data["feature_names"])
        y = data["target"]

        meta = Stacking(
            LogisticRegression(),
            [
                ("tree", DecisionTreeClassifier(max_depth=1, random_state=0)),
                ("svm", SVC(probability=True, gamma="auto", random_state=0))
            ]
        )
        meta.fit(x, y)
        assert meta.n_features_in_ == len(data["feature_names"])
        assert_array_equal(meta.feature_names_in_, data["feature_names"])

        meta.fit(x.values, y)
        assert meta.n_features_in_ == len(data["feature_names"])
        with pytest.raises(AttributeError, match="'Stacking' object has no attribute 'feature_names_in_'"):
            meta.feature_names_in_  # pylint: disable=pointless-statement


class TestStackingSurvivalAnalysis:
    @staticmethod
    def test_fit(whas_data_with_estimator):
        x, y, meta = whas_data_with_estimator
        assert 2 == len(meta)
        meta.fit(x, y)

        p = meta._predict_estimators(x)
        assert (x.shape[0], 2) == p.shape

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
    def test_predict(whas_data_with_estimator):
        x, y, meta = whas_data_with_estimator
        meta.fit(x, y)

        # result is different if randomForestSRC has not been compiled with OpenMP support
        p = meta.predict(x)
        assert_cindex_almost_equal(y["fstat"], y["lenfol"], p,
                                   (0.7848807, 58983, 16166, 0, 14))

    @staticmethod
    def test_predict_proba():
        meta = Stacking(_PredictDummy(),
                        [('coxph', CoxPHSurvivalAnalysis()),
                         ('svm', FastSurvivalSVM(random_state=0))],
                        probabilities=False)

        with pytest.raises(AttributeError,
                           match="'_PredictDummy' object has no attribute 'predict_proba'"):
            meta.predict_proba  # pylint: disable=pointless-statement

    @staticmethod
    def test_score(whas_data_with_estimator):
        x, y, meta = whas_data_with_estimator
        meta.fit(x, y)
        c_index = meta.score(x, y)

        assert round(abs(c_index - 0.7848807), 5) == 0

    @staticmethod
    def test_feature_names_in(make_whas500):
        whas500 = make_whas500(with_mean=False, with_std=False, to_numeric=True)

        meta = Stacking(
            MeanEstimator(),
            [("coxph", CoxPHSurvivalAnalysis()), ("svm", FastSurvivalSVM(random_state=0))],
            probabilities=False
        )
        meta.fit(whas500.x_data_frame, whas500.y)
        names = whas500.x_data_frame.columns.values
        assert meta.n_features_in_ == len(names)
        assert_array_equal(meta.feature_names_in_, names)

        meta.fit(whas500.x, whas500.y)
        assert meta.n_features_in_ == len(names)
        with pytest.raises(AttributeError, match="'Stacking' object has no attribute 'feature_names_in_'"):
            meta.feature_names_in_  # pylint: disable=pointless-statement
