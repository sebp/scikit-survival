import numpy
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest

from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import scale

from sksurv.svm.minlip import MinlipSurvivalAnalysis, HingeLossSurvivalSVM
from sksurv.datasets import load_gbsg2
from sksurv.column import encode_categorical
from sksurv.svm._minlip import create_difference_matrix
from sksurv.testing import assert_cindex_almost_equal
from sksurv.util import Surv


@pytest.fixture
def toy_data():
    x = numpy.array([[1., 1.],
                     [10.2, 15.],
                     [20., 5.],
                     [40, 30],
                     [45, 21],
                     [50, 36]])

    y = Surv.from_arrays([True, True, False, True, False, False],
                         numpy.arange(1, 7) + 2 ** numpy.arange(1, 7),
                         name_event='status')
    return x, y


@pytest.fixture
def gbsg2():
    x, y = load_gbsg2()
    x = encode_categorical(x)
    return x.values, y


class TestDifferenceMatrix(object):

    @staticmethod
    def test_toy_create_difference_matrix_direct_neighbor_without_censoring(toy_data):
        _, y = toy_data
        status = numpy.ones(y.shape, dtype=bool)
        mat = create_difference_matrix(status.astype(numpy.uint8), y["time"], kind="next")

        expected = numpy.zeros((5, 6), dtype=numpy.int8)
        expected[0, 0] = -1
        expected[0, 1] = 1
        expected[1, 1] = -1
        expected[1, 2] = 1
        expected[2, 2] = -1
        expected[2, 3] = 1
        expected[3, 3] = -1
        expected[3, 4] = 1
        expected[4, 4] = -1
        expected[4, 5] = 1

        assert_array_equal(expected, mat.toarray())

        # should return first-order differences of y["time"]
        actual_diff = mat.dot(y["time"])
        expected_diff = (y["time"][1:] - y["time"][:-1])

        assert_array_almost_equal(expected_diff, actual_diff)

    @staticmethod
    def test_toy_create_difference_matrix_direct_neighbor_without_censoring_shuffled(toy_data):
        _, y = toy_data
        status = numpy.ones(y.shape, dtype=bool)
        order = [3, 2, 5, 0, 1, 4]  # = [ 20.  11.  70.   3.   6.  37.]
        time = y["time"][order]
        mat = create_difference_matrix(status.astype(numpy.uint8), time, kind="next")

        expected = numpy.zeros((5, 6), dtype=numpy.int8)
        expected[0, 3] = -1
        expected[0, 4] = 1
        expected[1, 4] = -1
        expected[1, 1] = 1
        expected[2, 1] = -1
        expected[2, 0] = 1
        expected[3, 0] = -1
        expected[3, 5] = 1
        expected[4, 5] = -1
        expected[4, 2] = 1

        assert_array_equal(expected, mat.toarray())

        # should return first-order differences of y["time"]
        actual_diff = mat.dot(time)
        expected_diff = (y["time"][1:] - y["time"][:-1])

        assert_array_almost_equal(expected_diff, actual_diff)

    @staticmethod
    def test_toy_create_difference_matrix_direct_neighbor_with_censoring(toy_data):
        _, y = toy_data
        mat = create_difference_matrix(y["status"].astype(numpy.uint8), y["time"], kind="next")

        expected = numpy.zeros((3, 6), dtype=numpy.int8)
        expected[0, 0] = -1
        expected[0, 1] = 1
        expected[1, 1] = -1
        expected[1, 2] = 1
        expected[2, 3] = -1
        expected[2, 4] = 1

        assert_array_equal(expected, mat.toarray())

        # should return first-order differences of y["time"]
        actual_diff = mat.dot(y["time"])
        comparable_pairs = numpy.array([True, True, False, True, False])
        expected_diff = (y["time"][1:] - y["time"][:-1])[comparable_pairs]

        assert_array_almost_equal(expected_diff, actual_diff)

    @staticmethod
    def test_toy_create_difference_matrix_nearest_neighbor(toy_data):
        _, y = toy_data
        status = numpy.repeat(True, len(y))
        mat = create_difference_matrix(status.astype(numpy.uint8), y["time"], kind="nearest")

        expected = numpy.zeros((5, 6), dtype=numpy.int8)
        expected[0, 0] = -1
        expected[0, 1] = 1
        expected[1, 1] = -1
        expected[1, 2] = 1
        expected[2, 2] = -1
        expected[2, 3] = 1
        expected[3, 3] = -1
        expected[3, 4] = 1
        expected[4, 4] = -1
        expected[4, 5] = 1

        assert_array_equal(expected, mat.toarray())

    @staticmethod
    def test_toy_create_difference_matrix_nearest_neighbor_censored(toy_data):
        _, y = toy_data
        mat = create_difference_matrix(y["status"].astype(numpy.uint8), y["time"], kind="nearest")

        expected = numpy.zeros((5, 6), dtype=numpy.int8)
        expected[0, 0] = -1
        expected[0, 1] = 1
        expected[1, 1] = -1
        expected[1, 2] = 1
        expected[2, 1] = -1
        expected[2, 3] = 1
        expected[3, 3] = -1
        expected[3, 4] = 1
        expected[4, 3] = -1
        expected[4, 5] = 1

        assert_array_equal(expected, mat.toarray())

    @staticmethod
    def test_toy_create_difference_matrix_full(toy_data):
        _, y = toy_data
        status = numpy.repeat(True, len(y))
        mat = create_difference_matrix(status.astype(numpy.uint8), y["time"], kind="all")

        expected = numpy.zeros((15, 6), dtype=numpy.int8)
        expected[0, 1] = 1
        expected[0, 0] = -1

        expected[1:3, 2] = 1
        expected[1, 0] = -1
        expected[2, 1] = -1

        expected[3:6, 3] = 1
        expected[3, 0] = -1
        expected[4, 1] = -1
        expected[5, 2] = -1

        expected[6:10, 4] = 1
        expected[6, 0] = -1
        expected[7, 1] = -1
        expected[8, 2] = -1
        expected[9, 3] = -1

        expected[10:15, 5] = 1
        expected[10, 0] = -1
        expected[11, 1] = -1
        expected[12, 2] = -1
        expected[13, 3] = -1
        expected[14, 4] = -1

        assert_array_equal(expected, mat.toarray())

    @staticmethod
    def test_toy_create_difference_matrix_full_censored(toy_data):
        _, y = toy_data
        mat = create_difference_matrix(y["status"].astype(numpy.uint8), y["time"], kind="all")

        expected = numpy.zeros((11, 6), dtype=numpy.int8)
        expected[0, 1] = 1
        expected[0, 0] = -1

        expected[1:3, 2] = 1
        expected[1, 0] = -1
        expected[2, 1] = -1

        expected[3:5, 3] = 1
        expected[3, 0] = -1
        expected[4, 1] = -1

        expected[5:8, 4] = 1
        expected[5, 0] = -1
        expected[6, 1] = -1
        expected[7, 3] = -1

        expected[8:12, 5] = 1
        expected[8, 0] = -1
        expected[9, 1] = -1
        expected[10, 3] = -1

        assert_array_equal(expected, mat.toarray())


class TestToyCvxpyExample(object):

    @property
    def minlip_model(self):
        return MinlipSurvivalAnalysis(solver="cvxpy", alpha=1, pairs="next")

    @property
    def svm_model(self):
        return HingeLossSurvivalSVM(solver="cvxpy", alpha=1)

    def test_toy_minlip_fit_cvxpy(self, toy_data):
        x, y = toy_data
        m = self.minlip_model
        m.set_params(alpha=2)
        m.fit(x, y)

        assert (1, x.shape[0]) == m.coef_.shape
        assert 1 == m.coef0
        expected_coef = numpy.array([[-7.18695994e-02, 7.18695994e-02, -7.51880574e-13,
                                      -2.14618562e-01, 2.14618562e-01, 0]])
        assert_array_almost_equal(m.coef_, expected_coef)

    def test_toy_minlip_timeit(self, toy_data):
        x, y = toy_data
        m = self.minlip_model
        m.set_params(timeit=7)
        m.fit(x, y)

        assert 7 == len(m.timings_)

    def test_toy_minlip_predict_1_cvxpy(self, toy_data):
        x, y = toy_data
        m = self.minlip_model
        m.fit(x, y)

        p = m.predict(x)
        assert_cindex_almost_equal(y['status'], y['time'], p,
                                   (1.0, 11, 0, 0, 0))

    def test_toy_minlip_predict_2_cvxpy(self, toy_data):
        x, y = toy_data
        m = self.minlip_model
        m.set_params(pairs="next")
        y = y.copy()
        y["time"] = numpy.arange(1, 7)
        m.fit(x, y)

        p = m.predict(numpy.array([[3, 4], [41, 29]]))
        assert_array_almost_equal(numpy.array([-0.341626, -5.374394]), p, decimal=5)

    def test_toy_hinge_fit(self, toy_data):
        x, y = toy_data
        m = self.svm_model
        m.fit(x, y)

        assert (1, x.shape[0]) == m.coef_.shape

        p = m.predict(x)
        assert_cindex_almost_equal(y['status'], y['time'], p,
                                   (1.0, 11, 0, 0, 0))

    def test_toy_hinge_predict_cvxpy(self, toy_data):
        x, y = toy_data
        m = self.svm_model
        m.fit(x, y)

        p = m.predict(numpy.array([[3, 4], [41, 29]]))
        assert_array_almost_equal(numpy.array([-0.34162189, -5.37433203]), p, decimal=5)

    def test_toy_hinge_nearest_fit(self, toy_data):
        x, y = toy_data
        m = self.svm_model
        m.set_params(pairs="nearest")
        m.fit(x, y)

        assert(1, x.shape[0]) == m.coef_.shape

        p = m.predict(x)
        assert_cindex_almost_equal(y['status'], y['time'], p,
                                   (1.0, 11, 0, 0, 0))

    def test_toy_hinge_nearest_predict_cvxpy(self, toy_data):
        x, y = toy_data
        m = self.svm_model
        m.set_params(pairs="nearest")
        m.fit(x, y)

        p = m.predict(numpy.array([[3, 4], [41, 29]]))
        assert_array_almost_equal(numpy.array([-0.3416230366, -5.3743455497]), p)


def has_cvxopt():
    try:
        import cvxopt  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not has_cvxopt(), reason='no cvxopt installed')
class TestToyCvxoptExample(object):

    @property
    def minlip_model(self):
        return MinlipSurvivalAnalysis(solver="cvxopt", alpha=1, pairs="next", max_iter=1000)

    @property
    def svm_model(self):
        return HingeLossSurvivalSVM(solver="cvxopt", alpha=1, max_iter=1000)

    def test_toy_minlip_fit_cvxopt(self, toy_data):
        x, y = toy_data
        m = self.minlip_model
        m.fit(x, y)

        assert (1, x.shape[0]) == m.coef_.shape
        assert 1 == m.coef0
        expected_coef = numpy.array([[-7.18695994e-02, 7.18695994e-02, -7.51880574e-13,
                                      -2.14618562e-01, 2.14618562e-01, 0]])
        assert_array_almost_equal(m.coef_, expected_coef)

    def test_toy_minlip_predict_1_cvxopt(self, toy_data):
        x, y = toy_data
        m = self.minlip_model
        m.fit(x, y)

        p = m.predict(x)
        assert_cindex_almost_equal(y['status'], y['time'], p,
                                   (1.0, 11, 0, 0, 0))

    def test_toy_minlip_predict_2_cvxopt(self, toy_data):
        x, y = toy_data
        m = self.minlip_model
        y = y.copy()
        y["time"] = numpy.arange(1, 7)
        m.fit(x, y)

        p = m.predict(numpy.array([[3, 4], [41, 29]]))
        assert_array_almost_equal(numpy.array([-0.34162365, -5.37435297]), p)

    def test_toy_hinge_predict_cvxopt(self, toy_data):
        x, y = toy_data
        m = self.svm_model
        m.fit(x, y)

        p = m.predict(numpy.array([[3, 4], [41, 29]]))
        assert_array_almost_equal(numpy.array([-0.341622, -5.374336]), p)

    def test_toy_hinge_nearest_predict_cvxopt(self, toy_data):
        x, y = toy_data
        m = self.svm_model
        m.set_params(pairs="nearest")
        m.fit(x, y)

        p = m.predict(numpy.array([[3, 4], [41, 29]]))
        assert_array_almost_equal(numpy.array([-0.341623, -5.374339]), p)


class TestMinlipCvxpy(object):

    @staticmethod
    def test_breast_cancer_cvxpy(gbsg2):
        x, y = gbsg2
        x = scale(x)
        m = MinlipSurvivalAnalysis(solver="cvxpy", alpha=1, pairs="next")
        m.fit(x, y)

        assert (1, x.shape[0]) == m.coef_.shape

        p = m.predict(x)
        assert_cindex_almost_equal(y['cens'], y['time'], p,
                                   (0.5990741854033906, 79720, 53352, 0, 32))

    @staticmethod
    @pytest.mark.slow
    def test_breast_cancer_rbf_cvxpy(gbsg2):
        x, y = gbsg2
        x = scale(x)
        m = MinlipSurvivalAnalysis(solver="cvxpy", alpha=1, kernel="rbf",
                                   gamma=1./8, pairs="next", max_iter=1000)
        m.fit(x, y)

        assert (1, x.shape[0]) == m.coef_.shape

        p = m.predict(x)
        assert_cindex_almost_equal(y['cens'], y['time'], p,
                                   (0.6105867500300589, 81252, 51820, 0, 32))

    @staticmethod
    def test_unknown_solver(gbsg2):
        x, y = gbsg2
        m = MinlipSurvivalAnalysis(solver=None)
        with pytest.raises(ValueError, match="unknown solver: None"):
            m.fit(x, y)

        m.set_params(solver="i don't know")
        with pytest.raises(ValueError, match="unknown solver: i don't know"):
            m.fit(x, y)

        m.set_params(solver=[('why', 'are'), ('you', 'doing this')])
        with pytest.raises(ValueError,
                           match=r"unknown solver: \[\('why', 'are'\), \('you', 'doing this'\)\]"):
            m.fit(x, y)

    @staticmethod
    @pytest.mark.slow
    def test_kernel_precomputed(gbsg2):
        x, y = gbsg2
        from sklearn.metrics.pairwise import pairwise_kernels
        from sklearn.utils.metaestimators import _safe_split

        m = MinlipSurvivalAnalysis(kernel="precomputed", solver="cvxpy")
        K = pairwise_kernels(x, metric="rbf", gamma=1./32)

        train_idx = numpy.arange(50, x.shape[0])
        test_idx = numpy.arange(50)
        X_fit, y_fit = _safe_split(m, K, y, train_idx)
        X_test, y_test = _safe_split(m, K, y, test_idx, train_idx)

        m.fit(X_fit, y_fit)

        p = m.predict(X_test)
        assert_cindex_almost_equal(y_test['cens'], y_test['time'], p,
                                   (0.6386271870794078, 466, 260, 17, 0))


@pytest.mark.skipif(not has_cvxopt(), reason='no cvxopt installed')
class TestMinlipCvxopt(object):

    @property
    def model(self):
        return MinlipSurvivalAnalysis(solver="cvxopt", alpha=1, pairs="next", max_iter=1000)

    def test_breast_cancer_cvxopt(self, gbsg2):
        x, y = gbsg2
        m = self.model
        m.fit(x, y)

        assert (1, x.shape[0]) == m.coef_.shape

        p = m.predict(x)
        assert_cindex_almost_equal(y['cens'], y['time'], p,
                                   (0.59570007214139709, 79271, 53801, 0, 32))

    def test_breast_cancer_rbf_cvxopt(self, gbsg2):
        x, y = gbsg2
        x = scale(x)
        m = self.model
        m.set_params(kernel="rbf", gamma=1./8)
        m.fit(x, y)

        assert (1, x.shape[0]) == m.coef_.shape

        p = m.predict(x)
        assert_cindex_almost_equal(y['cens'], y['time'], p,
                                   (0.6106092942166647, 81255, 51817, 0, 32))

    @staticmethod
    def test_max_iter(gbsg2):
        x, y = gbsg2
        x = scale(x)
        m = MinlipSurvivalAnalysis(solver="cvxopt", alpha=1, kernel="polynomial",
                                   degree=2, pairs="next", max_iter=5)

        with pytest.warns(ConvergenceWarning,
                          match=r"cvxopt solver did not converge: unknown \(duality gap = [.0-9]+\)"):
            m.fit(x, y)
