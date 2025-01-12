import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import scale

from sksurv.column import encode_categorical
from sksurv.datasets import load_gbsg2
from sksurv.exceptions import NoComparablePairException
from sksurv.kernels import ClinicalKernelTransform
from sksurv.svm._minlip import create_difference_matrix
from sksurv.svm.minlip import HingeLossSurvivalSVM, MinlipSurvivalAnalysis
from sksurv.testing import FixtureParameterFactory, assert_cindex_almost_equal
from sksurv.util import Surv


def create_toy_data():
    x = np.array(
        [
            [1.0, 1.0],
            [10.2, 15.0],
            [20.0, 5.0],
            [40, 30],
            [45, 21],
            [50, 36],
        ]
    )

    t = np.random.RandomState(0).exponential(scale=8, size=x.shape[0])
    t.sort()
    y = Surv.from_arrays(
        [True, True, False, True, False, False],
        t,
        name_event="status",
    )
    return x, y


@pytest.fixture()
def toy_data():
    return create_toy_data()


@pytest.fixture()
def toy_test_data():
    x = np.array(
        [
            [1.0, 1.0],
            [40, 30],
            [50, 36],
        ]
    )
    rnd = np.random.RandomState(136)
    x += rnd.randn(*x.shape)
    return x


@pytest.fixture()
def gbsg2():
    x, y = load_gbsg2()
    x = encode_categorical(x)
    return x.values, y


class DifferenceMatrixToyDataCases(FixtureParameterFactory):
    @property
    def time_and_event(self):
        _, y = create_toy_data()
        time = y["time"]
        status = y["status"]
        return time, status

    def _create_expected(self, shape=(5, 6)):
        return np.zeros(shape, dtype=np.int8)

    def data_direct_neighbor_without_censoring(self):
        kind = "next"
        time, status = self.time_and_event
        status = np.ones_like(status)

        expected = self._create_expected()
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

        expected_diff = time[1:] - time[:-1]

        return time, status.astype(np.uint8), kind, expected, expected_diff

    def data_direct_neighbor_without_censoring_shuffled(self):
        kind = "next"
        time, status = self.time_and_event
        status = np.ones_like(status)
        order = [3, 2, 5, 0, 1, 4]  # = [ 20.  11.  70.   3.   6.  37.]

        expected = self._create_expected()
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

        expected_diff = time[1:] - time[:-1]
        time = time[order]

        return time, status.astype(np.uint8), kind, expected, expected_diff

    def data_direct_neighbor_with_censoring(self):
        kind = "next"
        time, status = self.time_and_event

        expected = self._create_expected((3, 6))
        expected[0, 0] = -1
        expected[0, 1] = 1
        expected[1, 1] = -1
        expected[1, 2] = 1
        expected[2, 3] = -1
        expected[2, 4] = 1

        comparable_pairs = np.array([True, True, False, True, False])
        expected_diff = (time[1:] - time[:-1])[comparable_pairs]

        return time, status.astype(np.uint8), kind, expected, expected_diff

    def data_nearest_neighbor(self):
        kind = "nearest"
        time, status = self.time_and_event
        status = np.ones_like(status)

        expected = self._create_expected()
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

        return time, status.astype(np.uint8), kind, expected, None

    def data_nearest_neighbor_censored(self):
        kind = "nearest"
        time, status = self.time_and_event

        expected = self._create_expected()
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

        return time, status.astype(np.uint8), kind, expected, None

    def data_full(self):
        kind = "all"
        time, status = self.time_and_event
        status = np.ones_like(status)

        expected = self._create_expected((15, 6))
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

        return time, status.astype(np.uint8), kind, expected, None

    def data_full_censored(self):
        kind = "all"
        time, status = self.time_and_event

        expected = self._create_expected((11, 6))
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

        return time, status.astype(np.uint8), kind, expected, None


@pytest.mark.parametrize("time,status,kind,expected_mat,expected_diff", DifferenceMatrixToyDataCases().get_cases())
def test_toy_create_difference_matrix(time, status, kind, expected_mat, expected_diff):
    mat = create_difference_matrix(status, time, kind=kind)

    assert_array_equal(expected_mat, mat.toarray())

    if expected_diff is not None:
        # should return first-order differences of y["time"]
        actual_diff = mat.dot(time)
        assert_array_almost_equal(expected_diff, actual_diff)


@pytest.fixture()
def minlip_model_factory():
    def create_and_fit_model(solver, x, y, **kwargs):
        params = {"solver": solver, "alpha": 1, "pairs": "next"}
        params.update(kwargs)
        return MinlipSurvivalAnalysis(**params).fit(x, y)

    return create_and_fit_model


@pytest.fixture()
def toy_data_standardized(toy_data, toy_test_data):
    x, y = toy_data
    m = np.mean(x, axis=0)
    sd = np.std(x, axis=0)
    return ((x - m) / sd, y), (toy_test_data - m) / sd


class TestToyMinlipSurvivalAnalysis:
    @staticmethod
    @pytest.mark.parametrize("solver,expected_iters", [("osqp", 100), ("ecos", 10)])
    def test_fit_alpha_2(minlip_model_factory, toy_data, solver, expected_iters):
        x, y = toy_data
        m = minlip_model_factory(solver, x, y, alpha=2.0)

        assert m.n_iter_ > expected_iters
        assert (1, 6) == m.coef_.shape
        assert 1 == m.coef0
        expected_coef = np.array(
            [[-0.011728003147, 0.011728002895, 0.000000000252, -0.017524801335, 0.017524801335, 0.0]]
        )
        assert_array_almost_equal(m.coef_, expected_coef)

    @staticmethod
    @pytest.mark.parametrize("solver", ["osqp", "ecos"])
    def test_timeit(minlip_model_factory, toy_data, solver):
        x, y = toy_data
        m = minlip_model_factory(solver, x, y, timeit=7)

        assert 7 == len(m.timings_)

    @staticmethod
    @pytest.mark.parametrize("solver", ["osqp", "ecos"])
    def test_predict_1(minlip_model_factory, toy_data, solver):
        x, y = toy_data
        p = minlip_model_factory(solver, x, y).predict(x)
        assert_cindex_almost_equal(y["status"], y["time"], p, (1.0, 11, 0, 0, 0))

    @staticmethod
    @pytest.mark.parametrize("solver", ["osqp", "ecos"])
    def test_predict_2(minlip_model_factory, toy_data_standardized, solver):
        (x, y), x_test = toy_data_standardized

        y = y.copy()
        y["time"] = np.arange(1, 7)

        p = minlip_model_factory(solver, x, y, pairs="next").predict(x_test)

        expected = np.array([1.368221203557392, -0.476483331099142, -1.009079072163642])
        assert_array_almost_equal(expected, p, decimal=5)


@pytest.fixture()
def svm_model_factory():
    def create_and_fit_model(solver, x, y, **kwargs):
        params = {"solver": solver, "alpha": 2}
        params.update(kwargs)
        return HingeLossSurvivalSVM(**params).fit(x, y)

    return create_and_fit_model


@pytest.mark.parametrize("solver", ["osqp", "ecos"])
@pytest.mark.parametrize("pairs", ["all", "nearest"])
def test_toy_hinge_fit_and_predict(svm_model_factory, toy_data_standardized, solver, pairs):
    (x, y), x_test = toy_data_standardized
    m = svm_model_factory(solver, x, y, pairs=pairs)

    assert (1, 6) == m.coef_.shape
    assert 1 == m.coef0
    expected_coef = np.array([[-1.893832101337, 1.083653895940, 0.810178205398, -2.0, 2.0, 0.0]])
    assert_array_almost_equal(m.coef_, expected_coef, decimal=5)

    p = m.predict(x)
    assert_cindex_almost_equal(y["status"], y["time"], p, (1.0, 11, 0, 0, 0))

    p = m.predict(x_test)
    expected = np.array([2.8571060045, -1.2661069033, -2.3044907774])
    assert_array_almost_equal(expected, p, decimal=5)


@pytest.fixture()
def gbsg2_scaled(gbsg2):
    x, y = gbsg2
    x = scale(x)
    return x, y


class TestMinlipBreastCancer:
    @staticmethod
    @pytest.mark.parametrize("solver,expected_iters", [("osqp", 1000), ("ecos", 10)])
    def test_fit_and_predict(gbsg2_scaled, minlip_model_factory, solver, expected_iters):
        x, y = gbsg2_scaled
        m = minlip_model_factory(solver, x, y)

        assert (1, x.shape[0]) == m.coef_.shape

        assert m.n_iter_ > expected_iters

        p = m.predict(x)
        assert_cindex_almost_equal(y["cens"], y["time"], p, (0.5990741854033906, 79720, 53352, 0, 42))

    @staticmethod
    @pytest.mark.slow()
    @pytest.mark.parametrize(
        "solver,expected_cindex",
        [
            ("osqp", (0.6106092942166647, 81255, 51817, 0, 42)),
            ("ecos", (0.6105867500300589, 81252, 51820, 0, 42)),
        ],
    )
    def test_fit_and_predict_rbf(gbsg2_scaled, minlip_model_factory, solver, expected_cindex):
        x, y = gbsg2_scaled
        m = minlip_model_factory(solver, x, y, kernel="rbf", gamma=1.0 / 8, pairs="next", max_iter=1000)
        m.fit(x, y)

        assert (1, x.shape[0]) == m.coef_.shape

        p = m.predict(x)
        assert_cindex_almost_equal(y["cens"], y["time"], p, expected_cindex)

    @staticmethod
    @pytest.mark.slow()
    @pytest.mark.parametrize("solver", ["osqp", "ecos"])
    def test_kernel_precomputed(gbsg2_scaled, solver):
        x, y = gbsg2_scaled
        from sklearn.metrics.pairwise import pairwise_kernels
        from sklearn.utils.metaestimators import _safe_split

        m = MinlipSurvivalAnalysis(solver=solver, kernel="precomputed", max_iter=25000)
        K = pairwise_kernels(x, metric="rbf", gamma=0.1)

        train_idx = np.arange(200, x.shape[0])
        test_idx = np.arange(200)
        X_fit, y_fit = _safe_split(m, K, y, train_idx)
        X_test, y_test = _safe_split(m, K, y, test_idx, train_idx)

        m.fit(X_fit, y_fit)

        p = m.predict(X_test)
        assert_cindex_almost_equal(y_test["cens"], y_test["time"], p, (0.6518928901200369, 8472, 4524, 0, 3))

    @staticmethod
    @pytest.mark.slow()
    def test_fit_clinical_kernel(make_whas500):
        whas500 = make_whas500(with_mean=False, with_std=False)

        trans = ClinicalKernelTransform()
        trans.fit(whas500.x_data_frame)

        m = MinlipSurvivalAnalysis(kernel=trans.pairwise_kernel)
        m.fit(whas500.x, whas500.y)

        assert not m.__sklearn_tags__().input_tags.pairwise

        c = m.score(whas500.x, whas500.y)
        assert c == pytest.approx(0.7314135916645598)

    @staticmethod
    @pytest.mark.parametrize("solver", ["osqp", "ecos"])
    def test_max_iter(gbsg2_scaled, solver):
        x, y = gbsg2_scaled
        m = MinlipSurvivalAnalysis(solver=solver, alpha=1, kernel="poly", degree=2, pairs="next", max_iter=5)

        with pytest.warns(
            ConvergenceWarning, match=f"{solver.upper()} solver did not converge: maximum iterations reached"
        ):
            m.fit(x, y)


@pytest.mark.parametrize(
    "solver",
    [
        None,
        "i don't know",
        [("why", "are"), ("you", "doing this")],
    ],
)
def test_unknown_solver(gbsg2, solver):
    x, y = gbsg2
    m = MinlipSurvivalAnalysis(solver=solver)

    msg = r"The 'solver' parameter of MinlipSurvivalAnalysis must be a str among \{.+\}\. Got .+ instead\."
    with pytest.raises(ValueError, match=msg):
        m.fit(x, y)


@pytest.mark.parametrize("model_cls", [MinlipSurvivalAnalysis, HingeLossSurvivalSVM])
@pytest.mark.parametrize("solver", ["ecos", "osqp"])
@pytest.mark.parametrize("pairs", ["all", "nearest", "next"])
def test_fit_uncomparable(whas500_uncomparable, model_cls, solver, pairs):
    ssvm = model_cls(solver=solver, pairs=pairs)
    with pytest.raises(NoComparablePairException):
        ssvm.fit(whas500_uncomparable.x, whas500_uncomparable.y)
