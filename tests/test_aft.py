import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
from scipy.sparse import csr_array
from sklearn.pipeline import make_pipeline

from sksurv.base import SurvivalAnalysisMixin
from sksurv.linear_model import IPCRidge
from sksurv.testing import assert_cindex_almost_equal


class TestIPCRidge:
    @staticmethod
    def test_fit(make_whas500):
        whas500 = make_whas500()
        model = IPCRidge()
        model.fit(whas500.x, whas500.y)

        assert model.intercept_ == pytest.approx(5.867520370855396, 1e-7)
        expected = np.array(
            [
                0.168481,
                -0.24962,
                2.185086,
                0.53682,
                -0.514611,
                0.09124,
                0.613114,
                0.480357,
                -0.055972,
                0.238472,
                -0.127209,
                -0.144063,
                -1.625081,
                -0.217591,
            ]
        )
        assert_array_almost_equal(model.coef_, expected)

    @staticmethod
    def test_predict(make_whas500):
        whas500 = make_whas500()
        model = IPCRidge()
        model.fit(whas500.x[:400], whas500.y[:400])

        x_test = whas500.x[400:]
        y_test = whas500.y[400:]
        p = model.predict(x_test)
        assert_cindex_almost_equal(
            y_test["fstat"],
            y_test["lenfol"],
            -p,
            (0.66925817946226107, 2066, 1021, 0, 1),
        )

        assert model.score(x_test, y_test) == 0.66925817946226107

    @staticmethod
    def test_pipeline_score(make_whas500):
        whas500 = make_whas500()
        pipe = make_pipeline(IPCRidge())
        pipe.fit(whas500.x[:400], whas500.y[:400])

        x_test = whas500.x[400:]
        y_test = whas500.y[400:]
        p = pipe.predict(x_test)
        assert_cindex_almost_equal(
            y_test["fstat"],
            y_test["lenfol"],
            -p,
            (0.66925817946226107, 2066, 1021, 0, 1),
        )

        assert SurvivalAnalysisMixin.score(pipe, x_test, y_test) == 0.66925817946226107

    @staticmethod
    def test_sparse(make_whas500):
        whas500 = make_whas500(to_numeric=True)

        X_train = csr_array(whas500.x[:400])
        X_test = csr_array(whas500.x[400:])
        y_train = whas500.y[:400]
        y_test = whas500.y[400:]

        model_1 = IPCRidge(solver="lsqr").fit(X_train, y_train)
        model_2 = IPCRidge(solver="lsqr").fit(whas500.x[:400], y_train)

        assert model_1.solver_ == model_2.solver_

        pred_1 = model_1.score(X_test, y_test)
        pred_2 = model_2.score(whas500.x[400:], y_test)

        assert pred_1 == pytest.approx(pred_2)
