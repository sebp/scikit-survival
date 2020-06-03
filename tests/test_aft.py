import numpy
from numpy.testing import assert_array_almost_equal

from sksurv.linear_model import IPCRidge
from sksurv.testing import assert_cindex_almost_equal


class TestIPCRidge(object):

    @staticmethod
    def test_fit(make_whas500):
        whas500 = make_whas500()
        model = IPCRidge()
        model.fit(whas500.x, whas500.y)

        assert round(abs(model.intercept_ - 5.867520370855396), 7) == 0
        expected = numpy.array([0.168481, -0.24962, 2.185086, 0.53682, -0.514611, 0.09124,
                                0.613114, 0.480357, -0.055972, 0.238472, -0.127209, -0.144063,
                                -1.625081, -0.217591])
        assert_array_almost_equal(model.coef_, expected)

    @staticmethod
    def test_predict(make_whas500):
        whas500 = make_whas500()
        model = IPCRidge()
        model.fit(whas500.x[:400], whas500.y[:400])

        x_test = whas500.x[400:]
        y_test = whas500.y[400:]
        p = model.predict(x_test)
        assert_cindex_almost_equal(y_test['fstat'], y_test['lenfol'], -p,
                                   (0.66925817946226107, 2066, 1021, 0, 1))

        assert model.score(x_test, y_test) == 1.0 - 0.66925817946226107
