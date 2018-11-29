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

        assert round(abs(model.intercept_ - 5.8673567124629571), 7) == 0
        expected = numpy.array([0.168517, -0.249717, 2.18515, 0.536795, -0.514571, 0.091203,
                                0.613006, 0.480385, -0.055949, 0.238529, -0.127148, -0.144134,
                                -1.625041, -0.217469])
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
                                   (0.66925817946226107, 2066, 1021, 0, 6))

        assert model.score(x_test, y_test) == 1.0 - 0.66925817946226107
