from os.path import join, dirname

import numpy
from numpy.testing import TestCase, run_module_suite, assert_array_almost_equal

from sksurv.linear_model import IPCRidge
from sksurv.column import standardize
from sksurv.datasets import load_whas500
from sksurv.metrics import concordance_index_censored


class TestIPCRidge(TestCase):
    def setUp(self):
        x, self.y, =  load_whas500()
        self.x = standardize(x)

    def test_fit(self):
        model = IPCRidge()
        model.fit(self.x, self.y)

        self.assertAlmostEqual(model.intercept_, 5.8673567124629571)
        expected = numpy.array([0.168517, -0.249717, 2.18515, 0.536795, -0.514571, 0.091203,
                                0.613006, 0.480385, -0.055949, 0.238529, -0.127148, -0.144134,
                                -1.625041, -0.217469])
        assert_array_almost_equal(model.coef_, expected)

    def test_predict(self):
        model = IPCRidge()
        model.fit(self.x[:400], self.y[:400])

        x_test = self.x[400:]
        y_test = self.y[400:]
        p = model.predict(x_test)
        ci = concordance_index_censored(y_test['fstat'], y_test['lenfol'], -p)

        self.assertAlmostEqual(ci[0], 0.66925817946226107)
        self.assertEqual(ci[1], 2066)
        self.assertEqual(ci[2], 1021)
        self.assertEqual(ci[3], 0)
        self.assertEqual(ci[4], 6)

        self.assertEqual(model.score(x_test, y_test), 1.0 - ci[0])


if __name__ == '__main__':
    run_module_suite()
