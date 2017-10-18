from numpy.testing import TestCase, run_module_suite, assert_array_equal
import numpy

from sksurv.functions import StepFunction


class TestStepFunction(TestCase):
    def setUp(self):
        self.x = numpy.array([0, 1, 1.2, 1.75, 2, 2.1, 3, 3.94, 5.4, 9])
        self.y = numpy.array([11, 9, 9.12, 7.5, 7.25, 5.14, 3, 2.94, 2.4, 1.9])
        self.f = StepFunction(self.x, self.y)

    def test_exact(self):
        actual = numpy.array([self.f(v) for v in self.x])
        assert_array_equal(actual, self.y)

    def test_not_exact(self):
        z = numpy.diff(self.x).min() / 2
        actual = numpy.array([self.f(v + z) for v in self.x[:-1]])
        assert_array_equal(actual, self.y[:-1])

    def test_out_of_bounds(self):
        eps = numpy.finfo(numpy.float_).eps * 8
        values = [self.x[0] - 100,
                  self.x[-1] + 100,
                  self.x[0] - eps,
                  self.x[-1] + eps]

        for v in values:
            self.assertRaisesRegex(ValueError,
                                   "x must be within \[0.0+; 9.0+\], but was.+",
                                   self.f, v)

    def test_not_finite(self):
        values = [numpy.infty, -numpy.infty, numpy.nan]

        for v in values:
            self.assertRaisesRegex(ValueError,
                                   "x must be finite",
                                   self.f, v)


if __name__ == '__main__':
    run_module_suite()
