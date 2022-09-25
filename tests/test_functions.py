import numpy as np
from numpy.testing import assert_array_equal
import pytest

from sksurv.functions import StepFunction


@pytest.fixture()
def a_step_function():
    x = np.array([0, 1, 1.2, 1.75, 2, 2.1, 3, 3.94, 5.4, 9])
    y = np.array([11, 9, 9.12, 7.5, 7.25, 5.14, 3, 2.94, 2.4, 1.9])
    f = StepFunction(x, y)
    return f


class TestStepFunction:

    @staticmethod
    def test_exact(a_step_function):
        actual = np.array([a_step_function(v) for v in a_step_function.x])
        assert_array_equal(actual, a_step_function.y)

    @staticmethod
    def test_not_exact(a_step_function):
        z = np.diff(a_step_function.x).min() / 2
        actual = np.array([a_step_function(v + z) for v in a_step_function.x[:-1]])
        assert_array_equal(actual, a_step_function.y[:-1])

    @staticmethod
    def test_out_of_bounds(a_step_function):
        eps = np.finfo(float).eps * 8
        values = [a_step_function.x[0] - 100,
                  a_step_function.x[-1] + 100,
                  a_step_function.x[0] - eps,
                  a_step_function.x[-1] + eps]

        for v in values:
            with pytest.raises(ValueError, match=r"x must be within \[0.0+; 9.0+\]"):
                a_step_function(v)

    @staticmethod
    def test_not_finite(a_step_function, non_finite_value):
        with pytest.raises(ValueError, match="x must be finite"):
            a_step_function(non_finite_value)

    @staticmethod
    def test_equal(a_step_function):
        x = np.array([0, 1, 1.2, 1.75, 2, 2.1, 3, 3.94, 5.4, 9])
        y = np.array([11, 9, 9.12, 7.5, 7.25, 5.14, 3, 2.94, 2.4, 1.9])
        other_step_function = StepFunction(x, y)

        assert a_step_function == other_step_function

        different_step_function = StepFunction(x+1, y)
        assert a_step_function != different_step_function

        assert a_step_function != x
