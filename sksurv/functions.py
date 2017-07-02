# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy
from sklearn.utils import check_consistent_length

__all__ = ['StepFunction']


class StepFunction(object):
    """Callable step function.

    .. math::

        f(z) = a * y_i + b,
        x_i \\leq z < x_{i + 1}

    Parameters
    ----------
    x : ndarray, shape = [n_points,]
        Values on the x axis in ascending order.

    y : ndarray, shape = [n_points,]
        Corresponding values on the y axis.

    a : float, optional
        Constant to multiply
    """
    def __init__(self, x, y, a=1., b=0.):
        check_consistent_length(x, y)
        self.x = x
        self.y = y
        self.a = a
        self.b = b

    def __call__(self, x):
        if not numpy.isfinite(x):
            raise ValueError("x must be finite")
        if x < self.x[0] or x > self.x[-1]:
            raise ValueError(
                "x must be within [%f; %f], but was %f" % (self.x[0], self.x[-1], x))
        i = numpy.searchsorted(self.x, x, side='left')
        if self.x[i] != x:
            i -= 1
        return self.a * self.y[i] + self.b

    def __repr__(self):
        return "StepFunction(x=%r, y=%r)" % (self.x, self.y)
