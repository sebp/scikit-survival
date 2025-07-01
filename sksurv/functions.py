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

import numpy as np
from sklearn.utils.validation import check_consistent_length

__all__ = ["StepFunction"]


class StepFunction:
    r"""A callable step function.

    The function is defined by a set of points :math:`(x_i, y_i)` and is
    evaluated as:

    .. math::

        f(z) = a \cdot y_i + b \quad \text{if} \quad x_i \leq z < x_{i + 1}

    Parameters
    ----------
    x : ndarray, shape = (n_points,)
        The values on the x-axis, must be in ascending order.
    y : ndarray, shape = (n_points,)
        The corresponding values on the y-axis.
    a : float, optional, default: 1.0
        A constant factor to scale ``y`` by.
    b : float, optional, default: 0.0
        A constant offset term.
    domain : tuple, optional, default: (0, None)
        A tuple ``(lower, upper)`` that defines the domain of the step function.
        If ``lower`` or ``upper`` is ``None``, the first or last value of ``x`` is
        used as the limit, respectively.
    """

    def __init__(self, x, y, *, a=1.0, b=0.0, domain=(0, None)):
        check_consistent_length(x, y)
        self.x = x
        self.y = y
        self.a = a
        self.b = b
        domain_lower = self.x[0] if domain[0] is None else domain[0]
        domain_upper = self.x[-1] if domain[1] is None else domain[1]
        self._domain = (float(domain_lower), float(domain_upper))

    @property
    def domain(self):
        """The domain of the function.

        The domain is the range of values that the function accepts.

        Returns
        -------
        lower_limit : float
            Lower limit of the omain.

        upper_limit : float
            Upper limit of the domain.
        """
        return self._domain

    def __call__(self, x):
        """Evaluate the step function at given values.

        Parameters
        ----------
        x : float or array-like, shape=(n_values,)
            The values at which to evaluate the step function.
            Values must be within the function's ``domain``.

        Returns
        -------
        y : float or array-like, shape=(n_values,)
            The value of the step function at ``x``.

        Raises
        ------
        ValueError
            If ``x`` contains values outside the function's ``domain``.
        """
        x = np.atleast_1d(x)
        if not np.isfinite(x).all():
            raise ValueError("x must be finite")
        if np.min(x) < self._domain[0] or np.max(x) > self.domain[1]:
            raise ValueError(f"x must be within [{self.domain[0]:f}; {self.domain[1]:f}]")

        # x is within the domain, but we need to account for self.domain[0] <= x < self.x[0]
        x = np.clip(x, a_min=self.x[0], a_max=None)

        i = np.searchsorted(self.x, x, side="left")
        not_exact = self.x[i] != x
        i[not_exact] -= 1
        value = self.a * self.y[i] + self.b
        if value.shape[0] == 1:
            return value[0]
        return value

    def __repr__(self):
        return f"StepFunction(x={self.x!r}, y={self.y!r}, a={self.a!r}, b={self.b!r})"

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return all(self.x == other.x) and all(self.y == other.y) and self.a == other.a and self.b == other.b
        return False
