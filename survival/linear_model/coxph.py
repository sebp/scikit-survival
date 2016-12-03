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
import warnings

import numpy
from scipy.linalg import solve
from sklearn.base import BaseEstimator
from sklearn.utils.extmath import squared_norm
from sklearn.utils.validation import check_is_fitted

from ..base import SurvivalAnalysisMixin
from ..util import check_arrays_survival

__all__ = ['CoxPHSurvivalAnalysis']


class CoxPHOptimizer:
    """Negative partial log-likelihood of Cox proportional hazards model"""

    def __init__(self, X, event, time, alpha):
        # sort descending
        o = numpy.argsort(-time, kind="mergesort")
        self.x = X[o, :]
        self.event = event[o]
        self.time = time[o]
        self.alpha = alpha

    def nlog_likelihood(self, w):
        """Compute negative partial log-likelihood

        Parameters
        ----------
        w : array, shape = [n_features]
            Estimate of coefficients

        Returns
        -------
        loss : float
            Average negative partial log-likelihood
        """
        time = self.time
        n_samples = self.x.shape[0]
        xw = numpy.dot(self.x, w)

        loss = 0
        risk_set = 0
        k = 0
        for i in range(n_samples):
            ti = time[i]
            while k < n_samples and ti == time[k]:
                risk_set += numpy.exp(xw[k])
                k += 1

            if self.event[i]:
                loss -= (xw[i] - numpy.log(risk_set)) / n_samples

        # add regularization term to log-likelihood
        return loss + self.alpha * squared_norm(w) / (2. * n_samples)

    def update(self, w, offset=0):
        """Compute gradient and Hessian matrix with respect to `w`."""
        time = self.time
        x = self.x
        exp_xw = numpy.exp(offset + numpy.dot(x, w))
        n_samples, n_features = x.shape

        gradient = numpy.zeros((1, n_features), dtype=float)
        hessian = numpy.zeros((n_features, n_features), dtype=float)

        inv_n_samples = 1. / n_samples
        risk_set = 0
        risk_set_x = 0
        risk_set_xx = 0
        k = 0
        # iterate time in descending order
        for i in range(n_samples):
            ti = time[i]
            while k < n_samples and ti == time[k]:
                risk_set += exp_xw[k]

                # preserve 2D shape of row vector
                xk = x[k:k + 1]
                risk_set_x += exp_xw[k] * xk

                # outer product
                xx = numpy.dot(xk.T, xk)
                risk_set_xx += exp_xw[k] * xx

                k += 1

            if self.event[i]:
                gradient -= (x[i:i + 1] - risk_set_x / risk_set) * inv_n_samples

                a = risk_set_xx / risk_set
                z = risk_set_x / risk_set
                # outer product
                b = numpy.dot(z.T, z)

                hessian += (a - b) * inv_n_samples

        if self.alpha > 0:
            gradient += self.alpha * inv_n_samples * w

            diag_idx = numpy.diag_indices(n_features)
            hessian[diag_idx] += self.alpha * inv_n_samples

        self.gradient = gradient.ravel()
        self.hessian = hessian


class CoxPHSurvivalAnalysis(BaseEstimator, SurvivalAnalysisMixin):
    """Cox proportional hazards model.

    Uses the Breslow method to handle ties and Newton-Raphson optimization.

    Parameters
    ----------
    alpha : float, optional, default = 0
        Regularization parameter for ridge regression penalty.

    n_iter : int, optional, default = 100
        Maximum number of iterations.

    tol : float, optional, default = 1e-9
        Convergence criteria. Convergence is based on the negative log-likelihood::

        |1 - (new neg. log-likelihood / old neg. log-likelihood) | < tol

    verbose : bool, optional, default = False
        Whether to print statistics on the convergence
        of the optimizer.

    Attributes
    ----------
    `coef_` : ndarray, shape = [n_features]
        Coefficients of the model

    References
    ----------
    .. [1] Cox, D. R. Regression models and life tables (with discussion).
           Journal of the Royal Statistical Society. Series B, 34, 187-220, 1972.
    """

    def __init__(self, alpha=0, n_iter=100, tol=1e-9, verbose=False):
        self.alpha = alpha
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose

    def fit(self, X, y):
        """Minimize negative partial log-likelihood for provided data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Data matrix

        y : structured array, shape = [n_samples]
            A structured array containing the binary event indicator
            as first field, and time of event or time of censoring as
            second field.
        """
        X, event, time = check_arrays_survival(X, y)

        if self.alpha < 0:
            raise ValueError("alpha must be positive, but was %r" % self.alpha)

        optimizer = CoxPHOptimizer(X, event, time, self.alpha)

        w = numpy.zeros(X.shape[1])
        w_prev = w
        i = 0
        loss = float('inf')
        while True:
            optimizer.update(w)
            delta = solve(optimizer.hessian, optimizer.gradient,
                          overwrite_a=False, overwrite_b=False, check_finite=False)

            if not numpy.all(numpy.isfinite(delta)):
                raise ValueError("search direction contains NaN or infinite values")

            w_new = w - delta
            loss_new = optimizer.nlog_likelihood(w_new)
            if loss_new > loss:
                # perform step-halving if negative log-likelihood does not decrease
                w = (w_prev + w) / 2
                loss = optimizer.nlog_likelihood(w)
                i += 1
                continue

            w_prev = w
            w = w_new
            i += 1

            res = numpy.abs(1 - (loss_new / loss))
            if res < self.tol:
                break
            elif i >= self.n_iter:
                warnings.warn(('Optimization did not converge: Maximum number of iterations has been exceeded.'),
                              stacklevel=2)
                break

            loss = loss_new

        if self.verbose:
            print("Optimization stopped after %d iterations" % i)

        self.coef_ = w

        return self

    def predict(self, X):
        """Predict risk scores.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Data matrix.

        Returns
        -------
        risk_score : array, shape = [n_samples]
            Predicted risk scores.
        """
        check_is_fitted(self, "coef_")

        X = numpy.atleast_2d(X)

        return numpy.dot(X, self.coef_)
