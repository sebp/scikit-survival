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
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.extmath import squared_norm
from sklearn.utils.validation import check_is_fitted

from ..base import SurvivalAnalysisMixin
from ..functions import StepFunction
from ..nonparametric import _compute_counts
from ..util import check_arrays_survival

__all__ = ['CoxPHSurvivalAnalysis']


class BreslowEstimator:
    """Breslow's estimator of the cumulative hazard function.

    Attributes
    ----------
    cum_baseline_hazard_ : :class:`sksurv.functions.StepFunction`
        Cumulative baseline hazard function.

    baseline_survival_ : :class:`sksurv.functions.StepFunction`
        Baseline survival function.
    """

    def fit(self, linear_predictor, event, time):
        """Compute baseline cumulative hazard function.

        Parameters
        ----------
        linear_predictor : array-like, shape = (n_samples,)
            Linear predictor of risk: `X @ coef`.

        event : array-like, shape = (n_samples,)
            Contains binary event indicators.

        time : array-like, shape = (n_samples,)
            Contains event/censoring times.

        Returns
        -------
        self
        """
        risk_score = numpy.exp(linear_predictor)
        order = numpy.argsort(time, kind="mergesort")
        risk_score = risk_score[order]
        uniq_times, n_events, n_at_risk = _compute_counts(event, time, order)

        divisor = numpy.empty(n_at_risk.shape, dtype=numpy.float_)
        value = numpy.sum(risk_score)
        divisor[0] = value
        k = 0
        for i in range(1, len(n_at_risk)):
            d = n_at_risk[i - 1] - n_at_risk[i]
            value -= risk_score[k:(k + d)].sum()
            k += d
            divisor[i] = value

        assert k == n_at_risk[0] - n_at_risk[-1]

        y = numpy.cumsum(n_events / divisor)
        self.cum_baseline_hazard_ = StepFunction(uniq_times, y)
        self.baseline_survival_ = StepFunction(self.cum_baseline_hazard_.x,
                                               numpy.exp(- self.cum_baseline_hazard_.y))
        return self

    def get_cumulative_hazard_function(self, linear_predictor):
        """Predict cumulative hazard function.

        Parameters
        ----------
        linear_predictor : array-like, shape = (n_samples,)
            Linear predictor of risk: `X @ coef`.

        Returns
        -------
        cum_hazard : ndarray, shape = (n_samples,)
            Predicted cumulative hazard functions.
        """
        risk_score = numpy.exp(linear_predictor)
        n_samples = risk_score.shape[0]
        funcs = numpy.empty(n_samples, dtype=numpy.object_)
        for i in range(n_samples):
            funcs[i] = StepFunction(x=self.cum_baseline_hazard_.x,
                                    y=self.cum_baseline_hazard_.y,
                                    a=risk_score[i])
        return funcs

    def get_survival_function(self, linear_predictor):
        """Predict survival function.

        Parameters
        ----------
        linear_predictor : array-like, shape = (n_samples,)
            Linear predictor of risk: `X @ coef`.

        Returns
        -------
        survival : ndarray, shape = (n_samples,)
            Predicted survival functions.
        """
        risk_score = numpy.exp(linear_predictor)
        n_samples = risk_score.shape[0]
        funcs = numpy.empty(n_samples, dtype=numpy.object_)
        for i in range(n_samples):
            funcs[i] = StepFunction(x=self.baseline_survival_.x,
                                    y=numpy.power(self.baseline_survival_.y, risk_score[i]))
        return funcs


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
        w : array, shape = (n_features,)
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


class VerboseReporter:

    def __init__(self, verbose):
        self.verbose = verbose

    def end_max_iter(self, i):
        if self.verbose > 0:
            print("iter {:>6d}: reached maximum number of iterations. Stopping.".format(i + 1))

    def end_converged(self, i):
        if self.verbose > 0:
            print("iter {:>6d}: optimization converged".format(i + 1))

    def update(self, i, delta, loss_new):
        if self.verbose > 2:
            print("iter {:>6d}: update = {}".format(i + 1, delta))
        if self.verbose > 1:
            print("iter {:>6d}: loss = {:.10f}".format(i + 1, loss_new))

    def step_halving(self, i, loss):
        if self.verbose > 1:
            print("iter {:>6d}: loss increased, performing step-halving. loss = {:.10f}".format(i, loss))


class CoxPHSurvivalAnalysis(BaseEstimator, SurvivalAnalysisMixin):
    """Cox proportional hazards model.

    Uses the Breslow method to handle ties and Newton-Raphson optimization.

    See [1]_ for further description.

    Parameters
    ----------
    alpha : float, optional, default: 0
        Regularization parameter for ridge regression penalty.

    n_iter : int, optional, default: 100
        Maximum number of iterations.

    tol : float, optional, default: 1e-9
        Convergence criteria. Convergence is based on the negative log-likelihood::

        |1 - (new neg. log-likelihood / old neg. log-likelihood) | < tol

    verbose : int, optional, default: 0
        Specified the amount of additional debug information
        during optimization.

    Attributes
    ----------
    coef_ : ndarray, shape = (n_features,)
        Coefficients of the model

    cum_baseline_hazard_ : :class:`sksurv.functions.StepFunction`
        Estimated baseline cumulative hazard function.

    baseline_survival_ : :class:`sksurv.functions.StepFunction`
        Estimated baseline survival function.

    References
    ----------
    .. [1] Cox, D. R. Regression models and life tables (with discussion).
           Journal of the Royal Statistical Society. Series B, 34, 187-220, 1972.
    """

    def __init__(self, alpha=0, n_iter=100, tol=1e-9, verbose=0):
        self.alpha = alpha
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose

        self._baseline_model = BreslowEstimator()

    @property
    def cum_baseline_hazard_(self):
        return self._baseline_model.cum_baseline_hazard_

    @property
    def baseline_survival_(self):
        return self._baseline_model.baseline_survival_

    def fit(self, X, y):
        """Minimize negative partial log-likelihood for provided data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data matrix

        y : structured array, shape = (n_samples,)
            A structured array containing the binary event indicator
            as first field, and time of event or time of censoring as
            second field.

        Returns
        -------
        self
        """
        X, event, time = check_arrays_survival(X, y)

        if self.alpha < 0:
            raise ValueError("alpha must be positive, but was %r" % self.alpha)

        optimizer = CoxPHOptimizer(X, event, time, self.alpha)

        verbose_reporter = VerboseReporter(self.verbose)
        w = numpy.zeros(X.shape[1])
        w_prev = w
        i = 0
        loss = float('inf')
        while True:
            if i >= self.n_iter:
                verbose_reporter.end_max_iter(i)
                warnings.warn(('Optimization did not converge: Maximum number of iterations has been exceeded.'),
                              stacklevel=2, category=ConvergenceWarning)
                break

            optimizer.update(w)
            delta = solve(optimizer.hessian, optimizer.gradient,
                          overwrite_a=False, overwrite_b=False, check_finite=False)

            if not numpy.all(numpy.isfinite(delta)):
                raise ValueError("search direction contains NaN or infinite values")

            w_new = w - delta
            loss_new = optimizer.nlog_likelihood(w_new)
            verbose_reporter.update(i, delta, loss_new)
            if loss_new > loss:
                # perform step-halving if negative log-likelihood does not decrease
                w = (w_prev + w) / 2
                loss = optimizer.nlog_likelihood(w)
                verbose_reporter.step_halving(i, loss)
                i += 1
                continue

            w_prev = w
            w = w_new

            res = numpy.abs(1 - (loss_new / loss))
            if res < self.tol:
                verbose_reporter.end_converged(i)
                break

            loss = loss_new
            i += 1

        self.coef_ = w
        self._baseline_model.fit(numpy.dot(X, self.coef_), event, time)
        return self

    def predict(self, X):
        """Predict risk scores.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data matrix.

        Returns
        -------
        risk_score : array, shape = (n_samples,)
            Predicted risk scores.
        """
        check_is_fitted(self, "coef_")

        X = numpy.atleast_2d(X)

        return numpy.dot(X, self.coef_)

    def predict_cumulative_hazard_function(self, X):
        """Predict cumulative hazard function.

        The cumulative hazard function for an individual
        with feature vector :math:`x` is defined as

        .. math::

            H(t \\mid x) = \\exp(x^\\top \\beta) H_0(t) ,

        where :math:`H_0(t)` is the baseline hazard function,
        estimated by Breslow's estimator.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data matrix.

        Returns
        -------
        cum_hazard : ndarray, shape = (n_samples,)
            Predicted cumulative hazard functions.
        """
        return self._baseline_model.get_cumulative_hazard_function(self.predict(X))

    def predict_survival_function(self, X):
        """Predict survival function.

        The survival function for an individual
        with feature vector :math:`x` is defined as

        .. math::

            S(t \\mid x) = S_0(t)^{\\exp(x^\\top \\beta)} ,

        where :math:`S_0(t)` is the baseline survival function,
        estimated by Breslow's estimator.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data matrix.

        Returns
        -------
        survival : ndarray, shape = (n_samples,)
            Predicted survival functions.
        """
        return self._baseline_model.get_survival_function(self.predict(X))
