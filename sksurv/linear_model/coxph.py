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
import numbers
import warnings

import numpy
from scipy.linalg import solve
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import check_array, check_is_fitted

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
        uniq_times, n_events, n_at_risk, _ = _compute_counts(event, time, order)

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

    def __init__(self, X, event, time, alpha, ties):
        # sort descending
        o = numpy.argsort(-time, kind="mergesort")
        self.x = X[o, :]
        self.event = event[o]
        self.time = time[o]
        self.alpha = alpha
        self.no_alpha = numpy.all(self.alpha < numpy.finfo(self.alpha.dtype).eps)
        if ties not in ("breslow", "efron"):
            raise ValueError("ties must be one of 'breslow', 'efron'")
        self._is_breslow = ties == "breslow"

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
        breslow = self._is_breslow
        xw = numpy.dot(self.x, w)

        loss = 0
        risk_set = 0
        k = 0
        while k < n_samples:
            ti = time[k]
            numerator = 0
            n_events = 0
            risk_set2 = 0
            while k < n_samples and ti == time[k]:
                if self.event[k]:
                    numerator += xw[k]
                    risk_set2 += numpy.exp(xw[k])
                    n_events += 1
                else:
                    risk_set += numpy.exp(xw[k])
                k += 1

            if n_events > 0:
                if breslow:
                    risk_set += risk_set2
                    loss -= (numerator - n_events * numpy.log(risk_set)) / n_samples
                else:
                    numerator /= n_events
                    for _ in range(n_events):
                        risk_set += risk_set2 / n_events
                        loss -= (numerator - numpy.log(risk_set)) / n_samples

        # add regularization term to log-likelihood
        return loss + numpy.sum(self.alpha * numpy.square(w)) / (2. * n_samples)

    def update(self, w, offset=0):
        """Compute gradient and Hessian matrix with respect to `w`."""
        time = self.time
        x = self.x
        breslow = self._is_breslow
        exp_xw = numpy.exp(offset + numpy.dot(x, w))
        n_samples, n_features = x.shape

        gradient = numpy.zeros((1, n_features), dtype=w.dtype)
        hessian = numpy.zeros((n_features, n_features), dtype=w.dtype)

        inv_n_samples = 1. / n_samples
        risk_set = 0
        risk_set_x = numpy.zeros((1, n_features), dtype=w.dtype)
        risk_set_xx = numpy.zeros((n_features, n_features), dtype=w.dtype)
        k = 0
        # iterate time in descending order
        while k < n_samples:
            ti = time[k]
            n_events = 0
            numerator = 0
            risk_set2 = 0
            risk_set_x2 = numpy.zeros_like(risk_set_x)
            risk_set_xx2 = numpy.zeros_like(risk_set_xx)
            while k < n_samples and ti == time[k]:
                # preserve 2D shape of row vector
                xk = x[k:k + 1]

                # outer product
                xx = numpy.dot(xk.T, xk)

                if self.event[k]:
                    numerator += xk
                    risk_set2 += exp_xw[k]
                    risk_set_x2 += exp_xw[k] * xk
                    risk_set_xx2 += exp_xw[k] * xx
                    n_events += 1
                else:
                    risk_set += exp_xw[k]
                    risk_set_x += exp_xw[k] * xk
                    risk_set_xx += exp_xw[k] * xx
                k += 1

            if n_events > 0:
                if breslow:
                    risk_set += risk_set2
                    risk_set_x += risk_set_x2
                    risk_set_xx += risk_set_xx2

                    z = risk_set_x / risk_set
                    gradient -= (numerator - n_events * z) * inv_n_samples

                    a = risk_set_xx / risk_set
                    # outer product
                    b = numpy.dot(z.T, z)

                    hessian += n_events * (a - b) * inv_n_samples
                else:
                    numerator /= n_events
                    for _ in range(n_events):
                        risk_set += risk_set2 / n_events
                        risk_set_x += risk_set_x2 / n_events
                        risk_set_xx += risk_set_xx2 / n_events

                        z = risk_set_x / risk_set
                        gradient -= (numerator - z) * inv_n_samples

                        a = risk_set_xx / risk_set
                        # outer product
                        b = numpy.dot(z.T, z)

                        hessian += (a - b) * inv_n_samples

        if not self.no_alpha:
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

    There are two possible choices for handling tied event times.
    The default is Breslow's method, which considers each of the
    events at a given time as distinct. Efron's method is more
    accurate if there are a large number of ties. When the number
    of ties is small, the estimated coefficients by Breslow's and
    Efron's method are quite close. Uses Newton-Raphson optimization.

    See [1]_, [2]_, [3]_ for further description.

    Parameters
    ----------
    alpha : float, ndarray of shape (n_features,), optional, default: 0
        Regularization parameter for ridge regression penalty.
        If a single float, the same penalty is used for all features.
        If an array, there must be one penalty for each feature.
        If you want to include a subset of features without penalization,
        set the corresponding entries to 0.

    ties : "breslow" | "efron", optional, default: "breslow"
        The method to handle tied event times. If there are
        no tied event times all the methods are equivalent.

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

    See also
    --------
    sksurv.linear_model.CoxnetSurvivalAnalysis
        Cox proportional hazards model with l1 (LASSO) and l2 (ridge) penalty.

    References
    ----------
    .. [1] Cox, D. R. Regression models and life tables (with discussion).
           Journal of the Royal Statistical Society. Series B, 34, 187-220, 1972.
    .. [2] Breslow, N. E. Covariance Analysis of Censored Survival Data.
           Biometrics 30 (1974): 89–99.
    .. [3] Efron, B. The Efficiency of Cox’s Likelihood Function for Censored Data.
           Journal of the American Statistical Association 72 (1977): 557–565.
    """

    def __init__(self, alpha=0, ties="breslow", n_iter=100, tol=1e-9, verbose=0):
        self.alpha = alpha
        self.ties = ties
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

        if isinstance(self.alpha, (numbers.Real, numbers.Integral)):
            alphas = numpy.empty(X.shape[1], dtype=numpy.float_)
            alphas[:] = self.alpha
        else:
            alphas = self.alpha

        alphas = check_array(alphas, ensure_2d=False, ensure_min_samples=0)
        if numpy.any(alphas < 0):
            raise ValueError("alpha must be positive, but was %r" % self.alpha)
        if alphas.shape[0] != X.shape[1]:
            raise ValueError(
                "Length alphas ({}) must match number of features ({}).".format(
                    alphas.shape[0], X.shape[1]))

        optimizer = CoxPHOptimizer(X, event, time, alphas, self.ties)

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
        cum_hazard : ndarray of :class:`sksurv.functions.StepFunction`, shape = (n_samples,)
            Predicted cumulative hazard functions.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from sksurv.datasets import load_whas500
        >>> from sksurv.linear_model import CoxPHSurvivalAnalysis

        Load the data.

        >>> X, y = load_whas500()
        >>> X = X.astype(float)

        Fit the model.

        >>> estimator = CoxPHSurvivalAnalysis().fit(X, y)

        Estimate the cumulative hazard function for the first 10 samples.

        >>> chf_funcs = estimator.predict_cumulative_hazard_function(X.iloc[:10])

        Plot the estimated cumulative hazard functions.

        >>> for fn in chf_funcs:
        ...     plt.step(fn.x, fn(fn.x), where="post")
        ...
        >>> plt.ylim(0, 1)
        >>> plt.show()
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
        survival : ndarray of :class:`sksurv.functions.StepFunction`, shape = (n_samples,)
            Predicted survival functions.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from sksurv.datasets import load_whas500
        >>> from sksurv.linear_model import CoxPHSurvivalAnalysis

        Load the data.

        >>> X, y = load_whas500()
        >>> X = X.astype(float)

        Fit the model.

        >>> estimator = CoxPHSurvivalAnalysis().fit(X, y)

        Estimate the survival function for the first 10 samples.

        >>> surv_funcs = estimator.predict_survival_function(X.iloc[:10])

        Plot the estimated survival functions.

        >>> for fn in surv_funcs:
        ...     plt.step(fn.x, fn(fn.x), where="post")
        ...
        >>> plt.ylim(0, 1)
        >>> plt.show()
        """
        return self._baseline_model.get_survival_function(self.predict(X))
