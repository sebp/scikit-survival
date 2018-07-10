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
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import normalize as f_normalize
from sklearn.utils.validation import assert_all_finite, check_array, check_is_fitted, check_non_negative, column_or_1d

from ..base import SurvivalAnalysisMixin
from ..util import check_arrays_survival
from ._coxnet import call_fit_coxnet

__all__ = ['CoxnetSurvivalAnalysis']


class CoxnetSurvivalAnalysis(BaseEstimator, SurvivalAnalysisMixin):
    """Cox's proportional hazard's model with elastic net penalty.

    Parameters
    ----------
    n_alphas : int, optional, default: 100
        Number of alphas along the regularization path.

    alphas : array-like or None, optional
        List of alphas where to compute the models.
        If ``None`` alphas are set automatically.

    alpha_min_ratio : float, optional, default 0.0001
        Determines minimum alpha of the regularization path
        if ``alphas`` is ``None``. The smallest value for alpha
        is computed as the fraction of the data derived maximum
        alpha (i.e. the smallest value for which all
        coefficients are zero).

    l1_ratio : float, optional, default: 0.5
        The ElasticNet mixing parameter, with ``0 < l1_ratio <= 1``.
        For ``l1_ratio = 0`` the penalty is an L2 penalty.
        For ``l1_ratio = 1`` it is an L1 penalty.
        For ``0 < l1_ratio < 1``, the penalty is a combination of L1 and L2.

    penalty_factor : array-like or None, optional
        Separate penalty factors can be applied to each coefficient.
        This is a number that multiplies alpha to allow differential
        shrinkage.  Can be 0 for some variables, which implies no shrinkage,
        and that variable is always included in the model.
        Default is 1 for all variables. Note: the penalty factors are
        internally rescaled to sum to n_features, and the alphas sequence
        will reflect this change.

    normalize : boolean, optional, default: False
        If True, the features X will be normalized before optimization by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

    copy_X : boolean, optional, default: True
        If ``True``, X will be copied; else, it may be overwritten.

    tol : float, optional, default: 1e-7
        The tolerance for the optimization: optimization continues
        until all updates are smaller than ``tol``.

    max_iter : int, optional, default: 100000
        The maximum number of iterations.

    verbose : bool, optional, default: False
        Whether to print additional information during optimization.

    Attributes
    ----------
    alphas_ : ndarray, shape=(n_alphas,)
        The actual sequence of alpha values used.

    penalty_factor_ : ndarray, shape=(n_features,)
        The actual penalty factors used.

    coef_ : ndarray, shape=(n_features, n_alphas)
        Matrix of coefficients.

    deviance_ratio_ : ndarray, shape=(n_alphas,)
        The fraction of (null) deviance explained.

    References
    ----------
    .. [1] Simon N, Friedman J, Hastie T, Tibshirani R.
           Regularization paths for Cox’s proportional hazards model via coordinate descent.
           Journal of statistical software. 2011 Mar;39(5):1.
    """

    def __init__(self, n_alphas=100, alphas=None, alpha_min_ratio=0.0001, l1_ratio=0.5,
                 penalty_factor=None, normalize=False, copy_X=True,
                 tol=1e-7, max_iter=100000, verbose=False):
        self.n_alphas = n_alphas
        self.alphas = alphas
        self.alpha_min_ratio = alpha_min_ratio
        self.l1_ratio = l1_ratio
        self.penalty_factor = penalty_factor
        self.normalize = normalize
        self.copy_X = copy_X
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose

    def _pre_fit(self, X, y):
        X, event, time = check_arrays_survival(X, y, copy=self.copy_X)
        # center feature matrix
        X_offset = numpy.average(X, axis=0)
        X -= X_offset
        if self.normalize:
            X = f_normalize(X, copy=False, axis=0)

        # sort descending
        o = numpy.argsort(-time, kind="mergesort")
        X = numpy.asfortranarray(X[o, :])
        event_num = event[o].astype(numpy.uint8)
        time = time[o].astype(numpy.float64)
        return X, event_num, time

    def _check_params(self, n_features):
        if not 0 < self.l1_ratio <= 1:
            raise ValueError("l1_ratio must be in interval ]0;1], but was %f" % self.l1_ratio)

        if self.tol <= 0:
            raise ValueError("tolerance must be positive, but was %f" % self.tol)

        if self.penalty_factor is None:
            penalty_factor = numpy.ones(n_features, dtype=numpy.float64)
        else:
            pf = column_or_1d(self.penalty_factor, warn=True)
            if pf.shape[0] != n_features:
                raise ValueError("penalty_factor must be array of length n_features (%d), "
                                 "but got %d" % (n_features, pf.shape[0]))
            assert_all_finite(pf)
            check_non_negative(pf, "penalty_factor")
            penalty_factor = pf * n_features / pf.sum()
            assert_all_finite(penalty_factor)

        create_path = self.alphas is None
        if create_path:
            if self.n_alphas <= 0:
                raise ValueError("n_alphas must be a positive integer")

            alphas = numpy.empty(int(self.n_alphas), dtype=numpy.float64)
        else:
            alphas = column_or_1d(self.alphas, warn=True)
            assert_all_finite(alphas)
            check_non_negative(alphas, "alphas")
            assert_all_finite(alphas)

        if self.max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")

        return create_path, alphas.astype(numpy.float64), penalty_factor.astype(numpy.float64)

    def fit(self, X, y):
        """Fit estimator.

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
        X, event_num, time = self._pre_fit(X, y)
        create_path, alphas, penalty = self._check_params(X.shape[1])

        coef, alphas, deviance_ratio, n_iter = call_fit_coxnet(
            X, time, event_num, penalty, alphas, create_path,
            self.alpha_min_ratio, self.l1_ratio, int(self.max_iter),
            self.tol, self.verbose)
        assert numpy.isfinite(coef).all()

        if numpy.all(numpy.absolute(coef) < numpy.finfo(numpy.float).eps):
            warnings.warn('all coefficients are zero, consider decreasing alpha.',
                          stacklevel=2)

        if n_iter >= self.max_iter:
            warnings.warn('Optimization terminated early, you might want'
                          ' to increase the number of iterations (max_iter=%d).'
                          % self.max_iter,
                          category=ConvergenceWarning,
                          stacklevel=2)

        self.alphas_ = alphas
        self.penalty_factor_ = penalty
        self.coef_ = coef
        self.deviance_ratio_ = deviance_ratio
        return self

    def _get_coef(self, alpha):
        check_is_fitted(self, "coef_")

        if alpha is None:
            coef = self.coef_[:, -1]
        else:
            coef = self._interpolate_coefficients(alpha)
        return coef

    def _interpolate_coefficients(self, alpha):
        """Interpolate coefficients by calculating the weighted average of coefficient vectors corresponding to
        neighbors of alpha in the list of alphas constructed during training."""
        exact = False
        coef_idx = None
        for i, val in enumerate(self.alphas_):
            if val > alpha:
                coef_idx = i
            elif alpha - val < numpy.finfo(numpy.float).eps:
                coef_idx = i
                exact = True
                break

        if coef_idx is None:
            coef = self.coef_[:, 0]
        elif exact or coef_idx == len(self.alphas_) - 1:
            coef = self.coef_[:, coef_idx]
        else:
            # interpolate between coefficients
            a1 = self.alphas_[coef_idx + 1]
            a2 = self.alphas_[coef_idx]
            frac = (alpha - a1) / (a2 - a1)
            coef = frac * self.coef_[:, coef_idx] + (1.0 - frac) * self.coef_[:, coef_idx + 1]

        return coef

    def predict(self, X, alpha=None):
        """The linear predictor of the model.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test data of which to calculate log-likelihood from

        alpha : float, optional
            Constant that multiplies the penalty terms. If the same alpha was used during training, exact
            coefficients are used, otherwise coefficients are interpolated from the closest alpha values that
            were used during training. If set to ``None``, the last alpha in the solution path is used.

        Returns
        -------
        T : array, shape = (n_samples,)
            The predicted decision function
        """
        X = check_array(X)
        coef = self._get_coef(alpha)
        return numpy.dot(X, coef)
