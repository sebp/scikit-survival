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

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import normalize as f_normalize
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import (
    assert_all_finite,
    check_is_fitted,
    check_non_negative,
    column_or_1d,
    validate_data,
)

from ..base import SurvivalAnalysisMixin
from ..util import check_array_survival
from ._coxnet import call_fit_coxnet
from .coxph import BreslowEstimator

__all__ = ["CoxnetSurvivalAnalysis"]


class CoxnetSurvivalAnalysis(BaseEstimator, SurvivalAnalysisMixin):
    r"""Cox's proportional hazard's model with elastic net penalty.

    See the :ref:`User Guide </user_guide/coxnet.ipynb>` and [1]_ for further description.

    Parameters
    ----------
    n_alphas : int, optional, default: 100
        Number of alphas along the regularization path.

    alphas : array-like or None, optional
        List of alphas where to compute the models.
        If ``None``, alphas are set automatically.

        In this case, the ``alphas`` sequence is determined by :math:`\alpha_\max`
        and ``alpha_min_ratio``. The latter determines the smallest alpha value
        :math:`\alpha_\min` in the generated alphas sequence such that
        ``alpha_min_ratio`` equals the ratio :math:`\frac{\alpha_\min}{\alpha_\max}`.
        The generated ``alphas`` sequence contains ``n_alphas`` values linear
        on the log scale from :math:`\alpha_\max` down to :math:`\alpha_\min`.
        :math:`\alpha_\max` is not user-specified but is computed from the
        input data.

    alpha_min_ratio : float or { "auto" }, optional, default: "auto"
        Determines the minimum alpha of the regularization path
        if ``alphas`` is ``None``. The smallest value for alpha
        is computed as the fraction of the maximum
        alpha (i.e. the smallest value for which all
        coefficients are zero), which is derived from the input data.

        If set to "auto", the value will depend on the
        sample size relative to the number of features:

        - If ``n_samples > n_features``, the default value is 0.0001.
        - If ``n_samples <= n_features``, the default value is 0.01.

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
        Default is 1 for all variables.

        Note: the penalty factors are internally rescaled to sum to
        `n_features`, and the alphas sequence will reflect this change.

    normalize : bool, optional, default: False
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
        The maximum number of iterations taken for the solver to converge.

    verbose : bool, optional, default: False
        Whether to print additional information during optimization.

    fit_baseline_model : bool, optional, default: False
        Whether to estimate baseline survival function
        and baseline cumulative hazard function for each alpha.
        If enabled, :meth:`predict_cumulative_hazard_function` and
        :meth:`predict_survival_function` can be used to obtain
        predicted  cumulative hazard function and survival function.

    Attributes
    ----------
    alphas_ : ndarray, shape=(n_alphas,)
        The actual sequence of alpha values used.

    alpha_min_ratio_ : float
        The inferred value of alpha_min_ratio.

    penalty_factor_ : ndarray, shape=(n_features,)
        The actual penalty factors used.

    coef_ : ndarray, shape=(n_features, n_alphas)
        Matrix of coefficients.

    offset_ : ndarray, shape=(n_alphas,)
        Bias term to account for non-centered features.

    deviance_ratio_ : ndarray, shape=(n_alphas,)
        The fraction of (null) deviance explained.
        The deviance is defined as :math:`2 \cdot (\text{loglike_sat} - \text{loglike})`,
        where `loglike_sat` is the log-likelihood for the saturated model
        (a model with a free parameter per observation). Null deviance is defined as
        :math:`2 \cdot (\text{loglike_sat} - \text{loglike(Null)})`;
        The NULL model is the model with all zero coefficients.
        Hence, ``deviance_ratio_`` is :math:`1 - \frac{\text{deviance}}{\text{null_deviance}}`.

    n_features_in_ : int
        Number of features seen during ``fit``.

    feature_names_in_ : ndarray, shape = (`n_features_in_`,)
        Names of features seen during ``fit``. Defined only when `X`
        has feature names that are all strings.

    unique_times_ : ndarray, shape = (n_unique_times,)
        Unique time points.

    References
    ----------
    .. [1] Simon N, Friedman J, Hastie T, Tibshirani R.
           Regularization paths for Cox’s proportional hazards model via coordinate descent.
           Journal of statistical software. 2011 Mar;39(5):1.
    """

    _parameter_constraints: dict = {
        "n_alphas": [Interval(numbers.Integral, 1, None, closed="left")],
        "alphas": ["array-like", None],
        "alpha_min_ratio": [Interval(numbers.Real, 0, None, closed="neither"), StrOptions({"auto"})],
        "l1_ratio": [Interval(numbers.Real, 0.0, 1.0, closed="right")],
        "penalty_factor": ["array-like", None],
        "normalize": ["boolean"],
        "copy_X": ["boolean"],
        "tol": [Interval(numbers.Real, 0, None, closed="left")],
        "max_iter": [Interval(numbers.Integral, 1, None, closed="left")],
        "verbose": ["verbose"],
        "fit_baseline_model": ["boolean"],
    }

    def __init__(
        self,
        *,
        n_alphas=100,
        alphas=None,
        alpha_min_ratio="auto",
        l1_ratio=0.5,
        penalty_factor=None,
        normalize=False,
        copy_X=True,
        tol=1e-7,
        max_iter=100000,
        verbose=False,
        fit_baseline_model=False,
    ):
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
        self.fit_baseline_model = fit_baseline_model

        self._baseline_models = None

    def _pre_fit(self, X, y):
        X = validate_data(self, X, ensure_min_samples=2, dtype=np.float64, copy=self.copy_X)
        event, time = check_array_survival(X, y)
        # center feature matrix
        X_offset = np.average(X, axis=0)
        X -= X_offset
        if self.normalize:
            X, X_scale = f_normalize(X, copy=False, axis=0, return_norm=True)
        else:
            X_scale = np.ones(X.shape[1], dtype=X.dtype)

        # sort descending
        o = np.argsort(-time, kind="mergesort")
        X = np.asfortranarray(X[o, :])
        event_num = event[o].astype(np.uint8)
        time = time[o].astype(np.float64)
        return X, event_num, time, X_offset, X_scale

    def _check_penalty_factor(self, n_features):
        if self.penalty_factor is None:
            penalty_factor = np.ones(n_features, dtype=np.float64)
        else:
            pf = column_or_1d(self.penalty_factor, warn=True)
            if pf.shape[0] != n_features:
                raise ValueError(
                    f"penalty_factor must be array of length n_features ({n_features}), but got {pf.shape[0]}"
                )
            assert_all_finite(pf, input_name="penalty_factor")
            check_non_negative(pf, "penalty_factor")
            penalty_factor = pf * n_features / pf.sum()
        return penalty_factor

    def _check_alphas(self):
        create_path = self.alphas is None
        if create_path:
            alphas = np.empty(int(self.n_alphas), dtype=np.float64)
        else:
            alphas = column_or_1d(self.alphas, warn=True)
            assert_all_finite(alphas, input_name="alphas")
            check_non_negative(alphas, "alphas")
        return alphas, create_path

    def _check_alpha_min_ratio(self, n_samples, n_features):
        alpha_min_ratio = self.alpha_min_ratio
        if isinstance(alpha_min_ratio, str) and self.alpha_min_ratio == "auto":
            if n_samples > n_features:
                alpha_min_ratio = 0.0001
            else:
                alpha_min_ratio = 0.01

        return alpha_min_ratio

    def _check_params(self, n_samples, n_features):
        self._validate_params()

        penalty_factor = self._check_penalty_factor(n_features)

        alphas, create_path = self._check_alphas()

        alpha_min_ratio = self._check_alpha_min_ratio(n_samples, n_features)

        return create_path, alphas.astype(np.float64), penalty_factor.astype(np.float64), alpha_min_ratio

    def fit(self, X, y):
        """Fit estimator.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data matrix

        y : structured array, shape = (n_samples,)
            A structured array with two fields. The first field is a boolean
            where ``True`` indicates an event and ``False`` indicates right-censoring.
            The second field is a float with the time of event or time of censoring.

        Returns
        -------
        self
        """
        X, event_num, time, X_offset, X_scale = self._pre_fit(X, y)
        create_path, alphas, penalty, alpha_min_ratio = self._check_params(*X.shape)

        coef, alphas, deviance_ratio, n_iter = call_fit_coxnet(
            X,
            time,
            event_num,
            penalty,
            alphas,
            create_path,
            alpha_min_ratio,
            self.l1_ratio,
            int(self.max_iter),
            self.tol,
            self.verbose,
        )
        assert np.isfinite(coef).all()

        if np.all(np.absolute(coef) < np.finfo(float).eps):
            warnings.warn("all coefficients are zero, consider decreasing alpha.", stacklevel=2)

        if n_iter >= self.max_iter:
            warnings.warn(
                "Optimization terminated early, you might want"
                f" to increase the number of iterations (max_iter={self.max_iter}).",
                category=ConvergenceWarning,
                stacklevel=2,
            )

        coef /= X_scale[:, np.newaxis]

        if self.fit_baseline_model:
            predictions = np.dot(X, coef)
            self._baseline_models = tuple(
                BreslowEstimator().fit(predictions[:, i], event_num, time) for i in range(coef.shape[1])
            )
        else:
            self._baseline_models = None

        self.alphas_ = alphas
        self.alpha_min_ratio_ = alpha_min_ratio
        self.penalty_factor_ = penalty
        self.coef_ = coef
        self.deviance_ratio_ = deviance_ratio
        self.offset_ = np.dot(X_offset, coef)
        return self

    def _get_coef(self, alpha):
        check_is_fitted(self, "coef_")

        if alpha is None:
            coef = self.coef_[:, -1], self.offset_[-1]
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
            elif alpha - val < np.finfo(float).eps:
                coef_idx = i
                exact = True
                break

        if coef_idx is None:
            coef = self.coef_[:, 0]
            offset = self.offset_[0]
        elif exact or coef_idx == len(self.alphas_) - 1:
            coef = self.coef_[:, coef_idx]
            offset = self.offset_[coef_idx]
        else:
            # interpolate between coefficients
            a1 = self.alphas_[coef_idx + 1]
            a2 = self.alphas_[coef_idx]
            frac = (alpha - a1) / (a2 - a1)
            coef = frac * self.coef_[:, coef_idx] + (1.0 - frac) * self.coef_[:, coef_idx + 1]
            offset = frac * self.offset_[coef_idx] + (1.0 - frac) * self.offset_[coef_idx + 1]

        return coef, offset

    def predict(self, X, alpha=None):
        """Predict risk scores.

        The risk score is the linear predictor of the model,
        computed as the dot product of the input features `X` and the
        estimated coefficients `coef_`. A higher score indicates a
        higher risk of experiencing the event.

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
        risk_score : array, shape = (n_samples,)
            Predicted risk scores.
        """
        X = validate_data(self, X, reset=False)
        coef, offset = self._get_coef(alpha)
        return np.dot(X, coef) - offset

    def _get_baseline_model(self, alpha):
        check_is_fitted(self, "coef_")
        if self._baseline_models is None:
            raise ValueError("`fit` must be called with the fit_baseline_model option set to True.")

        if alpha is None:
            baseline_model = self._baseline_models[-1]
        else:
            is_close = np.isclose(alpha, self.alphas_)
            if is_close.any():
                idx = np.flatnonzero(is_close)[0]
                baseline_model = self._baseline_models[idx]
            else:
                raise ValueError(f"alpha must be one value of alphas_: {self.alphas_}")

        return baseline_model

    def predict_cumulative_hazard_function(self, X, alpha=None, return_array=False):
        """Predict cumulative hazard function.

        Only available if :meth:`fit` has been called with `fit_baseline_model = True`.

        The cumulative hazard function for an individual
        with feature vector :math:`x_\\alpha` is defined as

        .. math::

            H(t \\mid x_\\alpha) = \\exp(x_\\alpha^\\top \\beta) H_0(t) ,

        where :math:`H_0(t)` is the baseline hazard function,
        estimated by Breslow's estimator.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data matrix.

        alpha : float, optional
            Constant that multiplies the penalty terms. The same alpha as used during training
            must be specified. If set to ``None``, the last alpha in the solution path is used.

        return_array : bool, default: False
            Whether to return a single array of cumulative hazard values
            or a list of step functions.

            If `False`, a list of :class:`sksurv.functions.StepFunction`
            objects is returned.

            If `True`, a 2d-array of shape `(n_samples, n_unique_times)` is
            returned, where `n_unique_times` is the number of unique
            event times in the training data. Each row represents the cumulative
            hazard function of an individual evaluated at `unique_times_`.

        Returns
        -------
        cum_hazard : ndarray
            If `return_array` is `False`, an array of `n_samples`
            :class:`sksurv.functions.StepFunction` instances is returned.

            If `return_array` is `True`, a numeric array of shape
            `(n_samples, n_unique_times_)` is returned.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from sksurv.datasets import load_breast_cancer
        >>> from sksurv.preprocessing import OneHotEncoder
        >>> from sksurv.linear_model import CoxnetSurvivalAnalysis

        Load and prepare the data.

        >>> X, y = load_breast_cancer()
        >>> X = OneHotEncoder().fit_transform(X)

        Fit the model.

        >>> estimator = CoxnetSurvivalAnalysis(l1_ratio=0.99, fit_baseline_model=True)
        >>> estimator.fit(X, y)

        Estimate the cumulative hazard function for one sample and the five highest alpha.

        >>> chf_funcs = {}
        >>> for alpha in estimator.alphas_[:5]:
        ...     chf_funcs[alpha] = estimator.predict_cumulative_hazard_function(
        ...         X.iloc[:1], alpha=alpha)
        ...

        Plot the estimated cumulative hazard functions.

        >>> for alpha, chf_alpha in chf_funcs.items():
        ...     for fn in chf_alpha:
        ...         plt.step(fn.x, fn(fn.x), where="post",
        ...                  label=f"alpha = {alpha:.3f}")
        ...
        >>> plt.ylim(0, 1)
        >>> plt.legend()
        >>> plt.show()
        """
        baseline_model = self._get_baseline_model(alpha)
        return self._predict_cumulative_hazard_function(baseline_model, self.predict(X, alpha=alpha), return_array)

    def predict_survival_function(self, X, alpha=None, return_array=False):
        r"""Predict survival function.

        Only available if :meth:`fit` has been called with `fit_baseline_model = True`.

        The survival function for an individual
        with feature vector :math:`x_\alpha` is defined as

        .. math::

            S(t \mid x_\alpha) = S_0(t)^{\exp(x_\alpha^\top \beta)} ,

        where :math:`S_0(t)` is the baseline survival function,
        estimated by Breslow's estimator.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data matrix.

        alpha : float, optional
            Constant that multiplies the penalty terms. The same alpha as used during training
            must be specified. If set to ``None``, the last alpha in the solution path is used.

        return_array : bool, default: False
            Whether to return a single array of survival probabilities
            or a list of step functions.

            If `False`, a list of :class:`sksurv.functions.StepFunction`
            objects is returned.

            If `True`, a 2d-array of shape `(n_samples, n_unique_times)` is
            returned, where `n_unique_times` is the number of unique
            event times in the training data. Each row represents the survival
            function of an individual evaluated at `unique_times_`.

        Returns
        -------
        survival : ndarray
            If `return_array` is `False`, an array of `n_samples`
            :class:`sksurv.functions.StepFunction` instances is returned.

            If `return_array` is `True`, a numeric array of shape
            `(n_samples, n_unique_times_)` is returned.


        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from sksurv.datasets import load_breast_cancer
        >>> from sksurv.preprocessing import OneHotEncoder
        >>> from sksurv.linear_model import CoxnetSurvivalAnalysis

        Load and prepare the data.

        >>> X, y = load_breast_cancer()
        >>> X = OneHotEncoder().fit_transform(X)

        Fit the model.

        >>> estimator = CoxnetSurvivalAnalysis(l1_ratio=0.99, fit_baseline_model=True)
        >>> estimator.fit(X, y)

        Estimate the survival function for one sample and the five highest alpha.

        >>> surv_funcs = {}
        >>> for alpha in estimator.alphas_[:5]:
        ...     surv_funcs[alpha] = estimator.predict_survival_function(
        ...         X.iloc[:1], alpha=alpha)
        ...

        Plot the estimated survival functions.

        >>> for alpha, surv_alpha in surv_funcs.items():
        ...     for fn in surv_alpha:
        ...         plt.step(fn.x, fn(fn.x), where="post",
        ...                  label=f"alpha = {alpha:.3f}")
        ...
        >>> plt.ylim(0, 1)
        >>> plt.legend()
        >>> plt.show()
        """
        baseline_model = self._get_baseline_model(alpha)
        return self._predict_survival_function(baseline_model, self.predict(X, alpha=alpha), return_array)

    @property
    def unique_times_(self):
        return self._get_baseline_model(None).unique_times_
