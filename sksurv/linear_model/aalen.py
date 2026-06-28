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
"""Aalen Additive Hazard Model.

References
----------
.. [1] Aalen, O. O. A linear regression model for the analysis of life times.
       Statistics in Medicine 8 (1989): 907–925.

.. [2] Lin, D. Y. & Ying, Z. Semiparametric analysis of the additive risk model.
       Biometrika 81 (1994): 61–71.
"""
import numbers

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted, validate_data

from .._dataframe import ensure_eager_dataframe
from ..base import SurvivalAnalysisMixin
from ..functions import StepFunction
from ..util import check_array_survival

__all__ = ["AalenAdditiveFitter"]


class AalenAdditiveFitter(BaseEstimator, SurvivalAnalysisMixin):
    """Aalen Additive Hazard Model.

    The Aalen model is a non-parametric additive hazards model that assumes
    the hazard function is a linear combination of time-varying coefficients:

    .. math::

        h(t \\mid x) = \\beta_0(t) + x_1 \\beta_1(t) + \\ldots + x_p \\beta_p(t)

    Cumulative coefficients :math:`B_j(t) = \\int_0^t \\beta_j(s)\\,ds` are estimated
    nonparametrically at each event time via weighted least squares. The survival
    function for subject :math:`x` (with intercept prepended) is:

    .. math::

        S(t \\mid x) = \\exp\\bigl(-\\tilde{x}^\\top B(t)\\bigr)

    where :math:`\\tilde{x} = (1, x_1, \\ldots, x_p)^\\top`.

    Parameters
    ----------
    fit_baseline_model : bool, optional, default: True
        If ``True``, estimate the intercept (baseline hazard) term :math:`\\beta_0(t)`.
        If ``False``, no intercept column is added (not recommended in most settings).

    alpha : float, optional, default: 0.0
        Ridge regularization parameter added to :math:`X_k^\\top X_k` at each event
        time for numerical stability. Larger values increase stability at the cost
        of bias.

    Attributes
    ----------
    unique_times_ : ndarray, shape = (n_event_times,)
        Unique event times observed during training (sorted ascending).

    cumulative_coefficients_ : ndarray, shape = (n_event_times, n_features + 1)
        Estimated cumulative coefficients :math:`B(t_k)` at each event time.
        The first column corresponds to the intercept (baseline), the remaining
        columns to the covariates in the order they appear in ``X``.

    cumulative_coefficient_functions_ : dict of str -> StepFunction
        A mapping from feature name (or ``"Intercept"`` for the baseline) to a
        :class:`sksurv.functions.StepFunction` representing the cumulative
        coefficient :math:`B_j(t)` over time.

    cum_baseline_hazard_ : :class:`sksurv.functions.StepFunction`
        Estimated cumulative baseline hazard function :math:`B_0(t)` (the intercept
        cumulative coefficient).

    coef_var_ : ndarray, shape = (n_event_times, n_features + 1)
        Pointwise variance estimates for the cumulative coefficients.

    n_features_in_ : int
        Number of features seen during ``fit``.

    feature_names_in_ : ndarray of str, shape = (n_features_in_,)
        Names of features seen during ``fit``. Only defined when ``X`` has feature
        names that are all strings.

    References
    ----------
    .. [1] Aalen, O. O. A linear regression model for the analysis of life times.
           Statistics in Medicine 8 (1989): 907–925.

    .. [2] Lin, D. Y. & Ying, Z. Semiparametric analysis of the additive risk model.
           Biometrika 81 (1994): 61–71.

    Examples
    --------
    >>> import numpy as np
    >>> from sksurv.datasets import load_whas500
    >>> from sksurv.linear_model import AalenAdditiveFitter
    >>> from sksurv.preprocessing import OneHotEncoder

    Load and encode the WHAS500 data:

    >>> X, y = load_whas500()
    >>> X = X.astype(float)

    Fit the model:

    >>> est = AalenAdditiveFitter(alpha=0.1)
    >>> est.fit(X, y)
    AalenAdditiveFitter(alpha=0.1)

    Predict survival functions for the first five subjects:

    >>> fns = est.predict_survival_function(X.iloc[:5])
    >>> fns[0].x[:3]
    array([1., 2., 3.])
    """

    _parameter_constraints: dict = {
        "fit_baseline_model": ["boolean"],
        "alpha": [Interval(numbers.Real, 0, None, closed="left")],
    }

    def __init__(self, fit_baseline_model=True, alpha=0.0):
        self.fit_baseline_model = fit_baseline_model
        self.alpha = alpha

    def fit(self, X, y):
        """Fit the Aalen Additive Hazard Model to the given data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data matrix. Rows are subjects, columns are covariates.

        y : structured array, shape = (n_samples,)
            A structured array with two fields. The first field is a boolean
            where ``True`` indicates an event (death / failure) and ``False``
            indicates right censoring. The second field is a float with the
            time of the event or the time of censoring.

        Returns
        -------
        self : AalenAdditiveFitter
            Fitted estimator.
        """
        self._validate_params()

        X = validate_data(self, ensure_eager_dataframe(X), ensure_min_samples=2, dtype=np.float64)
        event, time = check_array_survival(X, y)

        n_samples, n_features = X.shape
        n_coefs = n_features + 1  # intercept + covariates

        # Augment X with an intercept column (leftmost)
        X_aug = np.column_stack([np.ones(n_samples), X])  # shape (n, p+1)

        # Sort by time ascending for risk-set construction
        order = np.argsort(time, kind="mergesort")
        time_sorted = time[order]
        event_sorted = event[order]
        X_aug_sorted = X_aug[order]

        # Identify unique event times
        unique_event_times = np.unique(time_sorted[event_sorted])

        n_times = len(unique_event_times)

        # Storage for incremental coefficients dB at each event time
        dB = np.zeros((n_times, n_coefs))
        var_increments = np.zeros((n_times, n_coefs))  # variance increments

        alpha = float(self.alpha)

        for k, t_k in enumerate(unique_event_times):
            # Risk set: all subjects still at risk at t_k (time >= t_k)
            at_risk_mask = time_sorted >= t_k
            X_k = X_aug_sorted[at_risk_mask]  # shape (|R_k|, p+1)
            n_risk = X_k.shape[0]

            # Event indicators for subjects in risk set at this time
            dN_k = ((time_sorted[at_risk_mask] == t_k) & event_sorted[at_risk_mask]).astype(float)
            n_events_k = dN_k.sum()

            if n_events_k == 0:
                continue  # shouldn't happen since t_k is an event time

            # Gram matrix with optional ridge regularization: X_k'X_k + n*alpha*I
            gram = X_k.T @ X_k
            if alpha > 0.0:
                gram += n_samples * alpha * np.eye(n_coefs)

            # Solve (X_k'X_k + reg*I) dB_k = X_k' dN_k
            # Use lstsq for robustness
            rhs = X_k.T @ dN_k
            try:
                dB_k = np.linalg.solve(gram, rhs)
            except np.linalg.LinAlgError:
                dB_k, _, _, _ = np.linalg.lstsq(gram, rhs, rcond=None)

            dB[k] = dB_k

            # Variance increment: diag(gram_inv) * n_events_k / n_risk^2
            try:
                gram_inv = np.linalg.inv(gram)
                diag_gram_inv = np.diag(gram_inv)
            except np.linalg.LinAlgError:
                diag_gram_inv = np.zeros(n_coefs)

            var_increments[k] = diag_gram_inv * (n_events_k / n_risk**2)

        # Cumulative coefficients: B(t_k) = sum_{t_j <= t_k} dB_j
        cum_coefs = np.cumsum(dB, axis=0)  # shape (n_times, n_coefs)
        cum_var = np.cumsum(var_increments, axis=0)  # shape (n_times, n_coefs)

        self.unique_times_ = unique_event_times
        self.cumulative_coefficients_ = cum_coefs
        self.coef_var_ = cum_var

        # Build StepFunction objects for each coefficient
        feature_names = self._get_feature_names()
        coef_names = ["Intercept"] + list(feature_names)

        self.cumulative_coefficient_functions_ = {}
        for j, name in enumerate(coef_names):
            self.cumulative_coefficient_functions_[name] = StepFunction(
                x=unique_event_times,
                y=cum_coefs[:, j],
            )

        # Cumulative baseline hazard = cumulative intercept coefficient
        self.cum_baseline_hazard_ = self.cumulative_coefficient_functions_["Intercept"]

        return self

    def _get_feature_names(self):
        """Return feature names from training data, or generic names."""
        if hasattr(self, "feature_names_in_"):
            return list(self.feature_names_in_)
        return [f"x{j}" for j in range(self.n_features_in_)]

    def _eval_cumulative_hazard(self, X_aug):
        """Evaluate cumulative hazard x'B(t) for each sample.

        Parameters
        ----------
        X_aug : ndarray, shape = (n_samples, n_coefs)
            Augmented design matrix with intercept prepended.

        Returns
        -------
        cum_hazard_values : ndarray, shape = (n_samples, n_event_times)
            Cumulative hazard H(t_k|x) for each sample and event time.
        """
        # cum_coefs shape: (n_times, n_coefs)
        # X_aug shape: (n_samples, n_coefs)
        # Result: (n_samples, n_times)
        return X_aug @ self.cumulative_coefficients_.T  # (n_samples, n_times)

    def predict_cumulative_hazard_function(self, X, return_array=False):
        """Predict cumulative hazard function for each sample.

        The cumulative hazard for subject :math:`x` at time :math:`t` is:

        .. math::

            H(t \\mid x) = \\tilde{x}^\\top B(t)

        where :math:`\\tilde{x} = (1, x_1, \\ldots, x_p)^\\top`.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data matrix.

        return_array : bool, optional, default: False
            If ``False`` (default), return an array of
            :class:`sksurv.functions.StepFunction` objects, one per sample.
            If ``True``, return a 2-D array of shape
            ``(n_samples, n_unique_times_)`` with cumulative hazard values.

        Returns
        -------
        cum_hazard : ndarray
            If ``return_array`` is ``False``, an array of
            :class:`sksurv.functions.StepFunction` instances, shape ``(n_samples,)``.
            If ``return_array`` is ``True``, a numeric array of shape
            ``(n_samples, n_unique_times_)``.
        """
        check_is_fitted(self, "cumulative_coefficients_")
        X = validate_data(self, ensure_eager_dataframe(X), reset=False, dtype=np.float64)
        n_samples = X.shape[0]
        X_aug = np.column_stack([np.ones(n_samples), X])

        ch_values = self._eval_cumulative_hazard(X_aug)  # (n_samples, n_times)

        if return_array:
            return ch_values

        funcs = np.empty(n_samples, dtype=object)
        for i in range(n_samples):
            funcs[i] = StepFunction(x=self.unique_times_, y=ch_values[i])
        return funcs

    def predict_survival_function(self, X, return_array=False):
        """Predict survival function for each sample.

        The survival function for subject :math:`x` at time :math:`t` is:

        .. math::

            S(t \\mid x) = \\exp\\bigl(-\\tilde{x}^\\top B(t)\\bigr)

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data matrix.

        return_array : bool, optional, default: False
            If ``False`` (default), return an array of
            :class:`sksurv.functions.StepFunction` objects, one per sample.
            If ``True``, return a 2-D array of shape
            ``(n_samples, n_unique_times_)`` with survival probabilities.

        Returns
        -------
        survival : ndarray
            If ``return_array`` is ``False``, an array of
            :class:`sksurv.functions.StepFunction` instances, shape ``(n_samples,)``.
            If ``return_array`` is ``True``, a numeric array of shape
            ``(n_samples, n_unique_times_)``.
        """
        check_is_fitted(self, "cumulative_coefficients_")
        X = validate_data(self, ensure_eager_dataframe(X), reset=False, dtype=np.float64)
        n_samples = X.shape[0]
        X_aug = np.column_stack([np.ones(n_samples), X])

        ch_values = self._eval_cumulative_hazard(X_aug)  # (n_samples, n_times)
        surv_values = np.exp(-np.clip(ch_values, a_min=0, a_max=None))  # clip for numerical safety

        if return_array:
            return surv_values

        funcs = np.empty(n_samples, dtype=object)
        for i in range(n_samples):
            funcs[i] = StepFunction(x=self.unique_times_, y=surv_values[i])
        return funcs

    def predict(self, X):
        """Predict risk scores (negative median survival time approximation).

        This returns the cumulative hazard evaluated at the last observed event
        time, which serves as a monotone risk score. Higher values indicate higher
        risk.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data matrix.

        Returns
        -------
        risk_score : ndarray, shape = (n_samples,)
            Predicted risk scores. Larger values indicate higher risk.
        """
        check_is_fitted(self, "cumulative_coefficients_")
        X = validate_data(self, ensure_eager_dataframe(X), reset=False, dtype=np.float64)
        n_samples = X.shape[0]
        X_aug = np.column_stack([np.ones(n_samples), X])

        # Use cumulative hazard at last event time as risk score
        ch_values = self._eval_cumulative_hazard(X_aug)  # (n_samples, n_times)
        return ch_values[:, -1]
