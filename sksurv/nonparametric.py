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

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_array, check_consistent_length, check_is_fitted

from .util import check_y_survival

__all__ = [
    "CensoringDistributionEstimator",
    "kaplan_meier_estimator",
    "nelson_aalen_estimator",
    "ipc_weights",
    "SurvivalFunctionEstimator",
    "cumulative_incidence_competing_risks",
]


def _compute_counts(event, time, order=None):
    """Count right censored and uncensored samples at each unique time point.

    Parameters
    ----------
    event : array
        Boolean event indicator.
        Integer in the case of multiple risks.
        Zero means right-censored event.
        Positive values for each of the possible risk events.

    time : array
        Survival time or time of censoring.

    order : array or None
        Indices to order time in ascending order.
        If None, order will be computed.

    Returns
    -------
    times : array
        Unique time points.

    n_events : array
        Number of events at each time point.
        2D array with shape `(n_unique_time_points, n_risks + 1)` in the case of competing risks.

    n_at_risk : array
        Number of samples that have not been censored or have not had an event at each time point.

    n_censored : array
        Number of censored samples at each time point.
    """
    n_samples = event.shape[0]
    n_risks = event.max() if (np.issubdtype(event.dtype, np.integer) and event.max() > 1) else 0

    if order is None:
        order = np.argsort(time, kind="mergesort")

    uniq_times = np.empty(n_samples, dtype=time.dtype)
    uniq_events = np.empty((n_samples, n_risks + 1), dtype=int)
    uniq_counts = np.empty(n_samples, dtype=int)

    i = 0
    prev_val = time[order[0]]
    j = 0
    while True:
        count_event = np.zeros(n_risks + 1, dtype=int)
        count = 0
        while i < n_samples and prev_val == time[order[i]]:
            event_type = event[order[i]]
            if event_type:
                count_event[0] += 1
                if n_risks:
                    count_event[event_type] += 1

            count += 1
            i += 1

        uniq_times[j] = prev_val
        uniq_events[j] = count_event
        uniq_counts[j] = count
        j += 1

        if i == n_samples:
            break

        prev_val = time[order[i]]

    times = np.resize(uniq_times, j)
    total_count = np.resize(uniq_counts, j)
    if n_risks:
        n_events = np.resize(uniq_events, (j, n_risks + 1))
        n_censored = total_count - n_events[:, 0]
    else:
        n_events = np.resize(uniq_events, j)
        n_censored = total_count - n_events

    # offset cumulative sum by one
    total_count = np.r_[0, total_count]
    n_at_risk = n_samples - np.cumsum(total_count)

    return times, n_events, n_at_risk[:-1], n_censored


def _compute_counts_truncated(event, time_enter, time_exit):
    """Compute counts for left truncated and right censored survival data.

    Parameters
    ----------
    event : array
        Boolean event indicator.

    time_start : array
        Time when a subject entered the study.

    time_exit : array
        Time when a subject left the study due to an
        event or censoring.

    Returns
    -------
    times : array
        Unique time points.

    n_events : array
        Number of events at each time point.

    n_at_risk : array
        Number of samples that are censored or have an event at each time point.
    """
    if (time_enter > time_exit).any():
        raise ValueError("exit time must be larger start time for all samples")

    n_samples = event.shape[0]

    uniq_times = np.sort(np.unique(np.r_[time_enter, time_exit]), kind="mergesort")
    total_counts = np.empty(len(uniq_times), dtype=int)
    event_counts = np.empty(len(uniq_times), dtype=int)

    order_enter = np.argsort(time_enter, kind="mergesort")
    order_exit = np.argsort(time_exit, kind="mergesort")
    s_time_enter = time_enter[order_enter]
    s_time_exit = time_exit[order_exit]

    t0 = uniq_times[0]
    # everything larger is included
    idx_enter = np.searchsorted(s_time_enter, t0, side="right")
    # everything smaller is excluded
    idx_exit = np.searchsorted(s_time_exit, t0, side="left")

    total_counts[0] = idx_enter
    # except people die on the day they enter
    event_counts[0] = 0

    for i in range(1, len(uniq_times)):
        ti = uniq_times[i]

        while idx_enter < n_samples and s_time_enter[idx_enter] < ti:
            idx_enter += 1

        while idx_exit < n_samples and s_time_exit[idx_exit] < ti:
            idx_exit += 1

        risk_set = np.setdiff1d(order_enter[:idx_enter], order_exit[:idx_exit], assume_unique=True)
        total_counts[i] = len(risk_set)

        count_event = 0
        k = idx_exit
        while k < n_samples and s_time_exit[k] == ti:
            if event[order_exit[k]]:
                count_event += 1
            k += 1
        event_counts[i] = count_event

    return uniq_times, event_counts, total_counts


def _ci_logmlog(s, sigma_t, conf_level):
    r"""Compute the pointwise log-minus-log transformed confidence intervals.
    s refers to the prob_survival or the cum_inc (for the competing risks case).
    sigma_t is the square root of the variance of the log of the estimator of s.

    .. math::

        \sigma_t = \mathrm{Var}(\log(\hat{S}(t)))
    """
    eps = np.finfo(s.dtype).eps
    mask = s > eps
    log_p = np.zeros_like(s)
    np.log(s, where=mask, out=log_p)
    theta = np.zeros_like(s)
    np.true_divide(sigma_t, log_p, where=log_p < -eps, out=theta)

    z = stats.norm.isf((1.0 - conf_level) / 2.0)
    theta = z * np.multiply.outer([-1, 1], theta)
    ci = np.exp(np.exp(theta) * log_p)
    ci[:, ~mask] = 0.0
    return ci


def _km_ci_estimator(prob_survival, ratio_var, conf_level, conf_type):
    if conf_type not in {"log-log"}:
        raise ValueError(f"conf_type must be None or a str among {{'log-log'}}, but was {conf_type!r}")

    if not isinstance(conf_level, numbers.Real) or not np.isfinite(conf_level) or conf_level <= 0 or conf_level >= 1.0:
        raise ValueError(f"conf_level must be a float in the range (0.0, 1.0), but was {conf_level!r}")

    sigma = np.sqrt(np.cumsum(ratio_var))
    ci = _ci_logmlog(prob_survival, sigma, conf_level)
    return ci


def kaplan_meier_estimator(
    event,
    time_exit,
    time_enter=None,
    time_min=None,
    reverse=False,
    conf_level=0.95,
    conf_type=None,
):
    """Kaplan-Meier estimator of survival function.

    See [1]_ for further description.

    Parameters
    ----------
    event : array-like, shape = (n_samples,)
        Contains binary event indicators.

    time_exit : array-like, shape = (n_samples,)
        Contains event/censoring times.

    time_enter : array-like, shape = (n_samples,), optional
        Contains time when each individual entered the study for
        left truncated survival data.

    time_min : float, optional
        Compute estimator conditional on survival at least up to
        the specified time.

    reverse : bool, optional, default: False
        Whether to estimate the censoring distribution.
        When there are ties between times at which events are observed,
        then events come first and are subtracted from the denominator.
        Only available for right-censored data, i.e. `time_enter` must
        be None.

    conf_level : float, optional, default: 0.95
        The level for a two-sided confidence interval on the survival curves.

    conf_type : None or {'log-log'}, optional, default: None.
        The type of confidence intervals to estimate.
        If `None`, no confidence intervals are estimated.
        If "log-log", estimate confidence intervals using
        the log hazard or :math:`log(-log(S(t)))` as described in [2]_.

    Returns
    -------
    time : array, shape = (n_times,)
        Unique times.

    prob_survival : array, shape = (n_times,)
        Survival probability at each unique time point.
        If `time_enter` is provided, estimates are conditional probabilities.

    conf_int : array, shape = (2, n_times)
        Pointwise confidence interval of the Kaplan-Meier estimator
        at each unique time point.
        Only provided if `conf_type` is not None.

    Examples
    --------
    Creating a Kaplan-Meier curve:

    >>> x, y, conf_int = kaplan_meier_estimator(event, time, conf_type="log-log")
    >>> plt.step(x, y, where="post")
    >>> plt.fill_between(x, conf_int[0], conf_int[1], alpha=0.25, step="post")
    >>> plt.ylim(0, 1)
    >>> plt.show()

    See also
    --------
    sksurv.nonparametric.SurvivalFunctionEstimator
        Estimator API of the Kaplan-Meier estimator.

    References
    ----------
    .. [1] Kaplan, E. L. and Meier, P., "Nonparametric estimation from incomplete observations",
           Journal of The American Statistical Association, vol. 53, pp. 457-481, 1958.
    .. [2] Borgan Ø. and Liestøl K., "A Note on Confidence Intervals and Bands for the
           Survival Function Based on Transformations", Scandinavian Journal of
           Statistics. 1990;17(1):35–41.
    """
    event, time_enter, time_exit = check_y_survival(event, time_enter, time_exit, allow_all_censored=True)
    check_consistent_length(event, time_enter, time_exit)

    if conf_type is not None and reverse:
        raise NotImplementedError("Confidence intervals of the censoring distribution is not implemented.")

    if time_enter is None:
        uniq_times, n_events, n_at_risk, n_censored = _compute_counts(event, time_exit)

        if reverse:
            n_at_risk -= n_events
            n_events = n_censored
    else:
        if reverse:
            raise ValueError("The censoring distribution cannot be estimated from left truncated data")

        uniq_times, n_events, n_at_risk = _compute_counts_truncated(event, time_enter, time_exit)

    # account for 0/0 = nan
    ratio = np.divide(
        n_events,
        n_at_risk,
        out=np.zeros(uniq_times.shape[0], dtype=float),
        where=n_events != 0,
    )
    values = 1.0 - ratio

    if conf_type is not None:
        ratio_var = np.divide(
            n_events,
            n_at_risk * (n_at_risk - n_events),
            out=np.zeros(uniq_times.shape[0], dtype=float),
            where=(n_events != 0) & (n_at_risk != n_events),
        )

    if time_min is not None:
        mask = uniq_times >= time_min
        uniq_times = np.compress(mask, uniq_times)
        values = np.compress(mask, values)

    prob_survival = np.cumprod(values)

    if conf_type is None:
        return uniq_times, prob_survival

    if time_min is not None:
        ratio_var = np.compress(mask, ratio_var)

    ci = _km_ci_estimator(prob_survival, ratio_var, conf_level, conf_type)

    return uniq_times, prob_survival, ci


def nelson_aalen_estimator(event, time):
    """Nelson-Aalen estimator of cumulative hazard function.

    See [1]_, [2]_ for further description.

    Parameters
    ----------
    event : array-like, shape = (n_samples,)
        Contains binary event indicators.

    time : array-like, shape = (n_samples,)
        Contains event/censoring times.

    Returns
    -------
    time : array, shape = (n_times,)
        Unique times.

    cum_hazard : array, shape = (n_times,)
        Cumulative hazard at each unique time point.

    References
    ----------
    .. [1] Nelson, W., "Theory and applications of hazard plotting for censored failure data",
           Technometrics, vol. 14, pp. 945-965, 1972.

    .. [2] Aalen, O. O., "Nonparametric inference for a family of counting processes",
           Annals of Statistics, vol. 6, pp. 701–726, 1978.
    """
    event, time = check_y_survival(event, time)
    check_consistent_length(event, time)
    uniq_times, n_events, n_at_risk, _ = _compute_counts(event, time)

    y = np.cumsum(n_events / n_at_risk)

    return uniq_times, y


def ipc_weights(event, time):
    """Compute inverse probability of censoring weights

    Parameters
    ----------
    event : array, shape = (n_samples,)
        Boolean event indicator.

    time : array, shape = (n_samples,)
        Time when a subject experienced an event or was censored.

    Returns
    -------
    weights : array, shape = (n_samples,)
        inverse probability of censoring weights

    See also
    --------
    CensoringDistributionEstimator
        An estimator interface for estimating inverse probability
        of censoring weights for unseen time points.
    """
    if event.all():
        return np.ones(time.shape[0])

    unique_time, p = kaplan_meier_estimator(event, time, reverse=True)

    idx = np.searchsorted(unique_time, time[event])
    Ghat = p[idx]

    assert (Ghat > 0).all()

    weights = np.zeros(time.shape[0])
    weights[event] = 1.0 / Ghat

    return weights


class SurvivalFunctionEstimator(BaseEstimator):
    """Kaplan–Meier estimate of the survival function.

    Parameters
    ----------
    conf_level : float, optional, default: 0.95
        The level for a two-sided confidence interval on the survival curves.

    conf_type : None or {'log-log'}, optional, default: None.
        The type of confidence intervals to estimate.
        If `None`, no confidence intervals are estimated.
        If "log-log", estimate confidence intervals using
        the log hazard or :math:`log(-log(S(t)))`.

    See also
    --------
    sksurv.nonparametric.kaplan_meier_estimator
        Functional API of the Kaplan-Meier estimator.
    """

    _parameter_constraints = {
        "conf_level": [Interval(numbers.Real, 0.0, 1.0, closed="neither")],
        "conf_type": [None, StrOptions({"log-log"})],
    }

    def __init__(self, conf_level=0.95, conf_type=None):
        self.conf_level = conf_level
        self.conf_type = conf_type

    def fit(self, y):
        """Estimate survival distribution from training data.

        Parameters
        ----------
        y : structured array, shape = (n_samples,)
            A structured array containing the binary event indicator
            as first field, and time of event or time of censoring as
            second field.

        Returns
        -------
        self
        """
        self._validate_params()
        event, time = check_y_survival(y, allow_all_censored=True)

        values = kaplan_meier_estimator(event, time, conf_level=self.conf_level, conf_type=self.conf_type)
        if self.conf_type is None:
            unique_time, prob = values
        else:
            unique_time, prob, conf_int = values
            self.conf_int_ = np.column_stack((np.ones((2, 1)), conf_int))

        self.unique_time_ = np.r_[-np.inf, unique_time]
        self.prob_ = np.r_[1.0, prob]

        return self

    def predict_proba(self, time, return_conf_int=False):
        """Return probability of an event after given time point.

        :math:`\\hat{S}(t) = P(T > t)`

        Parameters
        ----------
        time : array, shape = (n_samples,)
            Time to estimate probability at.

        return_conf_int : bool, optional, default: False
            Whether to return the pointwise confidence interval
            of the survival function.
            Only available if :meth:`fit()` has been called
            with the `conf_type` parameter set.

        Returns
        -------
        prob : array, shape = (n_samples,)
            Probability of an event at the passed time points.

        conf_int : array, shape = (2, n_samples)
            Pointwise confidence interval at the passed time points.
            Only provided if `return_conf_int` is True.
        """
        check_is_fitted(self, "unique_time_")
        if return_conf_int and not hasattr(self, "conf_int_"):
            raise ValueError(
                "If return_conf_int is True, SurvivalFunctionEstimator must be fitted with conf_int != None"
            )

        time = check_array(time, ensure_2d=False, estimator=self, input_name="time")

        # K-M is undefined if estimate at last time point is non-zero
        extends = time > self.unique_time_[-1]
        if self.prob_[-1] > 0 and extends.any():
            raise ValueError(f"time must be smaller than largest observed time point: {self.unique_time_[-1]}")

        # beyond last time point is zero probability
        Shat = np.empty(time.shape, dtype=float)
        Shat[extends] = 0.0

        valid = ~extends
        time = time[valid]
        idx = np.searchsorted(self.unique_time_, time)
        # for non-exact matches, we need to shift the index to left
        eps = np.finfo(self.unique_time_.dtype).eps
        exact = np.absolute(self.unique_time_[idx] - time) < eps
        idx[~exact] -= 1
        Shat[valid] = self.prob_[idx]

        if not return_conf_int:
            return Shat

        ci = np.empty((2, time.shape[0]), dtype=float)
        ci[:, extends] = np.nan
        ci[:, valid] = self.conf_int_[:, idx]
        return Shat, ci


class CensoringDistributionEstimator(SurvivalFunctionEstimator):
    """Kaplan–Meier estimator for the censoring distribution."""

    def fit(self, y):
        """Estimate censoring distribution from training data.

        Parameters
        ----------
        y : structured array, shape = (n_samples,)
            A structured array containing the binary event indicator
            as first field, and time of event or time of censoring as
            second field.

        Returns
        -------
        self
        """
        event, time = check_y_survival(y)
        if event.all():
            self.unique_time_ = np.unique(time)
            self.prob_ = np.ones(self.unique_time_.shape[0])
        else:
            unique_time, prob = kaplan_meier_estimator(event, time, reverse=True)
            self.unique_time_ = np.r_[-np.inf, unique_time]
            self.prob_ = np.r_[1.0, prob]

        return self

    def predict_ipcw(self, y):
        """Return inverse probability of censoring weights at given time points.

        :math:`\\omega_i = \\delta_i / \\hat{G}(y_i)`

        Parameters
        ----------
        y : structured array, shape = (n_samples,)
            A structured array containing the binary event indicator
            as first field, and time of event or time of censoring as
            second field.

        Returns
        -------
        ipcw : array, shape = (n_samples,)
            Inverse probability of censoring weights.
        """
        event, time = check_y_survival(y)
        Ghat = self.predict_proba(time[event])

        if (Ghat == 0.0).any():
            raise ValueError("censoring survival function is zero at one or more time points")

        weights = np.zeros(time.shape[0])
        weights[event] = 1.0 / Ghat

        return weights


def _cum_inc_cr_ci_estimator(cum_inc, var, conf_level, conf_type):
    if conf_type not in {"log-log"}:
        raise ValueError(f"conf_type must be None or a str among {{'log-log'}}, but was {conf_type!r}")

    if not isinstance(conf_level, numbers.Real) or not np.isfinite(conf_level) or conf_level <= 0 or conf_level >= 1.0:
        raise ValueError(f"conf_level must be a float in the range (0.0, 1.0), but was {conf_level!r}")
    eps = np.finfo(var.dtype).eps
    sigma = np.zeros_like(var)
    np.divide(np.sqrt(var), cum_inc, where=var > eps, out=sigma)
    ci = _ci_logmlog(cum_inc, sigma, conf_level)
    # make first axis the competing risks, the second axis the lower and upper confidence interval
    ci = np.swapaxes(ci, 0, 1)
    return ci


def cumulative_incidence_competing_risks(
    event,
    time_exit,
    time_min=None,
    conf_level=0.95,
    conf_type=None,
    var_type="Aalen",
):
    """Non-parametric estimator of Cumulative Incidence function in the case of competing risks.

    See the :ref:`User Guide </user_guide/competing-risks.ipynb>` and [1]_ for further details.

    Parameters
    ----------
    event : array-like, shape = (n_samples,)
        Contains event indicators.

    time_exit : array-like, shape = (n_samples,)
        Contains event/censoring times. '0' indicates right-censoring.
        Positive integers (between 1 and n_risks, n_risks being the total number of different risks)
        indicate the possible different risks.
        It assumes there are events for all possible risks.

    time_min : float, optional, default: None
        Compute estimator conditional on survival at least up to
        the specified time.

    conf_level : float, optional, default: 0.95
        The level for a two-sided confidence interval on the cumulative incidence curves.

    conf_type : None or {'log-log'}, optional, default: None.
        The type of confidence intervals to estimate.
        If `None`, no confidence intervals are estimated.
        If "log-log", estimate confidence intervals using
        the log hazard or :math:`log(-log(S(t)))`.

    var_type : None or one of {'Aalen', 'Dinse', 'Dinse_Approx'}, optional, default: 'Aalen'
        The method for estimating the variance of the estimator.
        See [2]_, [3]_ and [4]_ for each of the methods.
        Only used if `conf_type` is not None.

    Returns
    -------
    time : array, shape = (n_times,)
        Unique times.

    cum_incidence : array, shape = (n_risks + 1, n_times)
        Cumulative incidence at each unique time point.
        The first dimension indicates total risk (``cum_incidence[0]``),
        the dimension `i=1,...,n_risks` the incidence for each competing risk.

    conf_int : array, shape = (n_risks + 1, 2, n_times)
        Pointwise confidence interval (second axis) of the Kaplan-Meier estimator
        at each unique time point (last axis)
        for all possible risks (first axis), including overall risk (``conf_int[0]``).
        Only provided if `conf_type` is not None.

    Examples
    --------
    Creating cumulative incidence curves:

    >>> from sksurv.datasets import load_bmt
    >>> dis, bmt_df = load_bmt()
    >>> event = bmt_df["status"]
    >>> time = bmt_df["ftime"]
    >>> n_risks = event.max()
    >>> x, y, conf_int = cumulative_incidence_competing_risks(event, time, conf_type="log-log")
    >>> plt.step(x, y[0], where="post", label="Total risk")
    >>> plt.fill_between(x, conf_int[0, 0], conf_int[0, 1], alpha=0.25, step="post")
    >>> for i in range(1, n_risks + 1):
    >>>    plt.step(x, y[i], where="post", label=f"{i}-risk")
    >>>    plt.fill_between(x, conf_int[i, 0], conf_int[i, 1], alpha=0.25, step="post")
    >>> plt.ylim(0, 1)
    >>> plt.legend()
    >>> plt.show()

    References
    ----------
    .. [1] Kalbfleisch, J.D. and Prentice, R.L. (2002)
           The Statistical Analysis of Failure Time Data. 2nd Edition, John Wiley and Sons, New York.
    .. [2] Aalen, O. (1978a). Annals of Statistics, 6, 534–545.
           We implement the formula in M. Pintilie: "Competing Risks: A Practical Perspective".
           John Wiley & Sons, 2006, Eq. 4.5
    .. [3] Dinse and Larson, Biometrika (1986), 379. Sect. 4, Eqs. 4 and 5.
    .. [4] Dinse and Larson, Biometrika (1986), 379. Sect. 4, Eq. 6.
    """
    event, time_exit = check_y_survival(event, time_exit, allow_all_censored=True, competing_risks=True)
    check_consistent_length(event, time_exit)

    n_risks = event.max()
    uniq_times, n_events_cr, n_at_risk, _n_censored = _compute_counts(event, time_exit)

    # account for 0/0 = nan
    n_t = uniq_times.shape[0]
    ratio = np.divide(
        n_events_cr,
        n_at_risk[..., np.newaxis],
        out=np.zeros((n_t, n_risks + 1), dtype=float),
        where=n_events_cr != 0,
    )

    if time_min is not None:
        mask = uniq_times >= time_min
        uniq_times = np.compress(mask, uniq_times)
        ratio = np.compress(mask, ratio, axis=0)

    kpe = np.cumprod(1.0 - ratio[:, 0])
    kpe_prime = np.r_[1.0, kpe[:-1]]
    cum_inc = np.empty((n_risks + 1, n_t), dtype=float)
    cum_inc[0] = 1.0 - kpe
    cum_inc[1:] = np.cumsum((ratio[:, 1:].T * kpe_prime), axis=1)

    if conf_type is None:
        return uniq_times, cum_inc

    if var_type == "Aalen":
        var = _var_aalen(n_events_cr, kpe_prime, n_at_risk, cum_inc)
    elif var_type == "Dinse_Approx":
        var = _var_dinse_approx(n_events_cr, kpe_prime, n_at_risk, cum_inc)
    elif var_type == "Dinse":
        var = _var_dinse(n_events_cr, kpe_prime, n_at_risk)
    else:
        raise ValueError(f"{var_type=} must be one of 'Aalen', 'Dinse', or 'Dinse_Approx'.")

    _x, _y, conf_int_km = kaplan_meier_estimator(event > 0, time_exit, conf_type="log-log")
    ci = np.empty(shape=(n_risks + 1, 2, n_t), dtype=conf_int_km.dtype)
    ci[0, :, :] = 1 - conf_int_km
    ci[1:, :, :] = _cum_inc_cr_ci_estimator(cum_inc[1:], var, conf_level, conf_type)

    return uniq_times, cum_inc, ci


def _var_dinse_approx(n_events_cr, kpe_prime, n_at_risk, cum_inc):
    """
    Variance estimator from Dinse and Larson, Biometrika (1986), 379
    See Section 4, Eqs. 6.
    This is an approximation from the _var_dinse, so that one should be preferred.
    However, this seems to be more common in the literature.
    """
    dr = n_events_cr[:, 0]
    dr_cr = n_events_cr[:, 1:].T
    irt = cum_inc[1:, :, np.newaxis] - cum_inc[1:, np.newaxis, :]
    mask = np.tril(np.ones_like(irt[0]))

    # var_a = np.sum(irt**2 * mask * (dr / (n_at_risk * (n_at_risk - dr))), axis=2)
    var_a = np.einsum("rjk,jk,k->rj", irt**2, mask, dr / (n_at_risk * (n_at_risk - dr)))
    var_b = np.cumsum(((n_at_risk - dr_cr) / n_at_risk) * (dr_cr / n_at_risk**2) * kpe_prime**2, axis=1)
    # var_c = -2 * np.sum(irt * mask * dr_cr[:, np.newaxis, :] * (kpe_prime / n_at_risk**2), axis=2)
    var_c = -2 * np.einsum("rjk,jk,rk,k->rj", irt, mask, dr_cr, kpe_prime / n_at_risk**2)

    var = var_a + var_b + var_c
    return var


def _var_dinse(n_events_cr, kpe_prime, n_at_risk):
    """
    Variance estimator from Dinse and Larson, Biometrika (1986), 379
    See Section 4, Eqs. 4 and 5
    """
    dr = n_events_cr[:, 0]
    dr_cr = n_events_cr[:, 1:].T
    theta = dr_cr * kpe_prime / n_at_risk
    x = dr / (n_at_risk * (n_at_risk - dr))
    cprod = np.cumprod(1 + x) / (1 + x)

    nt_range = np.arange(dr.size)
    i_idx = nt_range[:, None, None]
    j_idx = nt_range[None, :, None]
    k_idx = nt_range[None, None, :]
    mask = ((j_idx < i_idx) & (k_idx > j_idx) & (k_idx <= i_idx)).astype(int)

    _v1 = np.zeros_like(theta)
    np.divide((n_at_risk - dr_cr), n_at_risk * dr_cr, out=_v1, where=dr_cr > 0)
    v1 = np.cumsum(theta**2 * ((1 + _v1) * cprod - 1), axis=1)

    corr = (1 - 1 / n_at_risk) * cprod - 1
    v2 = 2 * np.einsum("rj,rk,ijk->ri", theta * corr, theta, mask)
    var = v1 + v2

    return var


def _var_aalen(n_events_cr, kpe_prime, n_at_risk, cum_inc):
    """
    Variance estimator from Aalen
    Aalen, O. (1978a). Nonparametric estimation of partial transition
    probabilities in multiple decrement models. Annals of Statistics, 6, 534–545.
    We implement it as shown in
    M. Pintilie: "Competing Risks: A Practical Perspective". John Wiley & Sons, 2006, Eq. 4.5
    This seems to be the estimator used in cmprsk, but there are some numerical differences with our implementation.
    """
    dr = n_events_cr[:, 0]
    dr_cr = n_events_cr[:, 1:].T
    irt = cum_inc[1:, :, np.newaxis] - cum_inc[1:, np.newaxis, :]
    mask = np.tril(np.ones_like(irt[0]))

    _va = np.zeros_like(kpe_prime)
    den_a = (n_at_risk - 1) * (n_at_risk - dr)
    np.divide(dr, den_a, out=_va, where=den_a > 0)
    # var_a = np.sum(irt**2 * mask * _va, axis=2)
    var_a = np.einsum("rjk,jk,k->rj", irt**2, mask, _va)

    _vb = np.zeros_like(kpe_prime)
    den_b = (n_at_risk - 1) * n_at_risk**2
    np.divide(1.0, den_b, out=_vb, where=den_b > 0)
    var_b = np.cumsum((n_at_risk - dr_cr) * dr_cr * _vb * kpe_prime**2, axis=1)

    _vca = dr_cr * (n_at_risk - dr_cr)
    _vcb = np.zeros_like(kpe_prime)
    den_c = n_at_risk * (n_at_risk - dr) * (n_at_risk - 1)
    np.divide(kpe_prime, den_c, out=_vcb, where=den_c > 0)
    # var_c = -2 * np.sum(irt * mask * _vca[:, np.newaxis, :] * _vcb, axis=2)
    var_c = -2 * np.einsum("rjk,jk,rk,k->rj", irt, mask, _vca, _vcb)

    var = var_a + var_b + var_c
    return var
