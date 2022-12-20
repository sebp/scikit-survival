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
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_consistent_length, check_is_fitted

from .util import check_y_survival

__all__ = [
    "CensoringDistributionEstimator",
    "kaplan_meier_estimator",
    "nelson_aalen_estimator",
    "ipc_weights",
    "SurvivalFunctionEstimator",
]


def _compute_counts(event, time, order=None):
    """Count right censored and uncensored samples at each unique time point.

    Parameters
    ----------
    event : array
        Boolean event indicator.

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

    n_at_risk : array
        Number of samples that have not been censored or have not had an event at each time point.

    n_censored : array
        Number of censored samples at each time point.
    """
    n_samples = event.shape[0]

    if order is None:
        order = np.argsort(time, kind="mergesort")

    uniq_times = np.empty(n_samples, dtype=time.dtype)
    uniq_events = np.empty(n_samples, dtype=int)
    uniq_counts = np.empty(n_samples, dtype=int)

    i = 0
    prev_val = time[order[0]]
    j = 0
    while True:
        count_event = 0
        count = 0
        while i < n_samples and prev_val == time[order[i]]:
            if event[order[i]]:
                count_event += 1

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
    n_events = np.resize(uniq_events, j)
    total_count = np.resize(uniq_counts, j)
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

        while idx_enter < n_samples and s_time_enter[idx_enter] <= ti:
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


def kaplan_meier_estimator(event, time_exit, time_enter=None, time_min=None, reverse=False, with_variance=False):
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

    with_variance : bool, optional, default: False
        Whether to estimate the variance of the Kaplan-Meier estimator.
        Only available if reverse=False.

    Returns
    -------
    time : array, shape = (n_times,)
        Unique times.

    prob_survival : array, shape = (n_times,)
        Survival probability at each unique time point.
        If `time_enter` is provided, estimates are conditional probabilities.

    var_prob_survival : array, shape = (n_times,)
        Variance of the KM estimator at each unique time point, based on Greenwood's formula [2]_.

    Examples
    --------
    Creating a Kaplan-Meier curve:

    >>> x, y = kaplan_meier_estimator(event, time)
    >>> plt.step(x, y, where="post")
    >>> plt.ylim(0, 1)
    >>> plt.show()

    References
    ----------
    .. [1] Kaplan, E. L. and Meier, P., "Nonparametric estimation from incomplete observations",
           Journal of The American Statistical Association, vol. 53, pp. 457-481, 1958.
    .. [2] Greenwood, Major (1926). A report on the natural duration of cancer.
           Issue 33 of Reports on public health and medical subjects. HMSO. OCLC 14713088
    """
    event, time_enter, time_exit = check_y_survival(event, time_enter, time_exit, allow_all_censored=True)

    check_consistent_length(event, time_enter, time_exit)

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
        n_events, n_at_risk,
        out=np.zeros(uniq_times.shape[0], dtype=float),
        where=n_events != 0,
    )
    values = 1.0 - ratio

    if with_variance:
        if reverse:
            raise ValueError("The variance of the censoring distribution is not implemented.")
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

    if with_variance:
        if time_min is not None:
            ratio_var = np.compress(mask, ratio_var)
        var_prob_survival = prob_survival**2 * np.cumsum(ratio_var)
        return uniq_times, prob_survival, var_prob_survival
    else:
        return uniq_times, prob_survival


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
    """Kaplan–Meier estimate of the survival function."""

    def __init__(self):
        pass

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
        event, time = check_y_survival(y, allow_all_censored=True)

        unique_time, prob, var_est = kaplan_meier_estimator(event, time, with_variance=True)
        self.unique_time_ = np.r_[-np.infty, unique_time]
        self.prob_ = np.r_[1., prob]
        self.var_est_ = np.r_[0.0, var_est]

        return self

    def predict_proba(self, time, with_variance=False):
        """Return probability of an event after given time point.

        :math:`\\hat{S}(t) = P(T > t)`

        Parameters
        ----------
        time : array, shape = (n_samples,)
            Time to estimate probability at.

        with_variance : bool, optional, default: False
            Whether to return the variance of the Kaplan-Meier estimator.

        Returns
        -------
        prob : array, shape = (n_samples,)
            Probability of an event.

        var : array, shape = (n_samples,)
            Variance of the Kaplan-Meier estimator
        """
        check_is_fitted(self, "unique_time_")
        time = check_array(time, ensure_2d=False, estimator=self, input_name="time")

        # K-M is undefined if estimate at last time point is non-zero
        extends = time > self.unique_time_[-1]
        if self.prob_[-1] > 0 and extends.any():
            raise ValueError(
                "time must be smaller than largest "
                "observed time point: {}".format(self.unique_time_[-1]))

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

        if with_variance:
            var = np.empty(Shat.shape, dtype=float)
            var[extends] = 0.0
            var[valid] = self.var_est_[idx]
            return Shat, var
        else:
            return Shat


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
            self.unique_time_ = np.r_[-np.infty, unique_time]
            self.prob_ = np.r_[1., prob]

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
