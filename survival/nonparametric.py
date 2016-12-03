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
from sklearn.utils.validation import check_consistent_length

from .util import check_y_survival

__all__ = ['kaplan_meier_estimator', 'nelson_aalen_estimator', 'ipc_weights']


def _compute_counts(event, time):
    """Count right censored and uncensored samples at each unique time point.

    Parameters
    ----------
    event : array
        Boolean event indicator.

    time : array
        Survival time or time of censoring.

    Returns
    -------
    times : array
        Unique time points.

    n_events : array
        Number of events at each time point.

    n_at_risk : array
        Number of samples that are censored or have an event at each time point.
    """
    n_samples = event.shape[0]

    order = numpy.argsort(time, kind="mergesort")

    uniq_times = numpy.empty(n_samples, dtype=time.dtype)
    uniq_events = numpy.empty(n_samples, dtype=numpy.int_)
    uniq_counts = numpy.empty(n_samples, dtype=numpy.int_)

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

    times = numpy.resize(uniq_times, j)
    n_events = numpy.resize(uniq_events, j)
    total_count = numpy.resize(uniq_counts, j)

    # offset cumulative sum by one
    total_count = numpy.concatenate(([0], total_count))
    n_at_risk = n_samples - numpy.cumsum(total_count)

    return times, n_events, n_at_risk[:-1]


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

    uniq_times = numpy.sort(numpy.unique(numpy.concatenate((time_enter, time_exit))), kind="mergesort")
    total_counts = numpy.empty(len(uniq_times), dtype=numpy.int_)
    event_counts = numpy.empty(len(uniq_times), dtype=numpy.int_)

    order_enter = numpy.argsort(time_enter, kind="mergesort")
    order_exit = numpy.argsort(time_exit, kind="mergesort")
    s_time_enter = time_enter[order_enter]
    s_time_exit = time_exit[order_exit]

    t0 = uniq_times[0]
    # everything larger is included
    idx_enter = numpy.searchsorted(s_time_enter, t0, side="right")
    # everything smaller is excluded
    idx_exit = numpy.searchsorted(s_time_exit, t0, side="left")

    total_counts[0] = idx_enter
    # except people die on the day they enter
    event_counts[0] = 0

    for i in range(1, len(uniq_times)):
        ti = uniq_times[i]

        while idx_enter < n_samples and s_time_enter[idx_enter] <= ti:
            idx_enter += 1

        while idx_exit < n_samples and s_time_exit[idx_exit] < ti:
            idx_exit += 1

        risk_set = numpy.setdiff1d(order_enter[:idx_enter], order_exit[:idx_exit], assume_unique=True)
        total_counts[i] = len(risk_set)

        count_event = 0
        k = idx_exit
        while k < n_samples and s_time_exit[k] == ti:
            if event[order_exit[k]]:
                count_event += 1
            k += 1
        event_counts[i] = count_event

    return uniq_times, event_counts, total_counts


def kaplan_meier_estimator(event, time_exit, time_enter=None, time_min=None):
    """Kaplan-Meier estimator of survival function.

    Parameters
    ----------
    event : array-like, shape = [n_samples,]
        Contains binary event indicators.

    time_exit : array-like, shape = [n_samples,]
        Contains event/censoring times.

    time_enter : array-like, shape = [n_samples,], optional
        Contains time when each individual entered the study for
        left truncated survival data.

    time_min : float, optional
        Compute estimator conditional on survival at least up to
        the specified time.

    Returns
    -------
    time : array, shape = [n_times]
        Unique times.

    prob_survival : array, shape = [n_times]
        Survival probability at each unique time point.
        If `time_enter` is provided, estimates are conditional probabilities.

    Example
    -------
    Creating a Kaplan-Meier curve:

    >>> x, y = kaplan_meier_estimator(event, time)
    >>> plt.step(x, y, where="post")
    >>> plt.ylim(0, 1)
    >>> plt.show()

    References
    ----------
    .. [1]: Kaplan, E. L. and Meier, P., "Nonparametric estimation from incomplete observations",
            Journal of The American Statistical Association, vol. 53, pp. 457-481, 1958.
    """
    event, time_enter, time_exit = check_y_survival(event, time_enter, time_exit)
    check_consistent_length(event, time_enter, time_exit)

    if time_enter is None:
        uniq_times, n_events, n_at_risk = _compute_counts(event, time_exit)
    else:
        uniq_times, n_events, n_at_risk = _compute_counts_truncated(event, time_enter, time_exit)

    values = 1 - n_events / n_at_risk

    if time_min is not None:
        mask = uniq_times >= time_min
        uniq_times = numpy.compress(mask, uniq_times)
        values = numpy.compress(mask, values)

    y = numpy.cumprod(values)
    return uniq_times, y


def nelson_aalen_estimator(event, time):
    """Nelson-Aalen estimator of cumulative hazard function.

    Parameters
    ----------
    y : structured array, shape = [n_samples]
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.

    Returns
    -------
    time : array, shape = [n_times]
        Unique times.

    cum_hazard : array, shape = [n_times]
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
    uniq_times, n_events, n_at_risk = _compute_counts(event, time)

    y = numpy.cumsum(n_events / n_at_risk)

    return uniq_times, y


def ipc_weights(event, time):
    """Compute inverse probability of censoring weights

    Parameters
    ----------
    event : array, shape = [n_samples]
        Boolean event indicator.

    time_start : array, shape = [n_samples]
        Time when a subject experienced an event or was censored.

    Returns
    -------
    weights : array, shape = [n_samples]
        inverse probability of censoring weights
    """
    if event.all():
        return numpy.ones(time.shape[0])

    unique_time, p = kaplan_meier_estimator(-event, time)

    idx = numpy.searchsorted(unique_time, time[event])
    Ghat = p[idx]

    assert (Ghat > 0).all()

    weights = numpy.zeros(time.shape[0])
    weights[event] = 1.0 / Ghat

    return weights
