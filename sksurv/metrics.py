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
from sklearn.utils import check_consistent_length, check_array

import numpy

__all__ = ['concordance_index_censored']


def concordance_index_censored(event_indicator, event_time, estimate):
    """Concordance index for right-censored data

    The concordance index is defined as the proportion of all comparable pairs
    in which the predictions and outcomes are concordant.

    Samples are comparable if for at least one of them an event occurred.
    If the estimated risk is larger for the sample with a higher time of
    event/censoring, the predictions of that pair are said to be concordant.
    If an event occurred for one sample and the other is known to be
    event-free at least until the time of event of the first, the second
    sample is assumed to *outlive* the first.
    When predicted risks are identical for a pair, 0.5 rather than 1 is added
    to the count of concordant pairs.
    A pair is not comparable if an event occurred for both of them at the same
    time or an event occurred for one of them but the time of censoring is
    smaller than the time of event of the first one.

    Parameters
    ----------
    event_indicator : array-like, shape = [n_samples,]
        Boolean array denotes whether an event occurred

    event_time : array-like, shape = [n_samples,]
        Array containing the time of an event or time of censoring

    estimate : array-like, shape = [n_samples,]
        Estimated risk of experiencing an event

    Returns
    -------
    cindex : float
        Concordance index

    concordant : int
        Number of concordant pairs

    discordant : int
        Number of discordant pairs

    tied_risk : int
        Number of pairs having tied estimated risks

    tied_time : int
        Number of pairs having an event at the same time

    References
    ----------
    .. [1] Harrell, F.E., Califf, R.M., Pryor, D.B., Lee, K.L., Rosati, R.A,
           "Multivariable prognostic models: issues in developing models,
           evaluating assumptions and adequacy, and measuring and reducing errors",
           Statistics in Medicine, 15(4), 361-87, 1996.
    """
    check_consistent_length(event_indicator, event_time, estimate)
    event_indicator = check_array(event_indicator, ensure_2d=False)
    event_time = check_array(event_time, ensure_2d=False)
    estimate = check_array(estimate, ensure_2d=False)

    if not numpy.issubdtype(event_indicator.dtype, numpy.bool_):
        raise ValueError(
            'only boolean arrays are supported as class labels for survival analysis, got {0}'.format(
                event_indicator.dtype))

    n_samples = len(event_time)
    if n_samples < 2:
        raise ValueError("Need a minimum of two samples")

    if not event_indicator.any():
        raise ValueError("All samples are censored")

    order = numpy.argsort(event_time)

    tied_time = 0
    comparable = {}
    for i in range(n_samples - 1):
        inext = i + 1
        j = inext
        time_i = event_time[order[i]]
        while j < n_samples and event_time[order[j]] == time_i:
            j += 1

        if event_indicator[order[i]]:
            mask = numpy.zeros(n_samples, dtype=bool)
            mask[inext:] = True
            if j - i > 1:
                # event times are tied, need to check for coinciding events
                event_at_same_time = event_indicator[order[inext:j]]
                mask[inext:j] = numpy.logical_not(event_at_same_time)
                tied_time += event_at_same_time.sum()
            comparable[i] = mask
        elif j - i > 1:
            # events at same time are comparable if at least one of them is positive
            mask = numpy.zeros(n_samples, dtype=bool)
            mask[inext:j] = event_indicator[order[inext:j]]
            comparable[i] = mask

    concordant = 0
    discordant = 0
    tied_risk = 0
    for ind, mask in comparable.items():
        est_i = estimate[order[ind]]
        event_i = event_indicator[order[ind]]

        est = estimate[order[mask]]

        if event_i:
            # an event should have a higher score
            con = (est < est_i).sum()
        else:
            # a non-event should have a lower score
            con = (est > est_i).sum()
        concordant += con

        tie = (est == est_i).sum()
        tied_risk += tie

        discordant += est.size - con - tie

    cindex = (concordant + 0.5 * tied_risk) / (concordant + discordant + tied_risk)
    return cindex, concordant, discordant, tied_risk, tied_time
