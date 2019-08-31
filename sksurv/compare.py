from collections import OrderedDict
import numpy
import pandas
from scipy import stats

from .util import check_arrays_survival

__all__ = ["compare_survival"]


def compare_survival(y, group_indicator, return_stats=False):
    """K-sample log-rank hypothesis test of identical survival functions.

    Compares the pooled hazard rate with each group-specific
    hazard rate. The alternative hypothesis is that the hazard
    rate of at least one group differs from the others at some time.

    See [1]_ for more details.

    Parameters
    ----------
    y : structured array, shape = (n_samples,)
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.

    group_indicator : array-like, shape = (n_samples,)
        Group membership of each sample.

    return_stats : bool, optional, default: False
        Whether to return a data frame with statistics for each group
        and the covariance matrix of the test statistic.

    Returns
    -------
    chisq : float
        Test statistic.
    pvalue : float
        Two-sided p-value with respect to the null hypothesis
        that the hazard rates across all groups are equal.
    stats : pandas.DataFrame
        Summary statistics for each group:  number of samples,
        observed number of events, expected number of events,
        and test statistic.
        Only provided if `return_stats` is True.
    covariance : array, shape=(n_groups, n_groups)
        Covariance matrix of the test statistic.
        Only provided if `return_stats` is True.

    References
    ----------
    .. [1] Fleming, T. R. and Harrington, D. P.
           A Class of Hypothesis Tests for One and Two Samples of Censored Survival Data.
           Communications In Statistics 10 (1981): 763-794.
    """

    group_indicator, event, time = check_arrays_survival(
        group_indicator, y, dtype="O", ensure_2d=False)

    n_samples = time.shape[0]
    groups, group_counts = numpy.unique(group_indicator, return_counts=True)
    n_groups = groups.shape[0]
    if n_groups == 1:
        raise ValueError("At least two groups must be specified, "
                         "but only one was provided.")

    # sort descending
    o = numpy.argsort(-time, kind="mergesort")
    x = group_indicator[o]
    event = event[o]
    time = time[o]

    at_risk = numpy.zeros(n_groups, dtype=numpy.int_)
    observed = numpy.zeros(n_groups, dtype=numpy.int_)
    expected = numpy.zeros(n_groups, dtype=numpy.float_)
    covar = numpy.zeros((n_groups, n_groups), dtype=numpy.float_)
    k = 0
    while k < n_samples:
        ti = time[k]
        total_events = 0
        while k < n_samples and ti == time[k]:
            idx = numpy.searchsorted(groups, x[k])
            if event[k]:
                observed[idx] += 1
                total_events += 1
            at_risk[idx] += 1
            k += 1

        if total_events != 0:
            total_at_risk = k
            expected += at_risk * (total_events / total_at_risk)
            if total_at_risk > 1:
                multiplier = total_events * (total_at_risk - total_events) / (total_at_risk * (total_at_risk - 1))
                for g1 in range(n_groups):
                    temp = at_risk[g1] * multiplier
                    covar[g1, g1] += temp
                    for g2 in range(n_groups):
                        covar[g1, g2] -= temp * at_risk[g2] / total_at_risk

    df = n_groups - 1
    zz = observed[:df] - expected[:df]
    chisq = numpy.linalg.solve(covar[:df, :df], zz).dot(zz)
    pval = stats.chi2.sf(chisq, df)

    if return_stats:
        table = OrderedDict()
        table["counts"] = group_counts
        table["observed"] = observed
        table["expected"] = expected
        table["statistic"] = observed - expected
        table = pandas.DataFrame.from_dict(table)
        table.index = pandas.Index(groups, name="group")
        return chisq, pval, table, covar

    return chisq, pval
