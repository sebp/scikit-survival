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
import pandas as pd
from scipy.integrate import trapz
from sklearn.utils import check_consistent_length, check_array

from .nonparametric import CensoringDistributionEstimator, SurvivalFunctionEstimator
from .util import check_y_survival

__all__ = [
    'concordance_index_censored',
    'concordance_index_ipcw',
    'cumulative_dynamic_auc',
    'brier_score',
    'integrated_brier_score',
    'calibration_curve',
]


def _check_estimate(estimate, test_time):
    estimate = check_array(estimate, ensure_2d=False)
    if estimate.ndim != 1:
        raise ValueError(
            'Expected 1D array, got {:d}D array instead:\narray={}.\n'.format(
                estimate.ndim, estimate))
    check_consistent_length(test_time, estimate)
    return estimate


def _check_inputs(event_indicator, event_time, estimate):
    check_consistent_length(event_indicator, event_time, estimate)
    event_indicator = check_array(event_indicator, ensure_2d=False)
    event_time = check_array(event_time, ensure_2d=False)
    estimate = _check_estimate(estimate, event_time)

    if not numpy.issubdtype(event_indicator.dtype, numpy.bool_):
        raise ValueError(
            'only boolean arrays are supported as class labels for survival analysis, got {0}'.format(
                event_indicator.dtype))

    if len(event_time) < 2:
        raise ValueError("Need a minimum of two samples")

    if not event_indicator.any():
        raise ValueError("All samples are censored")

    return event_indicator, event_time, estimate


def _get_comparable(event_indicator, event_time, order):
    n_samples = len(event_time)
    tied_time = 0
    comparable = {}
    i = 0
    while i < n_samples - 1:
        time_i = event_time[order[i]]
        start = i + 1
        end = start
        while end < n_samples and event_time[order[end]] == time_i:
            end += 1

        # check for tied event times
        event_at_same_time = event_indicator[order[i:end]]
        censored_at_same_time = ~event_at_same_time
        for j in range(i, end):
            if event_indicator[order[j]]:
                mask = numpy.zeros(n_samples, dtype=bool)
                mask[end:] = True
                # an event is comparable to censored samples at same time point
                mask[i:end] = censored_at_same_time
                comparable[j] = mask
                tied_time += censored_at_same_time.sum()
        i = end

    return comparable, tied_time


def _estimate_concordance_index(event_indicator, event_time, estimate, weights, tied_tol=1e-8):
    order = numpy.argsort(event_time)

    comparable, tied_time = _get_comparable(event_indicator, event_time, order)

    concordant = 0
    discordant = 0
    tied_risk = 0
    numerator = 0.0
    denominator = 0.0
    for ind, mask in comparable.items():
        est_i = estimate[order[ind]]
        event_i = event_indicator[order[ind]]
        w_i = weights[order[ind]]

        est = estimate[order[mask]]

        assert event_i, 'got censored sample at index %d, but expected uncensored' % order[ind]

        ties = numpy.absolute(est - est_i) <= tied_tol
        n_ties = ties.sum()
        # an event should have a higher score
        con = est < est_i
        n_con = con[~ties].sum()

        numerator += w_i * n_con + 0.5 * w_i * n_ties
        denominator += w_i * mask.sum()

        tied_risk += n_ties
        concordant += n_con
        discordant += est.size - n_con - n_ties

    cindex = numerator / denominator
    return cindex, concordant, discordant, tied_risk, tied_time

def _calib_plot(fu_time, n_bins, pred_surv, time, dead, color, label, error_bars=0,alpha=1., markersize=1., markertype='o'):
    """
    Kaplan Meier Estimates for `n_bins` of predicted survival `pred_surv` at `fu_time` 
    plotted against mean `pred_surv` in the bin
    Typically deciles are used, despite arbitrary choice.
    Deprecated --> better use loess based/nearest neighbor estimated plot
    """
        # TODO: exchange lifelines KaplanMeierFitter with sksurv.nonparametric.SurvivalFunctionEstimator. Confidence interval estimation needed in advance.
    from lifelines import KaplanMeierFitter
    import matplotlib.pyplot as plt
#    cuts = numpy.concatenate((numpy.array([-1e6]),numpy.percentile(pred_surv, numpy.arange(100/n_bins,100,100/n_bins)),numpy.array([1e6])))
#    bin = pd.cut(pred_surv,cuts,labels=False)
    bin = pd.qcut(pred_surv,n_bins,labels=None)
    kmf = KaplanMeierFitter()
    est = []
    ci_upper = []
    ci_lower = []
    mean_pred_surv = []
    for which_bin in range(max(bin)+1):
        kmf.fit(time[bin==which_bin], event_observed=dead[bin==which_bin])
        est.append(numpy.interp(fu_time, kmf.survival_function_.index.values, kmf.survival_function_.KM_estimate))
        ci_upper.append(numpy.interp(fu_time, kmf.survival_function_.index.values, kmf.confidence_interval_.loc[:,'KM_estimate_upper_0.95']))
        ci_lower.append(numpy.interp(fu_time, kmf.survival_function_.index.values, kmf.confidence_interval_.loc[:,'KM_estimate_lower_0.95']))
        mean_pred_surv.append(numpy.mean(pred_surv[bin==which_bin]))
    est = numpy.array(est)
    ci_upper = numpy.array(ci_upper)
    ci_lower = numpy.array(ci_lower)
    if error_bars:
        plt.errorbar(mean_pred_surv, est, yerr = numpy.transpose(numpy.column_stack((est-ci_lower,ci_upper-est))), fmt='o',
                 c=color,label=label)
    else:
        plt.plot(mean_pred_surv, est, markertype, c=color,label=label, alpha=alpha, markersize=markersize)
    return (numpy.array(mean_pred_surv), est, ci_upper,ci_lower)


def _calib_plot_loess(fu_time,
                      pred_surv, time, dead,
                      time_kfold,dead_kfold,
                      color, label,
                      ci=False, ci_alpha=.05, trunc =(1,99),
                      alpha=1., markersize=1., markertype='-',
                      loess=True):
    """
    plot calibration curve based on pseudovalues: est_i(t) = n * (prodlim(t)) - (n-1) * prodlim_i(t)
    see r package prodlim https://rdrr.io/cran/prodlim/man/jackknife.html
    see also    Gerds et al.,Calibration plots for risk prediction models in the presence of competing risks., https://doi.org/10.1002/sim.6152
                Graw et al., https://doi.org/10.1007/s10985-008-9107-z
    
    We combined the pseudovalues approach with skmisc.loess.loess smoothing (https://dx.doi.org/10.1002%2Fsim.5941)
    span = 0.75 (default) works good according to https://dx.doi.org/10.1002%2Fsim.5941
    
    Parameters:
    -----------
    * fu_time : int / float
        time for which cumulative calibration should be calculated
        
    * pred_surv : array, shape = (n_samples,)
        array with predicted survival probability at time t
        
    * time: array, shape = (n_samples,)
        array with time of event or censoring time
        
    * dead: array, shape = (n_samples,)
        event indicator array, True = event, False = no event
        
    * time_kfold: array, shape = (n_samples,)
        in case of cross validation 1 full data-set for KM estimation is necessary
        in case of internal validation, training data can also be used for KM Estimation
        array with time of event or censoring time
        
    * dead_kfold: array, shape = (n_samples,)
            in case of cross validation 1 full data-set for KM estimation is necessary    
            in case of internal validation, training data can also be used for KM Estimation
            event indicator array, True = event, False = no event
    * color: str
        color for the plot
        
    * label: str
        label of the plot
        
    * ci: bool
        controls, whether confidence intervals should be calculated
        
    * ci_alpha: float
        confidence niveau, default 0.05
        
    * trunc: tuple, type int / float
        truncate pred_surv at `trunc[0/1]`'th percentiles.
        
    * alpha: float
        alpha transluicency for plot
        
    * markersize: float
        markersize for pyplot
        
    * markertype: str
        type of marker used by matplotlib.pyplot, default '-' = line
        
    * loess: bool
        use loess smoother, if false (not recommended, as not tested) use k nearest neighbors.
        
    Returns:
    --------
    
    * s_pred_surv: array, shape = (n_samples*)
        random subsample (only if n_samples >30000) of predicted survival probability array at fu_time
    * actual: array, shape = (n_samples*)
        loess smoothed actual survival probability for each reduced sample
    * ci_lower:array, array, shape = (n_samples*)
        lower confidence limit if ci= True, otherwise ci_lower equals actual
    * ci_upper:array, array, shape = (n_samples*)
        upper confidence limit if ci= True, otherwise ci_lower equals actual
                    
    Example:
    --------

    """
    from sklearn.neighbors import KNeighborsRegressor
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from skmisc.loess import loess
    
    est=numpy.zeros(pred_surv.shape)
    
    # get jacknife pseudovals for whole population if internal validation
    # get jacknife pseudovals for one complete sample in case of repeated CV.
    est_kfold=_jacknife(fu_time,time_kfold,dead_kfold) 
    
    # fill pseudovals for testing population/ whole cv population
    for t,e in zip(time_kfold,dead_kfold):
        cond= (time==t) & (dead == e)
        cond_kfold = (time_kfold==t) & (dead_kfold == e)
        est[cond]=est_kfold[cond_kfold].mean()
        
    # hacky protection against memory overload, if samples are large
    pred_surv_split,_,est_split,_=train_test_split(pred_surv,est,train_size=30000,shuffle=True)
    order=numpy.argsort(pred_surv_split)
    pred_surv_split=pred_surv_split[order]
    est_split=est_split[order]
    
    if loess:
        loessfitter=loess(pred_surv_split,est_split)
        loessfitter.fit()
    else:
        n_neighbors=4
        loessfitter=KNeighborsRegressor(n_neighbors, weights='minkowski')
        loessfitter.fit(pred_surv,est)

    # truncate predicted survival at trunc, (e.g. 1 / 99 th percentile)
    s_pred_surv=pred_surv_split # to lazy for renaming the rest :-)
    s_pred_surv=s_pred_surv[numpy.logical_and(s_pred_surv>=numpy.percentile(s_pred_surv,trunc[0]),s_pred_surv<=numpy.percentile(s_pred_surv,trunc[1]))]

    if ci:
        prediction=loessfitter.predict(s_pred_surv,stderror=True) # max 30000 to avoid out of memory error
        confint=prediction.confidence(alpha=ci_alpha)
        actual = prediction.values
        ci_upper = confint.upper
        ci_lower = confint.lower
        plt.plot(s_pred_surv,actual,markertype, c=color,label=label, alpha=alpha, markersize=markersize)
        plt.fill_between(s_pred_surv,ci_lower,ci_upper, edgecolor=color,facecolor=color, alpha=0.5*alpha,antialiased=True)
    else:
        prediction=loessfitter.predict(s_pred_surv)
        actual = prediction.values
        ci_upper=actual
        ci_lower=actual
        plt.plot(s_pred_surv,actual,markertype, c=color,label=label, alpha=alpha, markersize=markersize)
    return (s_pred_surv, numpy.array(actual), numpy.array(ci_upper), numpy.array(ci_lower))

def _jacknife(fu_time,time,dead,verbose=True):
    """
    calc pseudovalues according to TA Gerds et al. 
    https://rdrr.io/cran/prodlim/man/jackknife.html
    """
    # TODO: exchange lifelines KaplanMeierFitter with sksurv.nonparametric kaplan meier estimator. 
    from sklearn.model_selection import LeaveOneOut
    from lifelines import KaplanMeierFitter
    import sys
    #generate pseudovalues
    llo=LeaveOneOut()
    kmf = KaplanMeierFitter()
    kmf.fit(time,event_observed=dead)
    kme=numpy.interp(fu_time, kmf.survival_function_.index.values, kmf.survival_function_.KM_estimate) # linear interpolation
    kmfi = KaplanMeierFitter()
    pseudovals=[]
    k=0
    for train_index, test_index in llo.split(time):
        kmfi.fit(time[train_index], event_observed=dead[train_index])
        kmei=numpy.interp(fu_time, kmfi.survival_function_.index.values, kmfi.survival_function_.KM_estimate) # linear interpolation
        pseudovali=(time.shape[0] * kme) - ((time.shape[0] - 1) * kmei)
        pseudovals.append(pseudovali)
        k+=1
        sys.stdout.write("\rStep: %i/%i.  " % (k,time.shape[0]))
        sys.stdout.flush()

    return numpy.array(pseudovals)


def concordance_index_censored(event_indicator, event_time, estimate, tied_tol=1e-8):
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
    event_indicator : array-like, shape = (n_samples,)
        Boolean array denotes whether an event occurred

    event_time : array-like, shape = (n_samples,)
        Array containing the time of an event or time of censoring

    estimate : array-like, shape = (n_samples,)
        Estimated risk of experiencing an event

    tied_tol : float, optional, default: 1e-8
        The tolerance value for considering ties.
        If the absolute difference between risk scores is smaller
        or equal than `tied_tol`, risk scores are considered tied.

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
        Number of comparable pairs sharing the same time

    References
    ----------
    .. [1] Harrell, F.E., Califf, R.M., Pryor, D.B., Lee, K.L., Rosati, R.A,
           "Multivariable prognostic models: issues in developing models,
           evaluating assumptions and adequacy, and measuring and reducing errors",
           Statistics in Medicine, 15(4), 361-87, 1996.
    """
    event_indicator, event_time, estimate = _check_inputs(
        event_indicator, event_time, estimate)

    w = numpy.ones_like(estimate)

    return _estimate_concordance_index(event_indicator, event_time, estimate, w, tied_tol)


def concordance_index_ipcw(survival_train, survival_test, estimate, tau=None, tied_tol=1e-8):
    """Concordance index for right-censored data based on inverse probability of censoring weights.

    This is an alternative to the estimator in :func:`concordance_index_censored`
    that does not depend on the distribution of censoring times in the test data.
    Therefore, the estimate is unbiased and consistent for a population concordance
    measure that is free of censoring.

    It is based on inverse probability of censoring weights, thus requires
    access to survival times from the training data to estimate the censoring
    distribution. Note that this requires that survival times `survival_test`
    lie within the range of survival times `survival_train`. This can be
    achieved by specifying the truncation time `tau`.
    The resulting `cindex` tells how well the given prediction model works in
    predicting events that occur in the time range from 0 to `tau`.

    The estimator uses the Kaplan-Meier estimator to estimate the
    censoring survivor function. Therefore, it is restricted to
    situations where the random censoring assumption holds and
    censoring is independent of the features.

    Parameters
    ----------
    survival_train : structured array, shape = (n_train_samples,)
        Survival times for training data to estimate the censoring
        distribution from.
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.

    survival_test : structured array, shape = (n_samples,)
        Survival times of test data.
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.

    estimate : array-like, shape = (n_samples,)
        Estimated risk of experiencing an event of test data.

    tau : float, optional
        Truncation time. The survival function for the underlying
        censoring time distribution :math:`D` needs to be positive
        at `tau`, i.e., `tau` should be chosen such that the
        probability of being censored after time `tau` is non-zero:
        :math:`P(D > \\tau) > 0`. If `None`, no truncation is performed.

    tied_tol : float, optional, default: 1e-8
        The tolerance value for considering ties.
        If the absolute difference between risk scores is smaller
        or equal than `tied_tol`, risk scores are considered tied.

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
        Number of comparable pairs sharing the same time

    References
    ----------
    .. [1] Uno, H., Cai, T., Pencina, M. J., D’Agostino, R. B., & Wei, L. J. (2011).
           "On the C-statistics for evaluating overall adequacy of risk prediction
           procedures with censored survival data".
           Statistics in Medicine, 30(10), 1105–1117.
    """
    test_event, test_time = check_y_survival(survival_test)

    if tau is not None:
        mask = test_time < tau
        survival_test = survival_test[mask]

    estimate = _check_estimate(estimate, test_time)

    cens = CensoringDistributionEstimator()
    cens.fit(survival_train)
    ipcw_test = cens.predict_ipcw(survival_test)
    if tau is None:
        ipcw = ipcw_test
    else:
        ipcw = numpy.empty(estimate.shape[0], dtype=ipcw_test.dtype)
        ipcw[mask] = ipcw_test
        ipcw[~mask] = 0

    w = numpy.square(ipcw)

    return _estimate_concordance_index(test_event, test_time, estimate, w, tied_tol)


def cumulative_dynamic_auc(survival_train, survival_test, estimate, times, tied_tol=1e-8):
    """Estimator of cumulative/dynamic AUC for right-censored time-to-event data.

    The receiver operating characteristic (ROC) curve and the area under the
    ROC curve (AUC) can be extended to survival data by defining
    sensitivity (true positive rate) and specificity (true negative rate)
    as time-dependent measures. *Cumulative cases* are all individuals that
    experienced an event prior to or at time :math:`t` (:math:`t_i \\leq t`),
    whereas *dynamic controls* are those with :math:`t_i > t`.
    The associated cumulative/dynamic AUC quantifies how well a model can
    distinguish subjects who fail by a given time (:math:`t_i \\leq t`) from
    subjects who fail after this time (:math:`t_i > t`).

    Given an estimator of the :math:`i`-th individual's risk score
    :math:`\\hat{f}(\\mathbf{x}_i)`, the cumulative/dynamic AUC at time
    :math:`t` is defined as

    .. math::

        \\widehat{\\mathrm{AUC}}(t) =
        \\frac{\\sum_{i=1}^n \\sum_{j=1}^n I(y_j > t) I(y_i \\leq t) \\omega_i
        I(\\hat{f}(\\mathbf{x}_j) \\leq \\hat{f}(\\mathbf{x}_i))}
        {(\\sum_{i=1}^n I(y_i > t)) (\\sum_{i=1}^n I(y_i \\leq t) \\omega_i)}

    where :math:`\\omega_i` are inverse probability of censoring weights (IPCW).

    To estimate IPCW, access to survival times from the training data is required
    to estimate the censoring distribution. Note that this requires that survival
    times `survival_test` lie within the range of survival times `survival_train`.
    This can be achieved by specifying `times` accordingly, e.g. by setting
    `times[-1]` slightly below the maximum expected follow-up time.
    IPCW are computed using the Kaplan-Meier estimator, which is
    restricted to situations where the random censoring assumption holds and
    censoring is independent of the features.

    The function also provides a single summary measure that refers to the mean
    of the :math:`\\mathrm{AUC}(t)` over the time range :math:`(\\tau_1, \\tau_2)`.

    .. math::

        \\overline{\\mathrm{AUC}}(\\tau_1, \\tau_2) =
        \\frac{1}{\\hat{S}(\\tau_1) - \\hat{S}(\\tau_2)}
        \\int_{\\tau_1}^{\\tau_2} \\widehat{\\mathrm{AUC}}(t)\\,d \\hat{S}(t)

    where :math:`\\hat{S}(t)` is the Kaplan–Meier estimator of the survival function.


    Parameters
    ----------
    survival_train : structured array, shape = (n_train_samples,)
        Survival times for training data to estimate the censoring
        distribution from.
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.

    survival_test : structured array, shape = (n_samples,)
        Survival times of test data.
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.

    estimate : array-like, shape = (n_samples,)
        Estimated risk of experiencing an event of test data.

    times : array-like, shape = (n_times,)
        The time points for which the area under the
        time-dependent ROC curve is computed. Values must be
        within the range of follow-up times of the test data
        `survival_test`.

    tied_tol : float, optional, default: 1e-8
        The tolerance value for considering ties.
        If the absolute difference between risk scores is smaller
        or equal than `tied_tol`, risk scores are considered tied.

    Returns
    -------
    auc : array, shape = (n_times,)
        The cumulative/dynamic AUC estimates (evaluated at `times`).
    mean_auc : float
        Summary measure referring to the mean cumulative/dynamic AUC
        over the specified time range `(times[0], times[-1])`.

    References
    ----------
    .. [1] H. Uno, T. Cai, L. Tian, and L. J. Wei,
           "Evaluating prediction rules for t-year survivors with censored regression models,"
           Journal of the American Statistical Association, vol. 102, pp. 527–537, 2007.
    .. [2] H. Hung and C. T. Chiang,
           "Estimation methods for time-dependent AUC models with survival data,"
           Canadian Journal of Statistics, vol. 38, no. 1, pp. 8–26, 2010.
    .. [3] J. Lambert and S. Chevret,
           "Summary measure of discrimination in survival models based on cumulative/dynamic time-dependent ROC curves,"
           Statistical Methods in Medical Research, 2014.
    """
    test_event, test_time = check_y_survival(survival_test)

    estimate = _check_estimate(estimate, test_time)

    times = check_array(numpy.atleast_1d(times), ensure_2d=False, dtype=test_time.dtype)
    times = numpy.unique(times)

    if times.max() >= test_time.max() or times.min() < test_time.min():
        raise ValueError(
            'all times must be within follow-up time of test data: [{}; {}['.format(
                test_time.min(), test_time.max()))

    # sort by risk score (descending)
    o = numpy.argsort(-estimate)
    test_time = test_time[o]
    test_event = test_event[o]
    estimate = estimate[o]
    survival_test = survival_test[o]

    cens = CensoringDistributionEstimator()
    cens.fit(survival_train)
    ipcw = cens.predict_ipcw(survival_test)

    n_samples = test_time.shape[0]
    scores = numpy.empty(times.shape[0], dtype=float)
    for k, t in enumerate(times):
        is_case = (test_time <= t) & test_event
        is_control = test_time > t
        n_controls = is_control.sum()

        true_pos = []
        false_pos = []
        tp_value = 0.0
        fp_value = 0.0
        est_prev = numpy.infty

        for i in range(n_samples):
            est = estimate[i]
            if numpy.absolute(est - est_prev) > tied_tol:
                true_pos.append(tp_value)
                false_pos.append(fp_value)
                est_prev = est
            if is_case[i]:
                tp_value += ipcw[i]
            elif is_control[i]:
                fp_value += 1
        true_pos.append(tp_value)
        false_pos.append(fp_value)

        sens = numpy.array(true_pos) / ipcw[is_case].sum()
        fpr = numpy.array(false_pos) / n_controls
        scores[k] = trapz(sens, fpr)

    if times.shape[0] == 1:
        mean_auc = scores[0]
    else:
        surv = SurvivalFunctionEstimator()
        surv.fit(survival_test)
        s_times = surv.predict_proba(times)
        # compute integral of AUC over survival function
        d = -numpy.diff(numpy.concatenate(([1.0], s_times)))
        integral = (scores * d).sum()
        mean_auc = integral / (1.0 - s_times[-1])

    return scores, mean_auc

def brier_score(survival_train,survival_test, estimate, times,
                     t_max=None,
                     use_mean_point=False,
                     internal_validation=True,
                     **kwargs):
    """ 
    Modification of the implementation in PySurvival by Stephane Fotso et al.
    TODO: NEED TO SHIP WITH AN APACHE LICENSE
    Computing the Brier score at all times t such that t <= t_max;
    it represents the average squared distances between 
    the observed survival status and the predicted
    survival probability.
    In the case of right censoring, it is necessary to adjust
    the score by weighting the squared distances to 
    avoid bias. It can be achieved by using 
    the inverse probability of censoring weights method (IPCW),
    (proposed by Graf et al. 1999; Gerds and Schumacher 2006)
    by using the estimator of the conditional survival function
    of the censoring times calculated using the Kaplan-Meier method,
    such that :
    BS(t) = 1/N*( W_1(t)*(Y_1(t) - S_1(t))^2 + ... + 
                  W_N(t)*(Y_N(t) - S_N(t))^2)
    In terms of benchmarks, a useful model will have a Brier score below 
    0.25. Indeed, it is easy to see that if for all i in [1,N], 
    if S(t, xi) = 0.5, then BS(t) = 0.25.
    Parameters:
    -----------
    survival_train : structured array, shape = (n_train_samples,)
        Survival times for training data to estimate if training
        and testing data are drawn from same sample.
        Set internal_validation to True in this case.
        Otherwise, use surival_test again as input.
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.

    survival_test : structured array, shape = (n_samples,)
        Survival times of test data.
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.

    estimate : array-like, shape = (n_samples,n_times)
        Estimated risk of experiencing an event for test data at `times`.

    times : array-like, shape = (n_times,)
        The time points for which the predicted Survival function
        is calculated and interpolation for a specific follow-up-time
        will be calculated from. Values must be
        within the range of follow-up times of the test data
        `survival_test`.
    
        
    * t_max: float 
        Maximal time for estimating the prediction error curves. 
        If missing the largest value of the response variable is used.
        
    * t_max: use_mean_point 
        not necessary at the moment. 
        Predicted survival will be calculated at the mean of a time bucket (between 2 breaks)
    Returns:
    --------
        * times : array, shape = (n_times*)
            represents the time axis (length n_times* = n_times[times <= t_max] at which the brier scores were 
            computed
        * brier_scores : array , shape = (n_times*)
            values of the brier scores
                    
    Example:
    --------
    """
    # check inputs
    test_event, test_time = check_y_survival(survival_test)

    
    times = check_array(numpy.atleast_1d(times), ensure_2d=False, dtype=test_time.dtype)
    times = numpy.unique(times)

#    if times.max() >= test_time.max() or times.min() < test_time.min():
#        raise ValueError(
#            'all times must be within follow-up time of test data: [{}; {}['.format(
#                test_time.min(), test_time.max()))
#


    # Checking the format of the data 
    E,T = check_y_survival(survival_test)

    # computing the Survival function at times
    Survival = estimate


    # Ordering Survival, T and E in descending order according to T
    order = numpy.argsort(-T)
    Survival = Survival[order, :]
    T = T[order]
    E = E[order]
    survival_test=survival_test[order]
    
    #fit IPCW estimator for estimation of IPCW at time t*
    cens=CensoringDistributionEstimator()
    if internal_validation: cens.fit(survival_train)
    else: cens.fit(survival_test)
    
    #setting time to last time observed, if not t_max set
    if t_max is None or t_max <= 0.:
        t_max = max(T)

    # Calculating the brier scores at each t <= t_max
    brierlist=[]
    for t in times[times<=t_max]:
        # init bs
        bs = numpy.zeros((T.shape[0]))
        if(use_mean_point): # in case of time buckets (breaks), use mean probability in the bucket
            Survival = (numpy.add(Survival,numpy.roll(Survival,1,axis=-1)))/2. 


        is_case= (T <=t) & E
        is_control = (T > t)
        
        # get survival function S(t) by interpolating the Survival function
        S=numpy.interpolate(Survival,times,t)
        S2=numpy.multiply(S,S)
        omS2=numpy.multiply(1 - S,1 - S)
        
        #calculate inverse probability of censoring weight at current timepoint t.
        struct_arr=numpy.zeros((T.shape[0],),dtype=[('event','bool'),('time','int64')])
        struct_arr['time'][:]=t
        struct_arr['event'][:]=E
        ipcw_t=cens.predict_ipcw(struct_arr)
        
        #calculate inverse probability of censoring weights at observation T from survival_train
        ipcw=cens.predict_ipcw(survival_test)

        bs[is_case]= numpy.multiply(S2[is_case],ipcw[is_case]) # multiplicative IPCW at T[i]
        bs[is_control] = numpy.multiply(omS2[is_control],ipcw_t[is_control]) # multiplicative IPCW at current t
        brierlist.append(numpy.mean(bs))
    
    return times[times<=t_max],numpy.array(brierlist)

def integrated_brier_score(ev_dictlist, times, t_max=None):
    """ The Integrated Brier Score (IBS) provides an overall calculation of 
        the model performance at all available times.
    """
     # Computing the brier scores
    times, brier_scores = brier_score(ev_dictlist, times, t_max=t_max)

    # Getting the proper value of t_max
    if t_max is None:
        t_max = max(times)
    else:
        t_max = min(t_max, max(times))

    # Computing the IBS
    ibs_value = numpy.trapz(brier_scores, times)/t_max
    
    return ibs_value


def calibration_curve(survival_train,survival_test, estimate, times,
                     fu_time,
                     n_bins =10,
                     my_alpha = 0.7,
                     my_markersize = 4.,
                     color='#377eb8',
                     label='Actual versus predicted survival probability with 95% CI',
                     ci=True,
                     ci_alpha= .05,
                     pseudovals=True,
                     loess=True,
                     internal_validation = True,
                     ):
    """
    A calibration plot based on pseudovalues (pseudovals = True): 
    A product limit estimator (Kaplan-Meier) is used to generate a jackknife pseudo-value
    for the i'th observation, by calculating the product limit estimate for 
    a n-1 subsample without the i'th observation.
    Pseudovalues are then calculated by:
        
        est_i(t) = n * (prodlim(t)) - (n-1) * prodlim_i(t)
        
    where prodlim is the KM estimate for the whole sample and prodlim_i is the
    one applied to the subsample without i'th observation.
    
    see r package prodlim https://rdrr.io/cran/prodlim/man/jackknife.html
    see also    Gerds et al.,Calibration plots for risk prediction models in the presence of competing risks., https://doi.org/10.1002/sim.6152
                Graw et al., https://doi.org/10.1007/s10985-008-9107-z
    
    We combined the pseudovalues approach with loess smoothing (https://dx.doi.org/10.1002%2Fsim.5941)
    Note: span = 0.75 (default) works good according to https://dx.doi.org/10.1002%2Fsim.5941
    
    Parameters:
    -----------
    survival_train : structured array, shape = (n_train_samples,)
        Survival times for training data to estimate if training
        and testing data are drawn from same sample.
        Set internal_validation to True in this case.
        Otherwise, use surival_test again as input.
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.

    survival_test : structured array, shape = (n_samples,)
        Survival times of test data.
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.

    estimate : array-like, shape = (n_samples,n_times)
        Estimated risk of experiencing an event for test data at `times`.

    times : array-like, shape = (n_times,)
        The time points for which the predicted Survival function
        is calculated and interpolation for a specific follow-up-time
        is calculated. Values must be
        within the range of follow-up times of the test data
        `survival_test`.
    
    fu_time : float,
        The timepoint for which the calibration curve should be plotted.
        
    * color: str
        color for the plot
        
    * label: str
        label of the plot
        
    * trunc: int / float
        truncate pred_surv at trunc'th percentile.
        
    * my_alpha: float
        alpha transluicency for plot
        
    * my_markersize: float
        markersize for pyplot
                
    * ci: bool
        controls, whether confidence intervals should be calculated
        
    * ci_alpha: float
        confidence niveau, default 0.05
    * loess: bool
        use loess smoothing (recommended) or k nearest neighbors regression?
    * pseudovals: bool
        controls, whether pseudovals or binning approach should be used.
    * internal_validation: bool
        survival_train and survival_test are considered as beeing drawn
        from same sample and are both used for calculation of Kaplan-Meier-estimates
    
    Returns:
    --------
        * (globmin_x,globmax_x,globmin_y, globmax_y) : tuple of floats
            - respective global extremum across the named axis
                    
    Example:
    --------

    """
    import matplotlib.pyplot as plt
    
    # check survival arrays for test_data
    test_event, test_time = check_y_survival(survival_test)
    train_event, train_time = check_y_survival(survival_train)
    times = check_array(numpy.atleast_1d(times), ensure_2d=False, dtype=test_time.dtype)
    
    if internal_validation:
        test_time_traintest=numpy.concatenate(test_time,train_time)
        test_event_traintest=numpy.concatenate(test_event,train_event)
    else:
        test_time_traintest=test_time
        test_event_traintest=test_event
    # interpolate predicted survival at fu_time.
    pred_surv = numpy.interpolate(estimate, times, fu_time)
    
    # sort by pred_surv in ascending order
    order=numpy.argsort(pred_surv)
    pred_surv=pred_surv[order]
    test_time=test_time[order]
    test_event=test_event[order]
    test_time_traintest = test_time_traintest[order]
    test_event_traintest = test_event_traintest[order]
    
    if pseudovals: # use pseudovals approach with loess smoothing
        (pred, actual,ci_upper,ci_lower)=_calib_plot_loess(fu_time,
                                                pred_surv, test_time,test_event,
                                                test_time_traintest, test_event_traintest,
                                                color=color,
                                                label=label,
                                                ci=ci,
                                                alpha=my_alpha,
                                                markersize=my_markersize,
                                                markertype='-',
                                                loess=loess)
    else: # use traditional binning (visual analog to Hosmer-Lemshaw-test)
        (pred, actual,ci_upper,ci_lower)=_calib_plot(fu_time,n_bins,
                                                pred_surv,test_time, test_event,
                                                color=color,
                                                label=label,
                                                error_bars=ci,
                                                alpha=my_alpha,
                                                markersize=my_markersize,
                                                markertype='-')
    if ci:
        globmin_x=pred.min() -0.02
        globmax_x=pred.max() +0.02
        globmin_y=ci_lower.min() -0.02
        globmax_y=ci_upper.max()+0.02
    else:
        globmin_x=min(actual.min(),pred.min())-0.02
        globmax_x=max(actual.max(),pred.max())+0.02
        globmin_y=min(actual.min(),pred.min())-0.02
        globmax_y=max(actual.max(),pred.max())+0.02

    return (globmin_x,globmax_x,globmin_y, globmax_y,)
