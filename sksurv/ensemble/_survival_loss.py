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
from abc import ABCMeta

import numpy as np
from sklearn._loss.link import IdentityLink
from sklearn._loss.loss import BaseLoss
from sklearn.utils.extmath import squared_norm

from ..nonparametric import ipc_weights
from ._coxph_loss import coxph_loss, coxph_negative_gradient


class SurvivalLossFunction(BaseLoss, metaclass=ABCMeta):  # noqa: B024
    """Base class for survival loss functions."""

    def __init__(self, sample_weight=None):
        super().__init__(closs=None, link=IdentityLink())


class CoxPH(SurvivalLossFunction):
    """Cox Partial Likelihood."""

    # pylint: disable=no-self-use

    def __call__(self, y_true, raw_prediction, sample_weight=None):  # pylint: disable=unused-argument
        """Compute the partial likelihood of prediction ``y_pred`` and ``y``."""
        # TODO add support for sample weights
        return coxph_loss(y_true["event"].astype(np.uint8), y_true["time"], raw_prediction.ravel())

    def gradient(self, y_true, raw_prediction, sample_weight=None, **kwargs):  # pylint: disable=unused-argument
        """
        Negative gradient of partial likelihood.

        Parameters
        ----------
        y : tuple, len = 2
            First element is boolean event indicator and second element survival/censoring time.
        y_pred : np.ndarray, shape=(n,):
            The predictions.

        Returns
        -------
        ndarray
            The gradient.
        """
        ret = coxph_negative_gradient(y_true["event"].astype(np.uint8), y_true["time"], raw_prediction.ravel())
        if sample_weight is not None:
            ret *= sample_weight
        return ret

    def update_terminal_regions(
        self, tree, X, y, residual, raw_predictions, sample_weight, sample_mask, learning_rate=0.1, k=0
    ):
        """
        Least squares does not need to update terminal regions.

        But it has to update the predictions.
        """
        # update predictions
        raw_predictions[:, k] += learning_rate * tree.predict(X).ravel()

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y, residual, raw_predictions, sample_weight):
        """Least squares does not need to update terminal regions."""

    def _scale_raw_prediction(self, raw_predictions):
        return raw_predictions


class CensoredSquaredLoss(SurvivalLossFunction):
    """
    Censoring-aware squared loss.

    Censoring is taken into account by only considering the residuals
    of samples that are not censored, or the predicted survival time
    is before the time of censoring.
    """

    # pylint: disable=no-self-use
    def __call__(self, y_true, raw_prediction, sample_weight=None):
        """Compute the partial likelihood of prediction ``y_pred`` and ``y``."""
        pred_time = y_true["time"] - raw_prediction.ravel()
        mask = (pred_time > 0) | y_true["event"]
        return 0.5 * squared_norm(pred_time.compress(mask, axis=0))

    def gradient(self, y_true, raw_prediction, **kwargs):  # pylint: disable=unused-argument
        """
        Negative gradient of partial likelihood.

        Parameters
        ----------
        y : tuple, len = 2
            First element is boolean event indicator and second element survival/censoring time.
        y_pred : np.ndarray, shape=(n,):
            The predictions.

        Returns
        -------
        ndarray
            The gradient.
        """
        pred_time = y_true["time"] - raw_prediction.ravel()
        mask = (pred_time > 0) | y_true["event"]
        ret = np.zeros(y_true["event"].shape[0])
        ret[mask] = pred_time.compress(mask, axis=0)
        return ret

    def update_terminal_regions(
        self, tree, X, y, residual, raw_predictions, sample_weight, sample_mask, learning_rate=0.1, k=0
    ):
        """
        Least squares does not need to update terminal regions.

        But it has to update the predictions.
        """
        # update predictions
        raw_predictions[:, k] += learning_rate * tree.predict(X).ravel()

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y, residual, raw_predictions, sample_weight):
        """Least squares does not need to update terminal regions."""

    def _scale_raw_prediction(self, raw_predictions):
        np.exp(raw_predictions, out=raw_predictions)
        return raw_predictions


class IPCWLeastSquaresError(SurvivalLossFunction):
    """Inverse probability of censoring weighted least squares error."""

    # pylint: disable=no-self-use

    def __call__(self, y_true, raw_prediction, sample_weight=None):
        sample_weight = ipc_weights(y_true["event"], y_true["time"])
        return 1.0 / sample_weight.sum() * np.sum(sample_weight * ((y_true["time"] - raw_prediction.ravel()) ** 2.0))

    def gradient(self, y_true, raw_prediction, **kwargs):  # pylint: disable=unused-argument
        return y_true["time"] - raw_prediction.ravel()

    def update_terminal_regions(self, tree, X, y, residual, y_pred, sample_weight, sample_mask, learning_rate=0.1, k=0):
        y_pred[:, k] += learning_rate * tree.predict(X).ravel()

    def _update_terminal_region(
        self, tree, terminal_regions, leaf, X, y, residual, pred, sample_weight
    ):  # pragma: no cover
        pass

    def _scale_raw_prediction(self, raw_predictions):
        np.exp(raw_predictions, out=raw_predictions)
        return raw_predictions


def encode_survival_target(event, time):
    """Pack event indicator and time into a single float array.

    Censored samples are stored with a negative sign, which allows the target to pass
    through scikit-learn's validation and train/validation splitting of a regression target.
    The sign bit is used for decoding, such that a censored time of zero remains censored.
    """
    time = np.asarray(time, dtype=np.float64)
    return np.where(event, time, -time)


def decode_survival_target(y):
    """Inverse of :func:`encode_survival_target`."""
    return ~np.signbit(y), np.abs(y)


def _breslow_log_risk_set(time, log_weighted_exp_f):
    """Logarithm of risk sets of the weighted partial likelihood with Breslow's handling of ties.

    Returns the order that sorts ``time`` and, in that order,
    ``log R_i`` with ``R_i = sum_{j: t_j >= t_i} s_j exp(f_j)``
    and the position of the last sample with the same time as sample ``i``.
    Sums are accumulated in log space, because predictions far apart would
    underflow ``exp`` and leave risk sets of events empty.
    """
    order = np.argsort(time, kind="stable")
    time_sorted = time[order]
    n_samples = time.shape[0]
    positions = np.arange(n_samples)
    # samples with tied times share the same risk set (first sample of block)
    # and the same set of preceding events (last sample of block)
    is_first = np.r_[True, time_sorted[1:] != time_sorted[:-1]]
    is_last = np.r_[is_first[1:], True]
    first_of_block = np.maximum.accumulate(np.where(is_first, positions, 0))
    last_of_block = np.minimum.accumulate(np.where(is_last, positions, n_samples)[::-1])[::-1]

    log_risk_set = np.logaddexp.accumulate(log_weighted_exp_f[order][::-1])[::-1][first_of_block]
    return order, log_risk_set, last_of_block


def _breslow_log_sums(event, time, raw_prediction, sample_weight):
    """Logarithm of sums over risk sets for the gradient and Hessian of the weighted partial likelihood.

    Returns, for each sample ``i`` in the original order, ``log(s_i exp(f_i))``,
    ``log A_i`` with ``A_i = sum_{k: event_k, t_k <= t_i} s_k / R_k``, and
    ``log B_i`` with ``B_i = sum_{k: event_k, t_k <= t_i} s_k / R_k^2``,
    where ``R_k`` is the risk set of :func:`_breslow_log_risk_set`.
    """
    weighted_event = sample_weight * event
    with np.errstate(divide="ignore"):
        log_weighted_exp_f = raw_prediction + np.log(sample_weight)
    order, log_risk_set, last_of_block = _breslow_log_risk_set(time, log_weighted_exp_f)

    # an event with positive weight belongs to its own risk set, which is therefore not empty,
    # other samples are dropped before an empty risk set turns their terms into NaN
    has_event = weighted_event[order] > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        log_a_terms = np.where(has_event, np.log(weighted_event[order]) - log_risk_set, -np.inf)
        log_b_terms = np.where(has_event, log_a_terms - log_risk_set, -np.inf)
    log_a = np.logaddexp.accumulate(log_a_terms)[last_of_block]
    log_b = np.logaddexp.accumulate(log_b_terms)[last_of_block]

    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.shape[0])
    return log_weighted_exp_f, log_a[inverse], log_b[inverse]


class HistCoxPH(BaseLoss):
    """Cox partial likelihood for histogram-based gradient boosting.

    The target must be encoded by :func:`encode_survival_target`.
    Since the Hessian of the partial likelihood is not diagonal,
    only its diagonal is used, as in XGBoost's ``survival:cox`` objective.
    """

    def __init__(self):
        super().__init__(closs=None, link=IdentityLink())

    def __call__(self, y_true, raw_prediction, sample_weight=None, n_threads=1):
        """Compute the negative log partial likelihood, averaged over samples like :meth:`BaseLoss.__call__`."""
        event, time = decode_survival_target(y_true)
        raw_prediction = raw_prediction.ravel()
        if sample_weight is None:
            sample_weight = np.ones_like(raw_prediction)
        with np.errstate(divide="ignore"):
            log_weighted_exp_f = raw_prediction + np.log(sample_weight)
        order, log_risk_set, _ = _breslow_log_risk_set(time, log_weighted_exp_f)
        weighted_event = (sample_weight * event)[order]
        has_event = weighted_event > 0
        log_likelihood = np.sum(
            weighted_event[has_event] * (raw_prediction[order][has_event] - log_risk_set[has_event])
        )
        return -log_likelihood / sample_weight.sum()

    def gradient_hessian(
        self,
        y_true,
        raw_prediction,
        sample_weight=None,
        gradient_out=None,
        hessian_out=None,
        n_threads=1,
    ):
        """Compute the gradient and the diagonal of the Hessian of the negative log partial likelihood."""
        event, time = decode_survival_target(y_true)
        raw_prediction = raw_prediction.ravel()
        if sample_weight is None:
            sample_weight = np.ones_like(raw_prediction)
        log_weighted_exp_f, log_a, log_b = _breslow_log_sums(event, time, raw_prediction, sample_weight)
        weighted_exp_f_a = np.exp(log_weighted_exp_f + log_a)
        gradient = weighted_exp_f_a - sample_weight * event
        hessian = weighted_exp_f_a - np.exp(2 * log_weighted_exp_f + log_b)

        if gradient_out is None:
            return gradient, hessian
        gradient_out[:] = gradient
        hessian_out[:] = hessian
        return gradient_out, hessian_out

    def fit_intercept_only(self, y_true, sample_weight=None):
        """The partial likelihood is invariant to adding a constant, therefore zero."""
        return np.zeros(1)


LOSS_FUNCTIONS = {
    "coxph": CoxPH,
    "squared": CensoredSquaredLoss,
    "ipcwls": IPCWLeastSquaresError,
}
