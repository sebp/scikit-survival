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
    """Cox Partial Likelihood"""

    # pylint: disable=no-self-use

    def __call__(self, y_true, raw_prediction, sample_weight=None):  # pylint: disable=unused-argument
        """Compute the partial likelihood of prediction ``y_pred`` and ``y``."""
        # TODO add support for sample weights
        return coxph_loss(y_true["event"].astype(np.uint8), y_true["time"], raw_prediction.ravel())

    def gradient(self, y_true, raw_prediction, sample_weight=None, **kwargs):  # pylint: disable=unused-argument
        """Negative gradient of partial likelihood

        Parameters
        ---------
        y : tuple, len = 2
            First element is boolean event indicator and second element survival/censoring time.
        y_pred : np.ndarray, shape=(n,):
            The predictions.
        """
        ret = coxph_negative_gradient(y_true["event"].astype(np.uint8), y_true["time"], raw_prediction.ravel())
        if sample_weight is not None:
            ret *= sample_weight
        return ret

    def update_terminal_regions(
        self, tree, X, y, residual, raw_predictions, sample_weight, sample_mask, learning_rate=0.1, k=0
    ):
        """Least squares does not need to update terminal regions.

        But it has to update the predictions.
        """
        # update predictions
        raw_predictions[:, k] += learning_rate * tree.predict(X).ravel()

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y, residual, raw_predictions, sample_weight):
        """Least squares does not need to update terminal regions"""

    def _scale_raw_prediction(self, raw_predictions):
        return raw_predictions


class CensoredSquaredLoss(SurvivalLossFunction):
    """Censoring-aware squared loss.

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
        """Negative gradient of partial likelihood

        Parameters
        ---------
        y : tuple, len = 2
            First element is boolean event indicator and second element survival/censoring time.
        y_pred : np.ndarray, shape=(n,):
            The predictions.
        """
        pred_time = y_true["time"] - raw_prediction.ravel()
        mask = (pred_time > 0) | y_true["event"]
        ret = np.zeros(y_true["event"].shape[0])
        ret[mask] = pred_time.compress(mask, axis=0)
        return ret

    def update_terminal_regions(
        self, tree, X, y, residual, raw_predictions, sample_weight, sample_mask, learning_rate=0.1, k=0
    ):
        """Least squares does not need to update terminal regions.

        But it has to update the predictions.
        """
        # update predictions
        raw_predictions[:, k] += learning_rate * tree.predict(X).ravel()

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y, residual, raw_predictions, sample_weight):
        """Least squares does not need to update terminal regions"""

    def _scale_raw_prediction(self, raw_predictions):
        np.exp(raw_predictions, out=raw_predictions)
        return raw_predictions


class IPCWLeastSquaresError(SurvivalLossFunction):
    """Inverse probability of censoring weighted least squares error"""

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


LOSS_FUNCTIONS = {
    "coxph": CoxPH,
    "squared": CensoredSquaredLoss,
    "ipcwls": IPCWLeastSquaresError,
}
