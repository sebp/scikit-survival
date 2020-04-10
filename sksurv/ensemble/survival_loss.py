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
import numpy

from sklearn.dummy import DummyRegressor
from sklearn.ensemble._gb_losses import RegressionLossFunction
from sklearn.utils.extmath import squared_norm

from ..nonparametric import ipc_weights
from ._coxph_loss import coxph_loss, coxph_negative_gradient


class DummySurvivalEstimator(DummyRegressor):

    def __init__(self, strategy="mean", constant=None, quantile=None):
        super().__init__(
            strategy=strategy,
            constant=constant,
            quantile=quantile,
        )

    def fit(self, X, y, sample_weight=None):
        _, time = y
        return super().fit(X, time, sample_weight=sample_weight)


class SurvivalLossFunction(RegressionLossFunction, metaclass=ABCMeta):
    """Base class for survival loss functions."""
    # pylint: disable=abstract-method,no-self-use
    def init_estimator(self):
        return DummySurvivalEstimator(strategy='constant', constant=0.)


class CoxPH(SurvivalLossFunction):
    """Cox Partial Likelihood"""
    # pylint: disable=no-self-use

    def __call__(self, y, raw_predictions, sample_weight=None):
        """Compute the partial likelihood of prediction ``y_pred`` and ``y``."""
        # TODO add support for sample weights
        return coxph_loss(y['event'].astype(numpy.uint8), y['time'], raw_predictions.ravel())

    def negative_gradient(self, y, raw_predictions, sample_weight=None, **kwargs):
        """Negative gradient of partial likelihood

        Parameters
        ---------
        y : tuple, len = 2
            First element is boolean event indicator and second element survival/censoring time.
        y_pred : np.ndarray, shape=(n,):
            The predictions.
        """
        ret = coxph_negative_gradient(
            y['event'].astype(numpy.uint8), y['time'], raw_predictions.ravel())
        if sample_weight is not None:
            ret *= sample_weight
        return ret

    def update_terminal_regions(self, tree, X, y, residual, raw_predictions,
                                sample_weight, sample_mask,
                                learning_rate=0.1, k=0):
        """Least squares does not need to update terminal regions.

        But it has to update the predictions.
        """
        # update predictions
        raw_predictions[:, k] += learning_rate * tree.predict(X).ravel()

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, raw_predictions, sample_weight):
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
    def __call__(self, y, raw_predictions, sample_weight=None):
        """Compute the partial likelihood of prediction ``y_pred`` and ``y``."""
        pred_time = y['time'] - raw_predictions.ravel()
        mask = (pred_time > 0) | y['event']
        return 0.5 * squared_norm(pred_time.compress(mask, axis=0))

    def negative_gradient(self, y, raw_predictions, **kwargs):
        """Negative gradient of partial likelihood

        Parameters
        ---------
        y : tuple, len = 2
            First element is boolean event indicator and second element survival/censoring time.
        y_pred : np.ndarray, shape=(n,):
            The predictions.
        """
        pred_time = y['time'] - raw_predictions.ravel()
        mask = (pred_time > 0) | y['event']
        ret = numpy.zeros(y['event'].shape[0])
        ret[mask] = pred_time.compress(mask, axis=0)
        return ret

    def update_terminal_regions(self, tree, X, y, residual, raw_predictions,
                                sample_weight, sample_mask,
                                learning_rate=0.1, k=0):
        """Least squares does not need to update terminal regions.

        But it has to update the predictions.
        """
        # update predictions
        raw_predictions[:, k] += learning_rate * tree.predict(X).ravel()

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, raw_predictions, sample_weight):
        """Least squares does not need to update terminal regions"""

    def _scale_raw_prediction(self, raw_predictions):
        numpy.exp(raw_predictions, out=raw_predictions)
        return raw_predictions


class IPCWLeastSquaresError(SurvivalLossFunction):
    """Inverse probability of censoring weighted least squares error"""
    # pylint: disable=no-self-use

    def __call__(self, y, raw_predictions, sample_weight=None):
        sample_weight = ipc_weights(y['event'], y['time'])
        return (1.0 / sample_weight.sum()
                * numpy.sum(sample_weight * ((y['time'] - raw_predictions.ravel()) ** 2.0)))

    def negative_gradient(self, y, raw_predictions, **kwargs):
        return y['time'] - raw_predictions.ravel()

    def update_terminal_regions(self, tree, X, y, residual, y_pred,
                                sample_weight, sample_mask,
                                learning_rate=0.1, k=0):
        y_pred[:, k] += learning_rate * tree.predict(X).ravel()

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, pred, sample_weight):  # pragma: no cover
        pass

    def _scale_raw_prediction(self, raw_predictions):
        numpy.exp(raw_predictions, out=raw_predictions)
        return raw_predictions


LOSS_FUNCTIONS = {
    "coxph": CoxPH,
    "squared": CensoredSquaredLoss,
    "ipcwls": IPCWLeastSquaresError,
}
