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

from sklearn.ensemble.gradient_boosting import LeastSquaresError, LossFunction, ZeroEstimator
from sklearn.utils.extmath import squared_norm

from ..nonparametric import ipc_weights
from ._coxph_loss import coxph_loss, coxph_negative_gradient


class ZeroSurvivalEstimator(ZeroEstimator):

    def fit(self, X, y, sample_weight=None):
        event, time = y
        return super().fit(X, time, sample_weight=sample_weight)


class CoxPH(LossFunction):
    """Cox Partial Likelihood"""

    def __call__(self, y, y_pred, sample_weight=None):
        """Compute the partial likelihood of prediction ``y_pred`` and ``y``."""
        # TODO add support for sample weights
        return coxph_loss(y['event'].astype(numpy.uint8), y['time'], y_pred.ravel())

    def negative_gradient(self, y, y_pred, sample_weight=None, **kwargs):
        """Negative gradient of partial likelihood

        Parameters
        ---------
        y : tuple, len = 2
            First element is boolean event indicator and second element survival/censoring time.
        y_pred : np.ndarray, shape=(n,):
            The predictions.
        """
        ret = coxph_negative_gradient(y['event'].astype(numpy.uint8), y['time'], y_pred.ravel())
        if sample_weight is not None:
            ret *= sample_weight
        return ret

    def init_estimator(self):
        return ZeroEstimator()

    def update_terminal_regions(self, tree, X, y, residual, y_pred,
                                sample_weight, sample_mask,
                                learning_rate=1.0, k=0):
        """Least squares does not need to update terminal regions.

        But it has to update the predictions.
        """
        # update predictions
        y_pred[:, k] += learning_rate * tree.predict(X).ravel()

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, pred, sample_weight):
        pass


class CensoredSquaredLoss(LossFunction):
    """Censoring-aware squared loss.

    Censoring is taken into account by only considering the residuals
    of samples that are not censored, or the predicted survival time
    is before the time of censoring.
    """
    def __call__(self, y, y_pred, sample_weight=None):
        """Compute the partial likelihood of prediction ``y_pred`` and ``y``."""
        pred_time = y['time'] - y_pred.ravel()
        mask = (pred_time > 0) | y['event']
        return 0.5 * squared_norm(pred_time.compress(mask, axis=0))

    def negative_gradient(self, y, y_pred, sample_weight=None, **kwargs):
        """Negative gradient of partial likelihood

        Parameters
        ---------
        y : tuple, len = 2
            First element is boolean event indicator and second element survival/censoring time.
        y_pred : np.ndarray, shape=(n,):
            The predictions.
        """
        pred_time = y['time'] - y_pred.ravel()
        mask = (pred_time > 0) | y['event']
        ret = numpy.zeros(y['event'].shape[0])
        ret[mask] = pred_time.compress(mask, axis=0)
        if sample_weight is not None:
            ret *= sample_weight
        return ret

    def init_estimator(self):
        return ZeroEstimator()

    def update_terminal_regions(self, tree, X, y, residual, y_pred,
                                sample_weight, sample_mask,
                                learning_rate=1.0, k=0):
        """Least squares does not need to update terminal regions.

        But it has to update the predictions.
        """
        # update predictions
        y_pred[:, k] += learning_rate * tree.predict(X).ravel()

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, pred, sample_weight):
        pass


class IPCWLeastSquaresError(LeastSquaresError):
    """Inverse probability of censoring weighted least squares error"""

    def __call__(self, y, pred, sample_weight=None):
        sample_weight = ipc_weights(y['event'], y['time'])
        return (1.0 / sample_weight.sum() *
                numpy.sum(sample_weight * ((y['time'] - pred.ravel()) ** 2.0)))

    def negative_gradient(self, y, pred, **kwargs):
        return y['time'] - pred.ravel()
