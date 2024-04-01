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


class SurvivalAnalysisMixin:
    def _predict_function(self, func_name, baseline_model, prediction, return_array):
        fns = getattr(baseline_model, func_name)(prediction)

        if not return_array:
            return fns

        times = baseline_model.unique_times_
        arr = np.empty((prediction.shape[0], times.shape[0]), dtype=float)
        for i, fn in enumerate(fns):
            arr[i, :] = fn(times)
        return arr

    def _predict_survival_function(self, baseline_model, prediction, return_array):
        """Return survival functions.

        Parameters
        ----------
        baseline_model : sksurv.linear_model.coxph.BreslowEstimator
            Estimator of baseline survival function.

        prediction : array-like, shape=(n_samples,)
            Predicted risk scores.

        return_array : bool
            If True, return a float array of the survival function
            evaluated at the unique event times, otherwise return
            an array of :class:`sksurv.functions.StepFunction` instances.

        Returns
        -------
        survival : ndarray
        """
        return self._predict_function("get_survival_function", baseline_model, prediction, return_array)

    def _predict_cumulative_hazard_function(self, baseline_model, prediction, return_array):
        """Return cumulative hazard functions.

        Parameters
        ----------
        baseline_model : sksurv.linear_model.coxph.BreslowEstimator
            Estimator of baseline cumulative hazard function.

        prediction : array-like, shape=(n_samples,)
            Predicted risk scores.

        return_array : bool
            If True, return a float array of the cumulative hazard function
            evaluated at the unique event times, otherwise return
            an array of :class:`sksurv.functions.StepFunction` instances.

        Returns
        -------
        cum_hazard : ndarray
        """
        return self._predict_function("get_cumulative_hazard_function", baseline_model, prediction, return_array)

    def score(self, X, y):
        """Returns the concordance index of the prediction.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        y : structured array, shape = (n_samples,)
            A structured array containing the binary event indicator
            as first field, and time of event or time of censoring as
            second field.

        Returns
        -------
        cindex : float
            Estimated concordance index.
        """
        from .metrics import concordance_index_censored

        name_event, name_time = y.dtype.names

        risk_score = self.predict(X)
        if not getattr(self, "_predict_risk_score", True):
            risk_score *= -1  # convert prediction on time scale to risk scale

        result = concordance_index_censored(y[name_event], y[name_time], risk_score)
        return result[0]

    def _more_tags(self):
        return {"requires_y": True}
