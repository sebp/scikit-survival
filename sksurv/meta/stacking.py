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
from sklearn.base import MetaEstimatorMixin, clone
from sklearn.utils._param_validation import HasMethods
from sklearn.utils.metaestimators import _BaseComposition, available_if

from ..base import SurvivalAnalysisMixin
from ..util import property_available_if


def _meta_estimator_has(attr):
    """Check that meta_estimator has `attr`.

    Used together with `available_if`."""

    def check(self):
        # raise original `AttributeError` if `attr` does not exist
        getattr(self.meta_estimator, attr)
        return True

    return check


class Stacking(MetaEstimatorMixin, SurvivalAnalysisMixin, _BaseComposition):
    """Meta estimator that combines multiple base learners.

    By default, base estimators' output corresponds to the array returned
    by `predict_proba`. If `predict_proba` is not available or `probabilities = False`,
    the output of `predict` is used.

    Parameters
    ----------
    meta_estimator : instance of estimator
        The estimator that is used to combine the output of different
        base estimators.

    base_estimators : list
        List of (name, estimator) tuples (implementing fit/predict) that are
        part of the ensemble.

    probabilities : bool, optional, default: True
        Whether to allow using `predict_proba` method of base learners, if available.

    Attributes
    ----------
    estimators_ : list of estimators
        The elements of the estimators parameter, having been fitted on the
        training data.

    named_estimators_ : dict
        Attribute to access any fitted sub-estimators by name.

    final_estimator_ : estimator
        The estimator which combines the output of `estimators_`.

    n_features_in_ : int
        Number of features seen during ``fit``.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during ``fit``. Defined only when `X`
        has feature names that are all strings.

    unique_times_ : array of shape = (n_unique_times,)
        Unique time points.
    """

    _parameter_constraints = {
        "meta_estimator": [HasMethods(["fit"])],
        "base_estimators": [list],
        "probabilities": ["boolean"],
    }

    def __init__(self, meta_estimator, base_estimators, *, probabilities=True):
        self.meta_estimator = meta_estimator
        self.base_estimators = base_estimators
        self.probabilities = probabilities

        self._extra_params = ["meta_estimator", "base_estimators", "probabilities"]

    def _validate_estimators(self):
        names, estimators = zip(*self.base_estimators)
        if len(set(names)) != len(self.base_estimators):
            raise ValueError(f"Names provided are not unique: {names}")

        for t in estimators:
            if not hasattr(t, "fit") or not (hasattr(t, "predict") or hasattr(t, "predict_proba")):
                raise TypeError(
                    "All base estimators should implement "
                    "fit and predict/predict_proba"
                    f" {t!s} (type {type(t)}) doesn't)"
                )

    def set_params(self, **params):
        """
        Set the parameters of an estimator from the ensemble.

        Valid parameter keys can be listed with `get_params()`. Note that you
        can directly set the parameters of the estimators contained in
        `estimators`.

        Parameters
        ----------
        **params : keyword arguments
            Specific parameters using e.g.
            `set_params(parameter_name=new_value)`. In addition, to setting the
            parameters of the estimator, the individual estimator of the
            estimators can also be set, or can be removed by setting them to
            'drop'.

        Returns
        -------
        self : object
            Estimator instance.
        """
        super()._set_params("base_estimators", **params)
        return self

    def get_params(self, deep=True):
        """
        Get the parameters of an estimator from the ensemble.

        Returns the parameters given in the constructor as well as the
        estimators contained within the `estimators` parameter.

        Parameters
        ----------
        deep : bool, default=True
            Setting it to True gets the various estimators and the parameters
            of the estimators as well.

        Returns
        -------
        params : dict
            Parameter and estimator names mapped to their values or parameter
            names mapped to their values.
        """
        return super()._get_params("base_estimators", deep=deep)

    def _split_fit_params(self, fit_params):
        fit_params_steps = {step: {} for step, _ in self.base_estimators}
        for pname, pval in fit_params.items():
            step, param = pname.split("__", 1)
            fit_params_steps[step][param] = pval
        return fit_params_steps

    def _fit_estimators(self, X, y=None, **fit_params):
        if hasattr(self, "feature_names_in_"):
            # Delete the attribute when the estimator is fitted on a new dataset
            # that has no feature names.
            delattr(self, "feature_names_in_")

        fit_params_steps = self._split_fit_params(fit_params)
        names = []
        estimators = []
        for name, estimator in self.base_estimators:
            est = clone(estimator).fit(X, y, **fit_params_steps[name])

            if hasattr(est, "n_features_in_"):
                self.n_features_in_ = est.n_features_in_
            if hasattr(est, "feature_names_in_"):
                self.feature_names_in_ = est.feature_names_in_

            estimators.append(est)
            names.append(name)

        self.named_estimators = dict(zip(names, estimators))
        self.estimators_ = estimators

    def _predict_estimators(self, X):
        Xt = None
        start = 0
        for estimator in self.estimators_:
            if self.probabilities and hasattr(estimator, "predict_proba"):
                p = estimator.predict_proba(X)
            else:
                p = estimator.predict(X)

            if p.ndim == 1:
                p = p[:, np.newaxis]

            if Xt is None:
                # assume that prediction array has the same size for all base learners
                n_classes = p.shape[1]
                Xt = np.empty((p.shape[0], n_classes * len(self.base_estimators)), order="F")
            Xt[:, slice(start, start + n_classes)] = p
            start += n_classes

        return Xt

    def __len__(self):
        return len(self.base_estimators)

    def fit(self, X, y=None, **fit_params):
        """Fit base estimators.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data.

        y : array-like, optional
            Target data if base estimators are supervised.

        Returns
        -------
        self
        """
        self._validate_params()
        self._validate_estimators()
        self._fit_estimators(X, y, **fit_params)
        Xt = self._predict_estimators(X)
        self.final_estimator_ = self.meta_estimator.fit(Xt, y)

        return self

    @available_if(_meta_estimator_has("predict"))
    def predict(self, X):
        """Perform prediction.

        Only available of the meta estimator has a predict method.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data with samples to predict.

        Returns
        -------
        prediction : array, shape = (n_samples, n_dim)
            Prediction of meta estimator that combines
            predictions of base estimators. `n_dim` depends
            on the return value of meta estimator's `predict`
            method.
        """
        Xt = self._predict_estimators(X)
        return self.final_estimator_.predict(Xt)

    @available_if(_meta_estimator_has("predict_proba"))
    def predict_proba(self, X):
        """Perform prediction.

        Only available if the meta estimator has a predict_proba method.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data with samples to predict.

        Returns
        -------
        prediction : ndarray, shape = (n_samples, n_dim)
            Prediction of meta estimator that combines
            predictions of base estimators. `n_dim` depends
            on the return value of meta estimator's `predict`
            method.
        """
        Xt = self._predict_estimators(X)
        return self.final_estimator_.predict_proba(Xt)

    @available_if(_meta_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X):
        """Perform prediction.

        Only available if the meta estimator has a predict_log_proba method.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data with samples to predict.

        Returns
        -------
        prediction : ndarray, shape = (n_samples, n_dim)
            Prediction of meta estimator that combines
            predictions of base estimators. `n_dim` depends
            on the return value of meta estimator's `predict`
            method.
        """
        Xt = self._predict_estimators(X)
        return self.final_estimator_.predict_log_proba(Xt)

    @property_available_if(_meta_estimator_has("unique_times_"))
    def unique_times_(self):
        return self.meta_estimator.unique_times_

    @available_if(_meta_estimator_has("predict_cumulative_hazard_function"))
    def predict_cumulative_hazard_function(self, X, return_array=False):
        """Perform prediction.

        Only available if the meta estimator has a predict_cumulative_hazard_function method.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data with samples to predict.

        return_array : boolean, default: False
            If set, return an array with the cumulative hazard rate
            for each `self.unique_times_`, otherwise an array of
            :class:`sksurv.functions.StepFunction`.

        Returns
        -------
        cum_hazard : ndarray
            If `return_array` is set, an array with the cumulative hazard rate
            for each `self.unique_times_`, otherwise an array of length `n_samples`
            of :class:`sksurv.functions.StepFunction` instances will be returned.
        """
        Xt = self._predict_estimators(X)
        return self.final_estimator_.predict_cumulative_hazard_function(Xt, return_array)

    @available_if(_meta_estimator_has("predict_survival_function"))
    def predict_survival_function(self, X, return_array=False):
        """Perform prediction.

        Only available if the meta estimator has a predict_survival_function method.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data with samples to predict.

        Returns
        -------
        survival : ndarray
            If `return_array` is set, an array with the probability of
            survival for each `self.unique_times_`, otherwise an array of
            length `n_samples` of :class:`sksurv.functions.StepFunction`
            instances will be returned.

        return_array : boolean, default: False
            If set, return an array with the probability
            of survival for each `self.unique_times_`,
            otherwise an array of :class:`sksurv.functions.StepFunction`.

        """
        Xt = self._predict_estimators(X)
        return self.final_estimator_.predict_survival_function(Xt, return_array)
