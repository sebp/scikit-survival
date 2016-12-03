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
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.utils import tosequence
from sklearn.utils.metaestimators import if_delegate_has_method


class Stacking(BaseEstimator, MetaEstimatorMixin):
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

    probabilities : bool, optional, default=True
        Whether to allow using `predict_proba` method of base learners, if available.
    """

    def __init__(self, meta_estimator, base_estimators, probabilities=True):
        self.meta_estimator = meta_estimator
        self.probabilities = probabilities

        self.named_estimators = dict(base_estimators)
        names, estimators = zip(*base_estimators)
        if len(self.named_estimators) != len(base_estimators):
            raise ValueError("Names provided are not unique: %s" % (names,))

        # shallow copy of steps
        self.base_estimators = tosequence(zip(names, estimators))

        self._extra_params = ["meta_estimator", "probabilities"]

        for t in estimators:
            if not hasattr(t, "fit") or not (hasattr(t, "predict") or hasattr(t, "predict_proba")):
                raise TypeError("All base estimators should implement "
                                "fit and predict/predict_proba"
                                " '%s' (type %s) doesn't)" % (t, type(t)))

        if not hasattr(meta_estimator, "fit"):
            raise TypeError("meta estimator should implement fit "
                            "'%s' (type %s) doesn't)"
                            % (meta_estimator, type(meta_estimator)))

    def get_params(self, deep=True):
        if not deep:
            return super(Stacking, self).get_params(deep=False)
        else:
            out = self.named_estimators.copy()
            for name, estimator in self.named_estimators.items():
                for key, value in estimator.get_params(deep=True).items():
                    out['%s__%s' % (name, key)] = value

            for param in self._extra_params:
                out[param] = getattr(self, param)
            return out

    def _split_fit_params(self, fit_params):
        fit_params_steps = dict((step, {}) for step, _ in self.base_estimators)
        for pname, pval in fit_params.items():
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval
        return fit_params_steps

    def _fit_estimators(self, X, y=None, **fit_params):
        fit_params_steps = self._split_fit_params(fit_params)
        for name, estimator in self.base_estimators:
            estimator.fit(X, y, **fit_params_steps[name])

    def _predict_estimators(self, X):
        Xt = None
        start = 0
        for _, estimator in self.base_estimators:

            if self.probabilities and hasattr(estimator, "predict_proba"):
                p = estimator.predict_proba(X)
            else:
                p = estimator.predict(X)

            if p.ndim == 1:
                p = p[:, numpy.newaxis]

            if Xt is None:
                # assume that prediction array has the same size for all base learners
                n_classes = p.shape[1]
                Xt = numpy.empty((p.shape[0], n_classes * len(self.base_estimators)), order='F')
            Xt[:, slice(start, start + n_classes)] = p
            start += n_classes

        return Xt

    def __len__(self):
        return len(self.base_estimators)

    def fit(self, X, y=None, **fit_params):
        """Fit base estimators.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training data.

        y : array-like, optional
            Target data if base estimators are supervised.

        Returns
        -------
        self
        """
        X = numpy.asarray(X)
        self._fit_estimators(X, y, **fit_params)
        Xt = self._predict_estimators(X)
        self.meta_estimator.fit(Xt, y)

        return self

    @if_delegate_has_method(delegate='meta_estimator')
    def predict(self, X):
        """Perform prediction.

        Only available of the meta estimator has a predict method.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Data with samples to predict.

        Returns
        -------
        prediction : array, shape = [n_samples, n_dim]
            Prediction of meta estimator that combines
            predictions of base estimators. `n_dim` depends
            on the return value of meta estimator's `predict`
            method.
        """
        X = numpy.asarray(X)
        Xt = self._predict_estimators(X)
        return self.meta_estimator.predict(Xt)

    @if_delegate_has_method(delegate='meta_estimator')
    def predict_proba(self, X):
        """Perform prediction.

        Only available of the meta estimator has a predict_proba method.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Data with samples to predict.

        Returns
        -------
        prediction : array, shape = [n_samples, n_dim]
            Prediction of meta estimator that combines
            predictions of base estimators. `n_dim` depends
            on the return value of meta estimator's `predict`
            method.
        """
        X = numpy.asarray(X)
        Xt = self._predict_estimators(X)
        return self.meta_estimator.predict_proba(Xt)

    @if_delegate_has_method(delegate='meta_estimator')
    def predict_log_proba(self, X):
        """Perform prediction.

        Only available of the meta estimator has a predict_log_proba method.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Data with samples to predict.

        Returns
        -------
        prediction : array, shape = [n_samples, n_dim]
            Prediction of meta estimator that combines
            predictions of base estimators. `n_dim` depends
            on the return value of meta estimator's `predict`
            method.
        """
        X = numpy.asarray(X)
        Xt = self._predict_estimators(X)
        return self.meta_estimator.predict_log_proba(Xt)
