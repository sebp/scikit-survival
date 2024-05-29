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
import pandas as pd
from pandas.api.types import CategoricalDtype, is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._clinical_kernel import (
    continuous_ordinal_kernel,
    continuous_ordinal_kernel_with_ranges,
    pairwise_continuous_ordinal_kernel,
    pairwise_nominal_kernel,
)

__all__ = ["clinical_kernel", "ClinicalKernelTransform"]


def _nominal_kernel(x, y, out):
    """Number of features that match exactly"""
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            out[i, j] += (x[i, :] == y[j, :]).sum()

    return out


def _get_continuous_and_ordinal_array(x):
    """Convert array from continuous and ordered categorical columns"""
    nominal_columns = x.select_dtypes(include=["object", "category"]).columns
    ordinal_columns = pd.Index([v for v in nominal_columns if x[v].cat.ordered])
    continuous_columns = x.select_dtypes(include=[np.number]).columns

    x_num = x.loc[:, continuous_columns].astype(np.float64).values
    if len(ordinal_columns) > 0:
        x = _ordinal_as_numeric(x, ordinal_columns)

        nominal_columns = nominal_columns.difference(ordinal_columns)
        x_out = np.column_stack((x_num, x))
    else:
        x_out = x_num

    return x_out, nominal_columns


def _ordinal_as_numeric(x, ordinal_columns):
    x_numeric = np.empty((x.shape[0], len(ordinal_columns)), dtype=np.float64)

    for i, c in enumerate(ordinal_columns):
        x_numeric[:, i] = x[c].cat.codes
    return x_numeric


def clinical_kernel(x, y=None):
    """Computes clinical kernel

    The clinical kernel distinguishes between continuous
    ordinal,and nominal variables.

    See [1]_ for further description.

    Parameters
    ----------
    x : pandas.DataFrame, shape = (n_samples_x, n_features)
        Training data

    y : pandas.DataFrame, shape = (n_samples_y, n_features)
        Testing data

    Returns
    -------
    kernel : array, shape = (n_samples_x, n_samples_y)
        Kernel matrix. Values are normalized to lie within [0, 1].

    References
    ----------
    .. [1] Daemen, A., De Moor, B.,
           "Development of a kernel function for clinical data".
           Annual International Conference of the IEEE Engineering in Medicine and Biology Society, 5913-7, 2009
    """
    if y is not None:
        if x.shape[1] != y.shape[1]:
            raise ValueError("x and y have different number of features")
        if not x.columns.equals(y.columns):
            raise ValueError("columns do not match")
    else:
        y = x

    mat = np.zeros((x.shape[0], y.shape[0]), dtype=float)

    x_numeric, nominal_columns = _get_continuous_and_ordinal_array(x)
    if id(x) != id(y):
        y_numeric, _ = _get_continuous_and_ordinal_array(y)
    else:
        y_numeric = x_numeric

    continuous_ordinal_kernel(x_numeric, y_numeric, mat)
    _nominal_kernel(x.loc[:, nominal_columns].values, y.loc[:, nominal_columns].values, mat)
    mat /= x.shape[1]
    return mat


class ClinicalKernelTransform(BaseEstimator, TransformerMixin):
    """Transform data using a clinical Kernel

    The clinical kernel distinguishes between continuous
    ordinal,and nominal variables.

    See [1]_ for further description.

    Parameters
    ----------
    fit_once : bool, optional
        If set to ``True``, fit() does only transform the training data, but not update
        its internal state. You should call prepare() once before calling transform().
        If set to ``False``, it behaves like a regular estimator, i.e., you need to
        call fit() before transform().

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during ``fit``.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during ``fit``. Defined only when `X`
        has feature names that are all strings.

    References
    ----------
    .. [1] Daemen, A., De Moor, B.,
           "Development of a kernel function for clinical data".
           Annual International Conference of the IEEE Engineering in Medicine and Biology Society, 5913-7, 2009
    """

    def __init__(self, *, fit_once=False, _numeric_ranges=None, _numeric_columns=None, _nominal_columns=None):
        self.fit_once = fit_once

        self._numeric_ranges = _numeric_ranges
        self._numeric_columns = _numeric_columns
        self._nominal_columns = _nominal_columns

    def prepare(self, X):
        """Determine transformation parameters from data in X.

        Use if `fit_once` is `True`, in which case `fit()` does
        not set the parameters of the clinical kernel.

        Parameters
        ----------
        X: pandas.DataFrame, shape = (n_samples, n_features)
            Data to estimate parameters from.
        """
        if not self.fit_once:
            raise ValueError("prepare can only be used if fit_once parameter is set to True")

        self._prepare_by_column_dtype(X)

    def _prepare_by_column_dtype(self, X):
        """Get distance functions for each column's dtype"""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        numeric_columns = []
        nominal_columns = []
        numeric_ranges = []

        fit_data = np.empty(X.shape, dtype=np.float64)

        for i, dt in enumerate(X.dtypes):
            col = X.iloc[:, i]
            if isinstance(dt, CategoricalDtype):
                if col.cat.ordered:
                    numeric_ranges.append(col.cat.codes.max() - col.cat.codes.min())
                    numeric_columns.append(i)
                else:
                    nominal_columns.append(i)

                col = col.cat.codes
            elif is_numeric_dtype(dt):
                numeric_ranges.append(col.max() - col.min())
                numeric_columns.append(i)
            else:
                raise TypeError(f"unsupported dtype: {dt!r}")

            fit_data[:, i] = col.values

        self._numeric_columns = np.asarray(numeric_columns)
        self._nominal_columns = np.asarray(nominal_columns)
        self._numeric_ranges = np.asarray(numeric_ranges, dtype=float)
        self.X_fit_ = fit_data

    def fit(self, X, y=None, **kwargs):  # pylint: disable=unused-argument
        """Determine transformation parameters from data in X.

        Subsequent calls to `transform(Y)` compute the pairwise
        distance to `X`.
        Parameters of the clinical kernel are only updated
        if `fit_once` is `False`, otherwise you have to
        explicitly call `prepare()` once.

        Parameters
        ----------
        X: pandas.DataFrame, shape = (n_samples, n_features)
            Data to estimate parameters from.

        y : None
            Argument is ignored (included for compatibility reasons).

        kwargs : dict
            Argument is ignored (included for compatibility reasons).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if X.ndim != 2:
            raise ValueError(f"expected 2d array, but got {X.ndim}")

        self._check_feature_names(X, reset=True)
        self._check_n_features(X, reset=True)

        if self.fit_once:
            self.X_fit_ = X
        else:
            self._prepare_by_column_dtype(X)

        return self

    def transform(self, Y):
        r"""Compute all pairwise distances between `self.X_fit_` and `Y`.

        Parameters
        ----------
        Y : array-like, shape = (n_samples_y, n_features)

        Returns
        -------
        kernel : ndarray, shape = (n_samples_y, n_samples_X_fit\_)
            Kernel matrix. Values are normalized to lie within [0, 1].
        """
        check_is_fitted(self, "X_fit_")

        self._check_feature_names(Y, reset=False)
        self._check_n_features(Y, reset=False)

        n_samples_x = self.X_fit_.shape[0]

        Y = np.asarray(Y)

        n_samples_y = Y.shape[0]

        mat = np.zeros((n_samples_y, n_samples_x), dtype=float)

        continuous_ordinal_kernel_with_ranges(
            Y[:, self._numeric_columns].astype(np.float64),
            self.X_fit_[:, self._numeric_columns].astype(np.float64),
            self._numeric_ranges,
            mat,
        )

        if len(self._nominal_columns) > 0:
            _nominal_kernel(Y[:, self._nominal_columns], self.X_fit_[:, self._nominal_columns], mat)

        mat /= self.n_features_in_

        return mat

    def __call__(self, X, Y):
        """Compute Kernel matrix between `X` and `Y`.

        Parameters
        ----------
        x : array-like, shape = (n_samples_x, n_features)
            Training data

        y : array-like, shape = (n_samples_y, n_features)
            Testing data

        Returns
        -------
        kernel : ndarray, shape = (n_samples_x, n_samples_y)
            Kernel matrix. Values are normalized to lie within [0, 1].
        """
        return self.fit(X).transform(Y).T

    def pairwise_kernel(self, X, Y):
        """Function to use with :func:`sklearn.metrics.pairwise.pairwise_kernels`

        Parameters
        ----------
        X : array, shape = (n_features,)

        Y : array, shape = (n_features,)

        Returns
        -------
        similarity : float
            Similarities are normalized to be within [0, 1]
        """
        check_is_fitted(self, "X_fit_")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"Incompatible dimension for X and Y matrices: X.shape[0] == {X.shape[0]} "
                f"while Y.shape[0] == {Y.shape[0]}"
            )

        val = pairwise_continuous_ordinal_kernel(
            X[self._numeric_columns], Y[self._numeric_columns], self._numeric_ranges
        )
        if len(self._nominal_columns) > 0:
            val += pairwise_nominal_kernel(
                X[self._nominal_columns].astype(np.int8), Y[self._nominal_columns].astype(np.int8)
            )

        val /= X.shape[0]

        return val
