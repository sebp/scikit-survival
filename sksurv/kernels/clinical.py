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
import narwhals.stable.v2 as nw
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from .._dataframe import (
    ensure_eager_dataframe,
    get_dataframe_library,
    is_supported_dataframe,
)
from ._clinical_dataframe import (
    _append_kernel_column,
    _check_clinical_kernel_inputs,
    _column_to_kernel_codes,
    _encode_dataframe_kernel_column,
    _extract_nominal_kernel_arrays,
    _extract_numeric_kernel_array,
    _resolve_ordinal_categories,
    _validate_ordinal_categories_against_schema,
)
from ._clinical_kernel import (
    continuous_ordinal_kernel,
    continuous_ordinal_kernel_with_ranges,
    pairwise_continuous_ordinal_kernel,
    pairwise_nominal_kernel,
)

__all__ = ["clinical_kernel", "ClinicalKernelTransform"]


def _nominal_kernel(x, y, out):
    """Number of features that match exactly."""
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            out[i, j] += (x[i, :] == y[j, :]).sum()

    return out


def clinical_kernel(x, y=None, *, ordinal_categories=None):
    """Computes clinical kernel.

    The clinical kernel distinguishes between continuous
    ordinal, and nominal variables.
    Kernel values are normalized to lie within [0, 1].

    See [1]_ for further description.

    Parameters
    ----------
    x : pandas.DataFrame or polars.DataFrame, shape = (n_samples_x, n_features)
        Training data. Polars and pandas inputs must not be mixed between
        ``x`` and ``y``.

    y : pandas.DataFrame or polars.DataFrame, shape = (n_samples_y, n_features)
        Testing data. Must use the same dataframe library as ``x``.

    ordinal_categories : mapping of str to sequence of labels, optional
        Columns to treat as ordinal, mapped to their category order, e.g.
        ``{"stage": ["I", "II", "III", "IV"]}``. Backend-independent. pandas
        ``Categorical(ordered=True)`` columns are additionally auto-detected.

    Returns
    -------
    kernel : array, shape = (n_samples_x, n_samples_y)
        Kernel matrix.

    References
    ----------
    .. [1] Daemen, A., De Moor, B.,
           "Development of a kernel function for clinical data".
           Annual International Conference of the IEEE Engineering in Medicine and Biology Society, 5913-7, 2009

    Examples
    --------
    Pandas input. Ordinal columns use the category order from
    ``pd.Categorical(ordered=True)``.

    >>> import pandas as pd
    >>> from sksurv.kernels import clinical_kernel
    >>>
    >>> data = pd.DataFrame({
    ...     'feature_num': [1.0, 2.0, 3.0],
    ...     'feature_ord': pd.Categorical(['low', 'medium', 'high'], ordered=True),
    ...     'feature_nom': pd.Categorical(['A', 'B', 'A'])
    ... })
    >>>
    >>> kernel_matrix = clinical_kernel(data)
    >>> print(kernel_matrix)
    [[1.         0.33333333 0.5       ]
     [0.33333333 1.         0.16666667]
     [0.5        0.16666667 1.        ]]

    """
    x = ensure_eager_dataframe(x)
    if y is not None:
        y = ensure_eager_dataframe(y)
        _check_clinical_kernel_inputs(x, y)
    else:
        y = x

    mat = np.zeros((x.shape[0], y.shape[0]), dtype=float)

    x_numeric, nominal_columns = _extract_numeric_kernel_array(x, ordinal_categories=ordinal_categories)
    if id(x) != id(y):
        y_numeric, _ = _extract_numeric_kernel_array(y, ordinal_categories=ordinal_categories)
    else:
        y_numeric = x_numeric

    continuous_ordinal_kernel(x_numeric, y_numeric, mat)
    x_nominal, y_nominal = _extract_nominal_kernel_arrays(x, y, nominal_columns)
    _nominal_kernel(x_nominal, y_nominal, mat)
    mat /= x.shape[1]
    return mat


class ClinicalKernelTransform(BaseEstimator, TransformerMixin):
    """Transform data using a clinical Kernel

    The clinical kernel distinguishes between continuous
    ordinal, and nominal variables.

    See [1]_ for further description.

    ``fit`` and ``transform`` must be called with the same dataframe library.
    Passing a pandas input to one and a polars input to the other raises
    :class:`TypeError`.

    Parameters
    ----------
    fit_once : bool, optional
        If set to ``True``, fit() does only transform the training data, but not update
        its internal state. You should call prepare() once before calling transform().
        In this mode, fit() expects the prepared numeric array and rejects a
        pandas or polars DataFrame with a :class:`TypeError`; call prepare(X)
        with the DataFrame first.
        If set to ``False``, it behaves like a regular estimator, i.e., you need to
        call fit() before transform().

    ordinal_categories : mapping of str to sequence of labels, optional
        Columns to treat as ordinal, mapped to their category order, e.g.
        ``{"stage": ["I", "II", "III", "IV"]}``. Backend-independent. pandas
        ``Categorical(ordered=True)`` columns are additionally auto-detected.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during ``fit``.

    feature_names_in_ : ndarray, shape = (`n_features_in_`,)
        Names of features seen during ``fit``. Defined only when `X`
        has feature names that are all strings.

    References
    ----------
    .. [1] Daemen, A., De Moor, B.,
           "Development of a kernel function for clinical data".
           Annual International Conference of the IEEE Engineering in Medicine and Biology Society, 5913-7, 2009
    """

    def __init__(
        self,
        *,
        fit_once=False,
        ordinal_categories=None,
        _numeric_ranges=None,
        _numeric_columns=None,
        _nominal_columns=None,
    ):
        self.fit_once = fit_once
        self.ordinal_categories = ordinal_categories

        self._numeric_ranges = _numeric_ranges
        self._numeric_columns = _numeric_columns
        self._nominal_columns = _nominal_columns

    def prepare(self, X):
        """Determine transformation parameters from data in X.

        Use if `fit_once` is `True`, in which case `fit()` does
        not set the parameters of the clinical kernel.

        Parameters
        ----------
        X: pandas.DataFrame or polars.DataFrame, shape = (n_samples, n_features)
            Data to estimate parameters from.
        """
        if not self.fit_once:
            raise ValueError("prepare can only be used if fit_once parameter is set to True")

        self._fit_kernel_columns(X)

    def _fit_kernel_columns(self, X):
        X = ensure_eager_dataframe(X)
        if get_dataframe_library(X) is None:
            raise TypeError("X must be a pandas DataFrame or supported Narwhals dataframe input")
        self._fit_implementation_ = nw.from_native(X).implementation
        return self._fit_dataframe_kernel_columns(X)

    def _fit_dataframe_kernel_columns(self, X):
        ordinal_categories = _resolve_ordinal_categories(X, self.ordinal_categories)
        nw_X = nw.from_native(X)

        schema = nw_X.schema
        _validate_ordinal_categories_against_schema(ordinal_categories, schema)

        n_samples, n_features = nw_X.shape
        fit_data = np.empty((n_samples, n_features), dtype=np.float64)

        numeric_columns = []
        nominal_columns = []
        numeric_ranges = []
        per_col_semantics = []

        for i, col_name in enumerate(nw_X.columns):
            col = nw_X.get_column(col_name)
            prepared = _encode_dataframe_kernel_column(
                col_name,
                col,
                ordinal_categories,
            )
            _append_kernel_column(
                i,
                prepared,
                fit_data,
                numeric_columns,
                nominal_columns,
                numeric_ranges,
                per_col_semantics,
            )

        self._numeric_columns = np.asarray(numeric_columns, dtype=int)
        self._nominal_columns = np.asarray(nominal_columns, dtype=int)
        self._numeric_ranges = np.asarray(numeric_ranges, dtype=float)
        self.X_fit_ = fit_data
        self._fitted_categorical_semantics = per_col_semantics

    def fit(self, X, y=None, **kwargs):  # pylint: disable=unused-argument
        """Determine transformation parameters from data in X.

        Subsequent calls to `transform(Y)` compute the pairwise
        distance to `X`.
        Parameters of the clinical kernel are only updated
        if `fit_once` is `False`, otherwise you have to
        explicitly call `prepare()` once.

        Parameters
        ----------
        X: pandas.DataFrame or polars.DataFrame, shape = (n_samples, n_features)
            Data to estimate parameters from.

        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`sklearn.pipeline.Pipeline`.

        kwargs : dict
            Ignored. This parameter exists only for compatibility with
            :class:`sklearn.pipeline.Pipeline`.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if is_supported_dataframe(X):
            ndim = 2  # supported dataframe inputs are always 2D
        else:
            ndim = getattr(X, "ndim", 2)
        if ndim != 2:
            raise ValueError(f"expected 2d array, but got {ndim}")

        X = ensure_eager_dataframe(X)
        validate_data(self, X, skip_check_array=True)

        if self.fit_once:
            if is_supported_dataframe(X):
                raise TypeError(
                    "fit_once=True expects a numeric array in fit(); call prepare(X) to set self.X_fit_, "
                    "then pass self.X_fit_ to fit()."
                )
            self.X_fit_ = X
        else:
            self._fit_kernel_columns(X)

        return self

    def _encode_dataframe_for_transform(self, Y):
        nw_Y = nw.from_native(Y)
        n_samples = nw_Y.shape[0]
        n_features = self.n_features_in_
        out = np.empty((n_samples, n_features), dtype=np.float64)
        for i, col_name in enumerate(nw_Y.columns):
            col = nw_Y.get_column(col_name)
            kind, semantics = self._fitted_categorical_semantics[i]
            col_dtype = col.dtype
            if col_dtype.is_numeric() or col_dtype.is_boolean() or kind == "numeric":
                out[:, i] = col.to_numpy().astype(np.float64)
            else:
                out[:, i] = _column_to_kernel_codes(col, semantics)
        return out

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

        Y = ensure_eager_dataframe(Y)

        validate_data(self, Y, reset=False, skip_check_array=True)

        y_is_dataframe = is_supported_dataframe(Y)
        fit_impl = getattr(self, "_fit_implementation_", None)
        if y_is_dataframe and fit_impl is not None and nw.from_native(Y).implementation != fit_impl:
            raise TypeError("fit and transform must use the same dataframe library")

        n_samples_x = self.X_fit_.shape[0]

        # Replay fit-time semantics if recorded; otherwise pass through
        # (fit_once / prepare paths, or pre-encoded numeric input).
        if hasattr(self, "_fitted_categorical_semantics") and y_is_dataframe:
            Y_arr = self._encode_dataframe_for_transform(Y)
        else:
            Y_arr = np.asarray(Y)

        n_samples_y = Y_arr.shape[0]

        mat = np.zeros((n_samples_y, n_samples_x), dtype=float)

        continuous_ordinal_kernel_with_ranges(
            Y_arr[:, self._numeric_columns].astype(np.float64),
            self.X_fit_[:, self._numeric_columns].astype(np.float64),
            self._numeric_ranges,
            mat,
        )

        if len(self._nominal_columns) > 0:
            _nominal_kernel(Y_arr[:, self._nominal_columns], self.X_fit_[:, self._nominal_columns], mat)

        mat /= self.n_features_in_

        return mat

    def __call__(self, X, Y):
        """Compute Kernel matrix between `X` and `Y`.

        Parameters
        ----------
        x : pandas.DataFrame or polars.DataFrame, shape = (n_samples_x, n_features)
            Training data.

        y : pandas.DataFrame or polars.DataFrame, shape = (n_samples_y, n_features)
            Testing data. Must use the same dataframe library as ``x``.

        Returns
        -------
        kernel : ndarray, shape = (n_samples_x, n_samples_y)
            Kernel matrix. Values are normalized to lie within [0, 1].
        """
        return self.fit(X).transform(Y).T

    def pairwise_kernel(self, X, Y):
        """Function to use with :func:`sklearn.metrics.pairwise.pairwise_kernels`.

        Parameters
        ----------
        X : ndarray, shape = (n_features,)

        Y : ndarray, shape = (n_features,)

        Returns
        -------
        similarity : float
            Similarities are normalized to be within [0, 1].
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
