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
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import _check_feature_names_in, check_is_fitted, validate_data

from ._dataframe import (
    ensure_eager_dataframe,
    expand_dataframe_with_one_hot_columns,
    infer_column_semantics,
    is_categorical_or_string_dtype,
)
from .column import encode_categorical

__all__ = ["OneHotEncoder", "encode_categorical"]


def check_columns_exist(actual, expected):
    """Check if all expected columns are present in a dataframe.

    Parameters
    ----------
    actual : pandas.Index
        The actual columns of a dataframe.
    expected : pandas.Index
        The expected columns.

    Raises
    ------
    ValueError
        If any of the expected columns are missing from the actual columns.
    """
    missing_features = expected.difference(actual)
    if len(missing_features) != 0:
        raise ValueError(f"{len(missing_features)} features are missing from data: {missing_features.tolist()}")


class OneHotEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features using a one-hot scheme.

    Accepts :class:`pandas.DataFrame` and :class:`polars.DataFrame` inputs.
    The following column dtypes are treated as categorical features:

    - pandas: ``category`` or ``object``
    - polars: :class:`polars.Categorical`, :class:`polars.Enum`, or
      :class:`polars.String`

    The features are encoded using a one-hot (or dummy) encoding scheme, which
    creates a binary column for each category. By default, one category per feature
    is dropped: a column with ``M`` categories is encoded as ``M - 1`` integer columns
    according to the one-hot scheme.

    The order of non-categorical columns is preserved. Encoded columns are inserted
    in place of the original column. The output dataframe library matches the input.

    ``fit`` and ``transform`` must be called with the same dataframe library.
    Passing a pandas input to one and a polars input to the other raises
    :class:`TypeError`.

    Parameters
    ----------
    allow_drop : bool, optional, default: True
        Whether to allow dropping categorical columns that only consist
        of a single category.

    Attributes
    ----------
    feature_names_ : pandas.Index
        Names of categorical features that were encoded.

    categories_ : dict
        A dictionary mapping each categorical feature name to a
        :class:`pandas.Index` of categories.

    encoded_columns_ : pandas.Index
        The full list of feature names in the transformed output.

    n_features_in_ : int
        Number of features seen during ``fit``.

    feature_names_in_ : ndarray, shape = (`n_features_in_`,)
        Names of features seen during ``fit``. Defined only when `X`
        has feature names that are all strings.
    """

    def __init__(self, *, allow_drop=True):
        self.allow_drop = allow_drop

    def fit(self, X, y=None):  # pylint: disable=unused-argument
        """Determine which features are categorical and should be one-hot encoded.

        Parameters
        ----------
        X : pandas.DataFrame or polars.DataFrame
            The data to determine categorical features from.
        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`sklearn.pipeline.Pipeline`.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.fit_transform(X)
        return self

    def fit_transform(self, X, y=None, **fit_params):  # pylint: disable=unused-argument
        """Fit to data, then transform it.

        Fits the transformer to ``X`` by identifying categorical features and
        then returns a transformed version of ``X`` with categorical features
        one-hot encoded.

        Parameters
        ----------
        X : pandas.DataFrame or polars.DataFrame
            The data to fit and transform.
        y : None, optional
            Ignored. This parameter exists only for compatibility with
            :class:`sklearn.pipeline.Pipeline`.
        fit_params : dict, optional
            Ignored. This parameter exists only for compatibility with
            :class:`sklearn.pipeline.Pipeline`.

        Returns
        -------
        Xt : pandas.DataFrame or polars.DataFrame
            The transformed data. The output dataframe library matches the input.
        """
        X = ensure_eager_dataframe(X)
        validate_data(self, X, skip_check_array=True)

        self._fit_implementation_ = nw.from_native(X).implementation
        return self._fit_transform_dataframe(X)

    def transform(self, X):
        """Transform ``X`` by one-hot encoding categorical features.

        Parameters
        ----------
        X : pandas.DataFrame or polars.DataFrame
            The data to transform.

        Returns
        -------
        Xt : pandas.DataFrame or polars.DataFrame
            The transformed data. The output dataframe library matches the input.
        """
        check_is_fitted(self, "encoded_columns_")
        X = ensure_eager_dataframe(X)
        nw_X = nw.from_native(X)
        check_columns_exist(pd.Index(nw_X.columns), pd.Index(self.feature_names_in_))
        validation_X = nw_X.select(list(self.feature_names_in_)).to_native()
        validate_data(self, validation_X, reset=False, skip_check_array=True)

        if nw.from_native(X).implementation != self._fit_implementation_:
            raise TypeError("fit and transform must use the same dataframe library")

        return self._transform_dataframe(X)

    def _fit_transform_dataframe(self, X):
        nw_X = nw.from_native(X)
        implementation = nw_X.implementation

        columns_to_encode_list = [name for name, dtype in nw_X.schema.items() if is_categorical_or_string_dtype(dtype)]

        self._categorical_semantics_ = {
            name: infer_column_semantics(nw_X.get_column(name)) for name in columns_to_encode_list
        }

        columns_to_encode = {name: (nw_X.get_column(name), sem) for name, sem in self._categorical_semantics_.items()}
        x_dummy = expand_dataframe_with_one_hot_columns(
            nw_X,
            columns_to_encode=columns_to_encode,
            allow_drop=self.allow_drop,
            implementation=implementation,
            on_empty="raise",
        )

        self.feature_names_ = pd.Index(columns_to_encode_list)
        self.categories_ = {
            name: pd.Index(self._categorical_semantics_[name].categories or ()) for name in columns_to_encode_list
        }
        self.encoded_columns_ = pd.Index(x_dummy.columns)
        return x_dummy

    def _transform_dataframe(self, X):
        nw_X = nw.from_native(X)
        implementation = nw_X.implementation

        check_columns_exist(pd.Index(nw_X.columns), self.feature_names_)

        columns_to_encode = {name: (nw_X.get_column(name), sem) for name, sem in self._categorical_semantics_.items()}
        result = expand_dataframe_with_one_hot_columns(
            nw_X,
            columns_to_encode=columns_to_encode,
            allow_drop=self.allow_drop,
            implementation=implementation,
            on_empty="raise",
        )
        return nw.from_native(result).select(list(self.encoded_columns_)).to_native()

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default: None
            Input features.

            - If `input_features` is `None`, then `feature_names_in_` is
              used as feature names in.
            - If `input_features` is an array-like, then `input_features` must
              match `feature_names_in_` if `feature_names_in_` is defined.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        check_is_fitted(self, "encoded_columns_")
        input_features = _check_feature_names_in(self, input_features)

        return self.encoded_columns_.to_numpy(copy=True)
