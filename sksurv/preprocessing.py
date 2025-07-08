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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import _check_feature_names, _check_feature_names_in, _check_n_features, check_is_fitted

from .column import encode_categorical

__all__ = ["OneHotEncoder"]


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

    This transformer only works on pandas DataFrames. It identifies columns
    with `category` or `object` data type as categorical features.
    The features are encoded using a one-hot (or dummy) encoding scheme, which
    creates a binary column for each category. By default, one category per feature
    is dropped. a column with `M` categories is encoded as `M-1` integer columns
    according to the one-hot scheme.

    The order of non-categorical columns is preserved. Encoded columns are inserted
    in place of the original column.

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
        A dictionary mapping each categorical feature name to a list of its
        categories.

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
        X : pandas.DataFrame
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

    def _encode(self, X, columns_to_encode):
        return encode_categorical(X, columns=columns_to_encode, allow_drop=self.allow_drop)

    def fit_transform(self, X, y=None, **fit_params):  # pylint: disable=unused-argument
        """Fit to data, then transform it.

        Fits the transformer to ``X`` by identifying categorical features and
        then returns a transformed version of ``X`` with categorical features
        one-hot encoded.

        Parameters
        ----------
        X : pandas.DataFrame
            The data to fit and transform.
        y : None, optional
            Ignored. This parameter exists only for compatibility with
            :class:`sklearn.pipeline.Pipeline`.
        fit_params : dict, optional
            Ignored. This parameter exists only for compatibility with
            :class:`sklearn.pipeline.Pipeline`.

        Returns
        -------
        Xt : pandas.DataFrame
            The transformed data.
        """
        _check_feature_names(self, X, reset=True)
        _check_n_features(self, X, reset=True)
        columns_to_encode = X.select_dtypes(include=["object", "category"]).columns
        x_dummy = self._encode(X, columns_to_encode)

        self.feature_names_ = columns_to_encode
        self.categories_ = {k: X[k].cat.categories for k in columns_to_encode}
        self.encoded_columns_ = x_dummy.columns
        return x_dummy

    def transform(self, X):
        """Transform ``X`` by one-hot encoding categorical features.

        Parameters
        ----------
        X : pandas.DataFrame
            The data to transform.

        Returns
        -------
        Xt : pandas.DataFrame
            The transformed data.
        """
        check_is_fitted(self, "encoded_columns_")
        _check_n_features(self, X, reset=False)
        check_columns_exist(X.columns, self.feature_names_)

        Xt = X.copy()
        for col, cat in self.categories_.items():
            Xt[col] = Xt[col].cat.set_categories(cat)

        new_data = self._encode(Xt, self.feature_names_)
        return new_data.loc[:, self.encoded_columns_]

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

        return self.encoded_columns_.values.copy()
