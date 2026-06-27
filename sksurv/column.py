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

from ._dataframe import (
    ensure_eager_dataframe,
    is_supported_dataframe,
)
from ._dataframe._column_impl import (
    categorical_to_numeric_narwhals,
    encode_categorical_narwhals,
    standardize_narwhals_dataframe,
)

__all__ = ["categorical_to_numeric", "encode_categorical", "standardize"]


def standardize_column(array, with_std=True):
    d = array.dtype
    if issubclass(d.type, np.number):
        output = array.astype(float)
        m = array.mean()
        output -= m

        if with_std:
            s = array.std(ddof=1)
            output /= s

        return output

    return array


def standardize(table, with_std=True):
    """Standardize numeric features by removing the mean and scaling to unit variance.

    This function performs Z-Normalization on each numeric column of the given
    table.

    If `table` is a :class:`pandas.DataFrame` or :class:`polars.DataFrame`,
    only numeric columns are modified; all other columns remain unchanged.
    If `table` is a :class:`numpy.ndarray`, it is only modified if it has a numeric
    dtype, in which case the returned array will have a floating-point dtype.

    Parameters
    ----------
    table : pandas.DataFrame, polars.DataFrame, or numpy.ndarray
        Data to standardize.
    with_std : bool, optional, default: True
        If ``False``, data is only centered (mean removed) and not scaled to
        unit variance.

    Returns
    -------
    normalized : pandas.DataFrame, polars.DataFrame, or numpy.ndarray
        The standardized data. The output dataframe library matches the input.
    """
    table = ensure_eager_dataframe(table)
    if is_supported_dataframe(table):
        return standardize_narwhals_dataframe(table, with_std=with_std)
    return np.apply_along_axis(standardize_column, 0, table, with_std=with_std)


def encode_categorical(table, columns=None, **kwargs):
    """One-hot encode categorical features.

    This function creates a binary column for each category and, by default,
    drops one of the categories per feature: a column with `M` categories
    is encoded as `M-1` integer columns according to the one-hot
    scheme.

    Parameters
    ----------
    table : pandas.DataFrame, pandas.Series, polars.DataFrame, or polars.Series
        Data with categorical columns to encode.
    columns : list-like, optional, default: None
        Column names in the DataFrame to be encoded.
        If `columns` is `None`, all columns with `object` or `category`
        dtype will be converted. This parameter is ignored if `table` is a
        Series.
    allow_drop : bool, optional, default: True
        Whether to allow dropping categorical columns that only consist
        of a single category.

    Returns
    -------
    encoded : pandas.DataFrame, pandas.Series, polars.DataFrame, or polars.Series
        The transformed data with categorical columns encoded as numeric.
        Numeric columns in the input table remain unchanged. The output
        dataframe library matches the input.
    """
    table = ensure_eager_dataframe(table)
    return encode_categorical_narwhals(table, columns=columns, **kwargs)


def categorical_to_numeric(table):
    """Encode categorical features as integers.

    This function converts each category to a unique integer value.

    Parameters
    ----------
    table : pandas.DataFrame, pandas.Series, polars.DataFrame, or polars.Series
        Data with categorical columns to encode.

    Returns
    -------
    encoded : pandas.DataFrame, pandas.Series, or polars.DataFrame / polars.Series
        The transformed data with categorical columns encoded as integers.
        The output dataframe library matches the input.
    """
    table = ensure_eager_dataframe(table)
    return categorical_to_numeric_narwhals(table)
