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
import logging

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype, is_object_dtype

__all__ = ["categorical_to_numeric", "encode_categorical", "standardize"]


def _apply_along_column(array, func1d, **kwargs):
    if isinstance(array, pd.DataFrame):
        return array.apply(func1d, **kwargs)
    return np.apply_along_axis(func1d, 0, array, **kwargs)


def standardize_column(series_or_array, with_std=True):
    d = series_or_array.dtype
    if issubclass(d.type, np.number):
        output = series_or_array.astype(float)
        m = series_or_array.mean()
        output -= m

        if with_std:
            s = series_or_array.std(ddof=1)
            output /= s

        return output

    return series_or_array


def standardize(table, with_std=True):
    """Standardize numeric features by removing the mean and scaling to unit variance.

    This function performs Z-Normalization on each numeric column of the given
    table.

    If `table` is a :class:`pandas.DataFrame`, only numeric columns are modified;
    all other columns remain unchanged. If `table` is a :class:`numpy.ndarray`,
    it is only modified if it has a numeric dtype, in which case the returned
    array will have a floating-point dtype.

    Parameters
    ----------
    table : pandas.DataFrame or numpy.ndarray
        Data to standardize.
    with_std : bool, optional, default: True
        If ``False``, data is only centered (mean removed) and not scaled to
        unit variance.

    Returns
    -------
    normalized : pandas.DataFrame or numpy.ndarray
        The standardized data. The output type will be the same as the input type.
    """
    new_frame = _apply_along_column(table, standardize_column, with_std=with_std)

    return new_frame


def _encode_categorical_series(series, allow_drop=True):
    values = _get_dummies_1d(series, allow_drop=allow_drop)
    if values is None:
        return

    enc, levels = values
    if enc is None:
        return pd.Series(index=series.index, name=series.name, dtype=series.dtype)

    if not allow_drop and enc.shape[1] == 1:
        return series

    names = []
    for key in range(1, enc.shape[1]):
        names.append(f"{series.name}={levels[key]}")
    series = pd.DataFrame(enc[:, 1:], columns=names, index=series.index)

    return series


def encode_categorical(table, columns=None, **kwargs):
    """One-hot encode categorical features.

    This function creates a binary column for each category and, by default,
    drops one of the categories per feature: a column with `M` categories
    is encoded as `M-1` integer columns according to the one-hot
    scheme.

    Parameters
    ----------
    table : pandas.DataFrame or pandas.Series
        Data with categorical columns to encode.
    columns : list-like, optional, default: None
        Column names in the DataFrame to be encoded.
        If `columns` is `None`, all columns with `object` or `category`
        dtype will be converted. This parameter is ignored if `table` is a
        pandas.Series.
    allow_drop : bool, optional, default: True
        Whether to allow dropping categorical columns that only consist
        of a single category.

    Returns
    -------
    encoded : pandas.DataFrame
        The transformed data with categorical columns encoded as numeric.
        Numeric columns in the input table remain unchanged.
    """
    if isinstance(table, pd.Series):
        if not isinstance(table.dtype, CategoricalDtype) and not is_object_dtype(table.dtype):
            raise TypeError(f"series must be of categorical dtype, but was {table.dtype}")
        return _encode_categorical_series(table, **kwargs)

    def _is_categorical_or_object(series):
        return isinstance(series.dtype, CategoricalDtype) or is_object_dtype(series.dtype)

    if columns is None:
        # for columns containing categories
        columns_to_encode = {nam for nam, s in table.items() if _is_categorical_or_object(s)}
    else:
        columns_to_encode = set(columns)

    items = []
    for name, series in table.items():
        if name in columns_to_encode:
            series = _encode_categorical_series(series, **kwargs)
            if series is None:
                continue
        items.append(series)

    # concat columns of tables
    new_table = pd.concat(items, axis=1, copy=False)
    return new_table


def _get_dummies_1d(data, allow_drop=True):
    # Series avoids inconsistent NaN handling
    cat = pd.Categorical(data)
    levels = cat.categories
    number_of_cols = len(levels)

    # if all NaN or only one level
    if allow_drop and number_of_cols < 2:
        logging.getLogger(__package__).warning(
            f"dropped categorical variable {data.name!r}, because it has only {number_of_cols} values"
        )
        return
    if number_of_cols == 0:
        return None, levels

    dummy_mat = np.eye(number_of_cols).take(cat.codes, axis=0)

    # reset NaN GH4446
    dummy_mat[cat.codes == -1] = np.nan

    return dummy_mat, levels


def categorical_to_numeric(table):
    """Encode categorical features as integers.

    This function converts each category to a unique integer value.

    Parameters
    ----------
    table : pandas.DataFrame or pandas.Series
        Data with categorical columns to encode.

    Returns
    -------
    encoded : pandas.DataFrame or pandas.Series
        The transformed data with categorical columns encoded as integers.
        The output type will be the same as the input type.
    """

    def transform(column):
        if isinstance(column.dtype, CategoricalDtype):
            return column.cat.codes
        if is_object_dtype(column.dtype):
            try:
                nc = column.astype(np.int64)
            except ValueError:
                classes = column.dropna().unique()
                classes.sort(kind="mergesort")
                nc = column.map(dict(zip(classes, range(classes.shape[0]))))
            return nc
        if column.dtype == bool:
            return column.astype(np.int64)

        return column

    if isinstance(table, pd.Series):
        return pd.Series(transform(table), name=table.name, index=table.index)
    return table.apply(transform, axis=0, result_type="expand")
