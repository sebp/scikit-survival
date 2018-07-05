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
from distutils.version import LooseVersion
import logging

import numpy
import pandas

from pandas.api.types import is_categorical_dtype

_pandas_version_under0p23 = LooseVersion(pandas.__version__) < LooseVersion('0.23')


__all__ = ['categorical_to_numeric', 'encode_categorical', 'standardize']


def _apply_along_column(array, func1d, **kwargs):
    if isinstance(array, pandas.DataFrame):
        return array.apply(func1d, **kwargs)
    else:
        return numpy.apply_along_axis(func1d, 0, array, **kwargs)


def standardize_column(series_or_array, with_std=True):
    d = series_or_array.dtype
    if issubclass(d.type, numpy.number):
        m = series_or_array.mean()
        series_or_array -= m

        if with_std:
            s = series_or_array.std()
            series_or_array /= s

    return series_or_array


def standardize(table, with_std=True):
    """
    Perform Z-Normalization on each numeric column of the given table.

    Parameters
    ----------
    table : pandas.DataFrame or numpy.ndarray
        Data to standardize.

    with_std : bool, optional, default: True
        If ``False`` data is only centered and not converted to unit variance.

    Returns
    -------
    normalized : pandas.DataFrame
        Table with numeric columns normalized.
        Categorical columns in the input table remain unchanged.
    """
    if isinstance(table, pandas.DataFrame):
        cat_columns = table.select_dtypes(include=['category']).columns
    else:
        cat_columns = []

    new_frame = _apply_along_column(table, standardize_column, with_std=with_std)

    # work around for apply converting category dtype to object
    # https://github.com/pydata/pandas/issues/9573
    for col in cat_columns:
        new_frame[col] = table[col].copy()

    return new_frame


def _encode_categorical_series(series, allow_drop=True):
    values = _get_dummies_1d(series, allow_drop=allow_drop)
    if values is None:
        return

    enc, levels = values
    if enc is None:
        return pandas.Series(index=series.index, name=series.name, dtype=series.dtype)

    names = []
    for key in range(1, enc.shape[1]):
        names.append("{}={}".format(series.name, levels[key]))
    series = pandas.DataFrame(enc[:, 1:], columns=names, index=series.index)

    return series


def encode_categorical(table, columns=None, **kwargs):
    """
    Encode categorical columns with `M` categories into `M-1` columns according
    to the one-hot scheme.

    Parameters
    ----------
    table : pandas.DataFrame
        Table with categorical columns to encode.

    columns : list-like, optional, default: None
        Column names in the DataFrame to be encoded.
        If `columns` is None then all the columns with
        `object` or `category` dtype will be converted.

    allow_drop : boolean, optional, default: True
        Whether to allow dropping categorical columns that only consist
        of a single category.

    Returns
    -------
    encoded : pandas.DataFrame
        Table with categorical columns encoded as numeric.
        Numeric columns in the input table remain unchanged.
    """
    if isinstance(table, pandas.Series):
        if not is_categorical_dtype(table.dtype) and not table.dtype.char == "O":
            raise TypeError("series must be of categorical dtype, but was {}".format(table.dtype))
        return _encode_categorical_series(table, **kwargs)

    def _is_categorical_or_object(series):
        return is_categorical_dtype(series.dtype) or series.dtype.char == "O"

    if columns is None:
        # for columns containing categories
        columns_to_encode = {nam for nam, s in table.iteritems() if _is_categorical_or_object(s)}
    else:
        columns_to_encode = set(columns)

    items = []
    for name, series in table.iteritems():
        if name in columns_to_encode:
            series = _encode_categorical_series(series, **kwargs)
            if series is None:
                continue
        items.append(series)

    # concat columns of tables
    new_table = pandas.concat(items, axis=1, copy=False)
    return new_table


def _get_dummies_1d(data, allow_drop=True):
    # Series avoids inconsistent NaN handling
    cat = pandas.Categorical(data)
    levels = cat.categories
    number_of_cols = len(levels)

    # if all NaN or only one level
    if allow_drop and number_of_cols < 2:
        logging.getLogger(__package__).warning(
            "dropped categorical variable '%s', because it has only %d values", data.name, number_of_cols)
        return
    elif number_of_cols == 0:
        return None, levels

    dummy_mat = numpy.eye(number_of_cols).take(cat.codes, axis=0)

    # reset NaN GH4446
    dummy_mat[cat.codes == -1] = numpy.nan

    return dummy_mat, levels


def categorical_to_numeric(table):
    """Encode categorical columns to numeric by converting each category to
    an integer value.

    Parameters
    ----------
    table : pandas.DataFrame
        Table with categorical columns to encode.

    Returns
    -------
    encoded : pandas.DataFrame
        Table with categorical columns encoded as numeric.
        Numeric columns in the input table remain unchanged.
    """
    def transform(column):
        if is_categorical_dtype(column.dtype):
            return column.cat.codes
        if column.dtype.char == "O":
            try:
                nc = column.astype(numpy.int64)
            except ValueError:
                classes = column.dropna().unique()
                classes.sort(kind="mergesort")
                nc = column.replace(classes, numpy.arange(classes.shape[0]))
            return nc
        elif column.dtype == bool:
            return column.astype(numpy.int64)

        return column

    if isinstance(table, pandas.Series):
        return pandas.Series(transform(table), name=table.name, index=table.index)
    else:
        if _pandas_version_under0p23:
            return table.apply(transform, axis=0, reduce=False)
        else:
            return table.apply(transform, axis=0, result_type='reduce')
