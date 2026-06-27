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
"""Dataframe implementations for :mod:`sksurv.column` functions."""

import logging

import narwhals.stable.v2 as nw
import numpy as np

from ._categorical_encoding import (
    capture_pandas_index,
    column_to_one_hot_matrix,
    detach_pandas_index,
    expand_dataframe_with_one_hot_columns,
    get_one_hot_column_names,
    reattach_pandas_index,
)
from ._categorical_semantics import ColumnSemantics, infer_column_semantics, is_categorical_or_string_dtype
from ._input import is_non_numeric_cast_error, is_supported_series

__all__ = [
    "categorical_to_numeric_narwhals",
    "encode_categorical_narwhals",
    "standardize_narwhals_dataframe",
]


def standardize_narwhals_dataframe(table, with_std):
    nw_table = nw.from_native(table)

    exprs = []
    for col_name, dtype in nw_table.schema.items():
        if not dtype.is_numeric():
            continue
        col = nw.col(col_name)
        # Mean/std must skip missing values, but polars only skips null in
        # aggregations and propagates float NaN, so compute the statistics on a
        # NaN-to-null normalized expression. The subtraction uses the original
        # column, so a present row keeps its standardized value and a row that is
        # missing stays missing. A fully-missing column standardizes to null on
        # polars and NaN on pandas; both become NaN via to_numpy(), which is how
        # the data reaches the estimators.
        stats_col = col.fill_nan(None) if dtype.is_float() else col
        standardized = col - stats_col.mean()
        if with_std:
            standardized = standardized / stats_col.std(ddof=1)
        exprs.append(standardized)

    if len(exprs) == 0:
        return nw_table.to_native()
    return nw_table.with_columns(exprs).to_native()


def encode_categorical_narwhals(table, columns=None, allow_drop=True):
    if is_supported_series(table):
        nw_series = nw.from_native(table, series_only=True)
        dt = nw_series.dtype
        if not is_categorical_or_string_dtype(dt):
            raise TypeError(f"series must be of categorical dtype, but was {dt}")
        return _encode_series_as_dataframe(
            nw_series,
            allow_drop=allow_drop,
            implementation=nw_series.implementation,
        )

    nw_table = nw.from_native(table)
    implementation = nw_table.implementation

    if columns is None:
        columns_to_encode = {name for name, dtype in nw_table.schema.items() if is_categorical_or_string_dtype(dtype)}
    else:
        columns_to_encode = set(columns)

    columns_to_encode_with_semantics = {}
    for col_name in nw_table.columns:
        if col_name not in columns_to_encode:
            continue
        col = nw_table.get_column(col_name)
        col, sem = _prepare_column_for_one_hot(col)
        columns_to_encode_with_semantics[col_name] = (col, sem)

    return expand_dataframe_with_one_hot_columns(
        nw_table,
        columns_to_encode=columns_to_encode_with_semantics,
        allow_drop=allow_drop,
        implementation=implementation,
        on_empty="empty_frame",
        logger=logging.getLogger("sksurv"),
    )


def _prepare_column_for_one_hot(col):
    dtype = col.dtype
    if dtype.is_boolean():
        # Match pandas get_dummies(drop_first=True): False is the dropped baseline.
        present = set(col.drop_nulls().unique())
        observed = [value for value in (False, True) if value in present]
        return col, ColumnSemantics(name=col.name, kind="nominal", categories=tuple(observed), ordered=False)
    if dtype.is_numeric():
        ordered_values = sorted(col.drop_nulls().unique())
        return col, ColumnSemantics(
            name=col.name,
            kind="nominal",
            categories=tuple(ordered_values),
            ordered=False,
        )
    return col, infer_column_semantics(col)


def _encode_series_as_dataframe(nw_series, allow_drop, implementation):
    original_index = capture_pandas_index(nw_series)
    semantics = infer_column_semantics(nw_series)
    n_cat = len(semantics.categories or ())

    logger = logging.getLogger("sksurv")
    if allow_drop and n_cat < 2:
        logger.warning(f"dropped categorical variable {semantics.name!r}, because it has only {n_cat} values")
        result = nw.from_dict({}, backend=implementation).to_native()
        if original_index is not None:
            result = result.reindex(original_index)
        return result
    if not allow_drop and n_cat <= 1:
        return nw_series.to_native()

    encoded = column_to_one_hot_matrix(nw_series, semantics, drop_first=True)
    new_names = get_one_hot_column_names(semantics, drop_first=True)
    result = nw.from_numpy(encoded, schema=new_names, backend=implementation).to_native()
    return reattach_pandas_index(result, original_index)


def categorical_to_numeric_narwhals(table):
    if is_supported_series(table):
        nw_series = nw.from_native(table, series_only=True)
        dt = nw_series.dtype
        if dt.is_numeric() and not dt.is_boolean():
            return nw_series.to_native()
        original_index = capture_pandas_index(nw_series)
        codes = _encode_series_as_numeric_codes(nw_series)
        result = nw.new_series(nw_series.name, codes, backend=nw_series.implementation).to_native()
        return reattach_pandas_index(result, original_index)

    nw_table = nw.from_native(table)

    if nw_table.shape[1] == 0:
        return nw_table.to_native()

    implementation = nw_table.implementation
    nw_table, original_index = detach_pandas_index(nw_table)
    output_frames = []
    for col_name in nw_table.columns:
        col = nw_table.get_column(col_name)
        if is_categorical_or_string_dtype(col.dtype) or col.dtype.is_boolean():
            codes = _encode_series_as_numeric_codes(col)
            output_frames.append(nw.new_series(col_name, codes, backend=implementation).to_frame())
        else:
            # Numeric (and any other) columns pass through unchanged.
            output_frames.append(nw_table.select(col_name))

    result = nw.concat(output_frames, how="horizontal").to_native()
    return reattach_pandas_index(result, original_index)


def _int_codes_to_numpy(int_series, null_mask):
    """Materialize a possibly null-containing Int64 series as a numpy array.

    ``fill_null`` runs before ``to_numpy`` so nullable backends (e.g. pandas'
    Arrow-backed strings) build an integer array instead of failing to convert a
    null to int; null positions are then restored to ``NaN``.
    """
    codes = int_series.fill_null(-1).to_numpy()
    if np.any(null_mask):
        codes = codes.astype(float)
        codes[null_mask] = np.nan
    return codes


def _encode_series_as_numeric_codes(nw_series):
    dt = nw_series.dtype
    if dt.is_boolean():
        return nw_series.cast(nw.Int64).to_numpy()
    null_mask = nw_series.is_null().to_numpy()
    if isinstance(dt, (nw.String, nw.Object)) and not np.any(null_mask):
        # No nulls: a string/object column may actually hold integer labels
        # (e.g. "1", "2", "3"); encode those as the integers themselves. If the
        # strings are not numeric, the cast raises a backend-specific
        # "non-numeric cast" error; swallow only that one and fall through to
        # category codes. The int cast is skipped when nulls are present, because
        # casting a null to int fails on nullable backends (e.g. pandas
        # Arrow-backed strings); those take the null-aware path below.
        try:
            return nw_series.cast(nw.Int64).to_numpy()
        except Exception as exc:
            if not is_non_numeric_cast_error(exc):
                raise
    # Reached for categorical / enum / string / object columns: numeric native
    # dtypes are returned early by both callers and booleans are handled above,
    # so infer_column_semantics always reports a nominal column here.
    semantics = infer_column_semantics(nw_series)
    # Category order is centralized in get_semantic_categories (declared order
    # for pandas Categorical / Enum, sorted for polars Categorical / String),
    # so use semantics.categories as-is here rather than re-sorting.
    categories = semantics.categories or ()
    mapping = {value: idx for idx, value in enumerate(categories)}
    codes_series = nw_series.replace_strict(mapping, default=-1, return_dtype=nw.Int64)
    if isinstance(dt, nw.String):
        return _int_codes_to_numpy(codes_series, null_mask)
    return codes_series.to_numpy()
