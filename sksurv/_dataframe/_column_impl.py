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
    column_to_one_hot_matrix,
    expand_dataframe_with_one_hot_columns,
    get_one_hot_column_names,
)
from ._categorical_semantics import ColumnSemantics, infer_column_semantics
from ._input import get_dataframe_library, is_non_numeric_cast_error, to_narwhals_dataframe

__all__ = [
    "categorical_to_numeric_narwhals",
    "encode_categorical_narwhals",
    "standardize_narwhals_dataframe",
]


def standardize_narwhals_dataframe(table, with_std):
    nw_table = to_narwhals_dataframe(table)
    implementation = nw_table.implementation

    if nw_table.shape[1] == 0:
        return nw_table.to_native()

    output_frames = []
    for col_name in nw_table.columns:
        col = nw_table.get_column(col_name)
        if col.dtype.is_numeric():
            arr = col.to_numpy().astype(float)
            mean = arr.mean()
            arr = arr - mean
            if with_std:
                std = arr.std(ddof=1)
                arr = arr / std
            # Pass the numpy buffer through; polars wraps it zero-copy and
            # pandas-like backends accept it without intermediate Python lists.
            new_col = nw.new_series(col_name, arr, backend=implementation)
            output_frames.append(new_col.to_frame())
        else:
            output_frames.append(nw_table.select(col_name))

    return nw.concat(output_frames, how="horizontal").to_native()


def encode_categorical_narwhals(table, columns=None, allow_drop=True):
    library = get_dataframe_library(table, allow_series=True)
    if library is not None and get_dataframe_library(table) is None:
        nw_series = nw.from_native(table, series_only=True)
        dt = nw_series.dtype
        if not isinstance(dt, (nw.Categorical, nw.Enum, nw.String)):
            raise TypeError(f"series must be of categorical dtype, but was {dt}")
        return _encode_series_as_dataframe(
            nw_series,
            allow_drop=allow_drop,
            implementation=nw_series.implementation,
        )

    nw_table = to_narwhals_dataframe(table)
    implementation = nw_table.implementation

    if columns is None:
        columns_to_encode = {
            name for name, dtype in nw_table.schema.items() if isinstance(dtype, (nw.Categorical, nw.Enum, nw.String))
        }
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
    if isinstance(dtype, nw.Boolean):
        # Match pandas get_dummies(drop_first=True): False is the dropped baseline.
        col = col.replace_strict({True: "True", False: "False"}, default=None, return_dtype=nw.String)
        observed = sorted({"True", "False"} & set(col.drop_nulls().unique().to_list()))
        observed = sorted(observed, key=lambda v: v != "False")
        return col, ColumnSemantics(name=col.name, kind="nominal", categories=tuple(observed), ordered=False)
    if dtype.is_numeric():
        ordered_values = sorted(col.drop_nulls().unique().to_list())
        col = col.cast(nw.String)
        return col, ColumnSemantics(
            name=col.name,
            kind="nominal",
            categories=tuple(str(v) for v in ordered_values),
            ordered=False,
        )
    return col, infer_column_semantics(col)


def _encode_series_as_dataframe(nw_series, allow_drop, implementation):
    semantics = infer_column_semantics(nw_series)
    n_cat = len(semantics.categories or ())

    logger = logging.getLogger("sksurv")
    if allow_drop and n_cat < 2:
        logger.warning(f"dropped categorical variable {semantics.name!r}, because it has only {n_cat} values")
        return nw.from_dict({}, backend=implementation).to_native()
    if not allow_drop and n_cat <= 1:
        return nw_series.to_native()

    encoded = column_to_one_hot_matrix(nw_series, semantics, drop_first=True)
    new_names = get_one_hot_column_names(semantics, drop_first=True)
    # Single backend construction keeps polars on its zero-copy numpy path.
    return nw.from_numpy(encoded, schema=list(new_names), backend=implementation).to_native()


def categorical_to_numeric_narwhals(table):
    library = get_dataframe_library(table, allow_series=True)
    if library is not None and get_dataframe_library(table) is None:
        nw_series = nw.from_native(table, series_only=True)
        dt = nw_series.dtype
        if dt.is_numeric() and not isinstance(dt, nw.Boolean):
            return nw_series.to_native()
        codes = _encode_series_as_numeric_codes(nw_series)
        return nw.new_series(nw_series.name, codes, backend=nw_series.implementation).to_native()

    nw_table = to_narwhals_dataframe(table)

    if nw_table.shape[1] == 0:
        return nw_table.to_native()

    implementation = nw_table.implementation
    output_frames = []
    for col_name in nw_table.columns:
        col = nw_table.get_column(col_name)
        dt = col.dtype
        if isinstance(dt, (nw.Categorical, nw.Enum, nw.String)) or isinstance(dt, nw.Boolean):
            codes = _encode_series_as_numeric_codes(col)
            new_col = nw.new_series(col_name, codes, backend=implementation)
        elif dt.is_numeric():
            output_frames.append(nw_table.select(col_name))
            continue
        else:
            output_frames.append(nw_table.select(col_name))
            continue
        output_frames.append(new_col.to_frame())

    return nw.concat(output_frames, how="horizontal").to_native()


def _encode_series_as_numeric_codes(nw_series):
    dt = nw_series.dtype
    if isinstance(dt, nw.Boolean):
        return nw_series.cast(nw.Int64).to_numpy()
    if isinstance(dt, nw.String):
        try:
            return nw_series.cast(nw.Int64).to_numpy()
        except Exception as exc:
            if not is_non_numeric_cast_error(exc):
                raise
    semantics = infer_column_semantics(nw_series)
    if semantics.kind == "numeric":
        return nw_series.to_numpy()
    if semantics.kind == "nominal" and isinstance(dt, (nw.Categorical, nw.String)):
        categories = tuple(sorted(semantics.categories or ()))
    else:
        categories = semantics.categories or ()
    mapping = {value: idx for idx, value in enumerate(categories)}
    codes = nw_series.replace_strict(mapping, default=-1, return_dtype=nw.Int64).to_numpy()
    if isinstance(dt, nw.String):
        null_mask = nw_series.is_null().to_numpy()
        if np.any(null_mask):
            codes = codes.astype(float)
            codes[null_mask] = np.nan
    return codes
