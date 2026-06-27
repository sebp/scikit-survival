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
"""DataFrame preparation helpers for the clinical kernel.

This module is private to :mod:`sksurv.kernels.clinical`. It keeps
DataFrame/Narwhals dtype dispatch and categorical semantics separate from
the public clinical-kernel API and the low-level kernel computations.
"""

from collections.abc import Mapping
from typing import NamedTuple

import narwhals.stable.v2 as nw
import numpy as np

from .._dataframe import (
    ColumnSemantics,
    column_to_category_codes,
    get_dataframe_library,
    infer_column_semantics,
    is_categorical_or_string_dtype,
)


class _PreparedClinicalColumn(NamedTuple):
    values: np.ndarray
    is_numeric_kernel_input: bool
    numeric_range: float | None
    semantics_entry: tuple[str, ColumnSemantics | None]


def _normalize_ordinal_categories(ordinal_categories):
    """Validate the public ``ordinal_categories`` mapping into ``{name: (cats...)}``."""
    if ordinal_categories is None:
        return {}
    if not isinstance(ordinal_categories, Mapping):
        raise TypeError(
            "ordinal_categories must be a mapping of column name to ordered categories, "
            f"got {type(ordinal_categories).__name__}"
        )
    normalized = {}
    for name, cats in ordinal_categories.items():
        if not isinstance(name, str):
            raise TypeError(f"ordinal_categories keys must be strings, got {type(name).__name__}: {name!r}")
        try:
            cat_tuple = tuple(cats)
        except TypeError as exc:
            raise TypeError(f"ordinal_categories[{name!r}] must be an iterable of category labels") from exc
        if len(cat_tuple) == 0:
            raise ValueError(f"ordinal_categories[{name!r}] must list at least one category")
        if len(set(cat_tuple)) != len(cat_tuple):
            raise ValueError(f"ordinal_categories[{name!r}] has duplicate categories: {cat_tuple!r}")
        normalized[name] = cat_tuple
    return normalized


def _resolve_ordinal_categories(table, ordinal_categories):
    """Merge user-declared ordinal categories with backend auto-detected ones.

    Explicit ``ordinal_categories`` win. In addition, pandas ``Categorical``
    columns declared with ``ordered=True`` are auto-detected as ordinal using
    their declared order. Other backends (e.g. polars) contribute nothing
    automatically.
    """
    resolved = _normalize_ordinal_categories(ordinal_categories)
    library = get_dataframe_library(table)
    if library is not None:
        for name, cats in library.ordinal_categories(table).items():
            resolved.setdefault(name, tuple(cats))
    return resolved


def _validate_ordinal_categories_against_schema(ordinal_categories, schema):
    unknown = set(ordinal_categories) - set(schema)
    if unknown:
        raise ValueError(f"ordinal_categories contains unknown column names: {sorted(unknown)!r}")
    for name in ordinal_categories:
        dtype = schema[name]
        if not is_categorical_or_string_dtype(dtype):
            raise ValueError(
                f"ordinal_categories={name!r} requires a categorical, string, or object column; got {dtype!r}"
            )


def _classify_kernel_column(col_name, dtype, col, ordinal_categories):
    if dtype.is_numeric() or dtype.is_boolean():
        return "continuous", col.to_numpy().astype(np.float64)
    if not is_categorical_or_string_dtype(dtype):
        raise TypeError(f"unsupported dtype: {dtype!r}")
    if col_name in ordinal_categories:
        semantics = ColumnSemantics(
            name=col_name,
            kind="ordinal",
            categories=tuple(ordinal_categories[col_name]),
            ordered=True,
        )
        codes = _column_to_kernel_codes(col, semantics)
        return "ordinal", codes
    return "nominal", None


def _extract_numeric_kernel_array(x, ordinal_categories=None):
    resolved = _resolve_ordinal_categories(x, ordinal_categories)
    nw_x = nw.from_native(x)

    schema = nw_x.schema
    _validate_ordinal_categories_against_schema(resolved, schema)

    n_rows = nw_x.shape[0]
    continuous_arrays = []
    ordinal_arrays = []
    nominal_names = []

    for col_name, dtype in schema.items():
        col = nw_x.get_column(col_name)
        kind, payload = _classify_kernel_column(col_name, dtype, col, resolved)
        if kind == "continuous":
            continuous_arrays.append(payload)
        elif kind == "ordinal":
            ordinal_arrays.append(payload)
        else:
            nominal_names.append(col_name)

    if continuous_arrays:
        x_num = (
            np.column_stack(continuous_arrays) if len(continuous_arrays) > 1 else continuous_arrays[0].reshape(-1, 1)
        )
    else:
        x_num = np.empty((n_rows, 0), dtype=np.float64)

    if ordinal_arrays:
        ord_arr = np.column_stack(ordinal_arrays) if len(ordinal_arrays) > 1 else ordinal_arrays[0].reshape(-1, 1)
        x_out = np.column_stack((x_num, ord_arr))
    else:
        x_out = x_num

    return x_out, nominal_names


def _continuous_range(values):
    # Missing values must not poison the range; guard the all-NaN case to
    # avoid numpy's All-NaN warning while keeping the NaN result.
    if values.size == 0 or np.all(np.isnan(values)):
        return np.nan
    return np.nanmax(values) - np.nanmin(values)


def _ordinal_range(codes):
    if np.all(np.isnan(codes)):
        return 0.0
    return np.nanmax(codes) - np.nanmin(codes)


def _make_numeric_kernel_column(values):
    return _PreparedClinicalColumn(
        values=values,
        is_numeric_kernel_input=True,
        numeric_range=_continuous_range(values),
        semantics_entry=("numeric", None),
    )


def _make_categorical_kernel_column(semantics, codes):
    is_numeric = semantics.kind == "ordinal"
    return _PreparedClinicalColumn(
        values=codes,
        is_numeric_kernel_input=is_numeric,
        numeric_range=_ordinal_range(codes) if is_numeric else None,
        semantics_entry=(semantics.kind, semantics),
    )


def _column_to_kernel_codes(col, semantics):
    codes = column_to_category_codes(col, semantics).astype(np.float64)
    invalid_mask = codes < 0
    null_mask = col.is_null().to_numpy()
    missing_mask = invalid_mask | null_mask
    if np.any(missing_mask):
        codes[missing_mask] = np.nan
    return codes


def _encode_dataframe_kernel_column(col_name, col, ordinal_categories):
    dtype = col.dtype
    if dtype.is_numeric() or dtype.is_boolean():
        arr = col.to_numpy().astype(np.float64)
        return _make_numeric_kernel_column(arr)
    if not is_categorical_or_string_dtype(dtype):
        raise TypeError(f"unsupported dtype: {dtype!r}")
    if col_name in ordinal_categories:
        semantics = ColumnSemantics(
            name=col_name,
            kind="ordinal",
            categories=tuple(ordinal_categories[col_name]),
            ordered=True,
        )
    else:
        semantics = infer_column_semantics(col)
    codes = _column_to_kernel_codes(col, semantics)
    return _make_categorical_kernel_column(semantics, codes)


def _append_kernel_column(
    i,
    prepared,
    fit_data,
    numeric_columns,
    nominal_columns,
    numeric_ranges,
    per_col_semantics,
):
    fit_data[:, i] = prepared.values
    per_col_semantics.append(prepared.semantics_entry)
    if prepared.is_numeric_kernel_input:
        numeric_columns.append(i)
        numeric_ranges.append(prepared.numeric_range)
    else:
        nominal_columns.append(i)


def _check_clinical_kernel_inputs(x, y):
    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y have different number of features")
    if get_dataframe_library(x) is not get_dataframe_library(y):
        raise TypeError("x and y must use the same dataframe library")
    if list(nw.from_native(x).columns) != list(nw.from_native(y).columns):
        raise ValueError("columns do not match")


def _normalize_nominal_nulls_for_kernel(arr):
    if arr.dtype != object:
        return arr
    # Element-wise ``is None`` over an object array; ``== None`` matches only
    # Python ``None`` (not ``NaN``), which is exactly what we normalize here.
    mask = arr == None  # noqa: E711  # pylint: disable=singleton-comparison
    if not mask.any():
        return arr
    out = arr.copy()
    out[mask] = np.nan
    return out


def _extract_nominal_dataframe_array(nw_frame, nominal_columns):
    columns = list(nominal_columns)
    if not columns:
        return np.empty((nw_frame.shape[0], 0), dtype=object)
    arrays = [nw_frame.get_column(name).to_numpy() for name in columns]
    if len(arrays) == 1:
        arr = arrays[0].reshape(-1, 1)
    else:
        arr = np.column_stack(arrays)
    return _normalize_nominal_nulls_for_kernel(arr)


def _extract_nominal_kernel_arrays(x, y, nominal_columns):
    nw_x = nw.from_native(x)
    nw_y = nw.from_native(y)
    return (
        _extract_nominal_dataframe_array(nw_x, nominal_columns),
        _extract_nominal_dataframe_array(nw_y, nominal_columns),
    )
