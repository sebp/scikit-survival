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

from typing import NamedTuple

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from .._dataframe import (
    ColumnSemantics,
    column_to_category_codes,
    infer_column_semantics,
    is_narwhals_dataframe,
    to_narwhals_dataframe,
)


class _PreparedClinicalColumn(NamedTuple):
    values: np.ndarray
    is_numeric_kernel_input: bool
    numeric_range: float | None
    semantics_entry: tuple[str, ColumnSemantics | None]


def _normalize_ordinal_columns(ordinal_columns):
    if ordinal_columns is None:
        return frozenset()
    try:
        names = list(ordinal_columns)
    except TypeError as exc:
        raise TypeError(
            f"ordinal_columns must be an iterable of column names, got {type(ordinal_columns).__name__}"
        ) from exc
    for n in names:
        if not isinstance(n, str):
            raise TypeError(f"ordinal_columns entries must be strings, got {type(n).__name__}: {n!r}")
    return frozenset(names)


def _validate_ordinal_columns_against_schema(ordinal_set, schema):
    unknown = ordinal_set - set(schema)
    if unknown:
        raise ValueError(f"ordinal_columns contains unknown column names: {sorted(unknown)!r}")
    for name in ordinal_set:
        if not isinstance(schema[name], nw.Enum):
            raise ValueError(
                f"ordinal_columns={name!r} requires a categorical dtype with declared category order; "
                f"got {schema[name]!r}"
            )


def _promote_to_ordinal_semantics(semantics):
    return ColumnSemantics(
        name=semantics.name,
        kind="ordinal",
        categories=semantics.categories,
        ordered=True,
    )


def _extract_numeric_kernel_array(x, ordinal_columns=None):
    if is_narwhals_dataframe(x):
        return _extract_numeric_kernel_array_dataframe(x, ordinal_columns=ordinal_columns)
    # pandas 4 deprecates the implicit ``str``-via-``object`` inclusion here,
    # but explicitly listing ``"str"`` raises ``TypeError`` on pandas <= 3.x.
    # Keep the cross-version-safe form and silence the deprecation via
    # the ``filterwarnings`` entry in ``pyproject.toml``.
    nominal_columns = x.select_dtypes(include=["object", "category"]).columns
    pd_ordinal_columns = pd.Index(
        [v for v in nominal_columns if isinstance(x[v].dtype, CategoricalDtype) and x[v].cat.ordered]
    )
    # Include "bool": numpy treats ``bool_`` as a sibling of ``number`` rather
    # than a subclass, so a plain ``include=[np.number]`` would silently drop
    # Boolean columns (see PR #590).
    continuous_columns = x.select_dtypes(include=[np.number, "bool"]).columns

    x_num = x.loc[:, continuous_columns].to_numpy(dtype=np.float64)
    if len(pd_ordinal_columns) > 0:
        x = _ordinal_as_numeric(x, pd_ordinal_columns)

        nominal_columns = nominal_columns.difference(pd_ordinal_columns)
        x_out = np.column_stack((x_num, x))
    else:
        x_out = x_num

    return x_out, nominal_columns


def _classify_kernel_column(col_name, dtype, col, ordinal_set):
    if dtype.is_numeric() or isinstance(dtype, nw.Boolean):
        return "continuous", col.to_numpy().astype(np.float64)
    if not isinstance(dtype, (nw.Categorical, nw.Enum, nw.String)):
        raise TypeError(f"unsupported dtype: {dtype!r}")
    semantics = infer_column_semantics(col)
    if col_name in ordinal_set:
        semantics = _promote_to_ordinal_semantics(semantics)
    if semantics.kind == "ordinal":
        codes = column_to_category_codes(col, semantics).astype(np.float64)
        return "ordinal", codes
    return "nominal", None


def _extract_numeric_kernel_array_dataframe(x, ordinal_columns=None):
    ordinal_set = _normalize_ordinal_columns(ordinal_columns)
    nw_x = to_narwhals_dataframe(x)

    schema = nw_x.schema
    _validate_ordinal_columns_against_schema(ordinal_set, schema)

    n_rows = nw_x.shape[0]
    continuous_arrays = []
    ordinal_arrays = []
    nominal_names = []

    for col_name, dtype in schema.items():
        col = nw_x.get_column(col_name)
        kind, payload = _classify_kernel_column(col_name, dtype, col, ordinal_set)
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


def _ordinal_as_numeric(x, ordinal_columns):
    x_numeric = np.empty((x.shape[0], len(ordinal_columns)), dtype=np.float64)

    for i, c in enumerate(ordinal_columns):
        x_numeric[:, i] = x[c].cat.codes
    return x_numeric


def _continuous_range(values):
    return (values.max() - values.min()) if values.size > 0 else np.nan


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


def _encode_dataframe_kernel_column(col_name, col, ordinal_set):
    dtype = col.dtype
    if dtype.is_numeric() or isinstance(dtype, nw.Boolean):
        arr = col.to_numpy().astype(np.float64)
        return _make_numeric_kernel_column(arr)
    if not isinstance(dtype, (nw.Categorical, nw.Enum, nw.String)):
        raise TypeError(f"unsupported dtype: {dtype!r}")
    semantics = infer_column_semantics(col)
    if col_name in ordinal_set:
        semantics = _promote_to_ordinal_semantics(semantics)
    codes = column_to_category_codes(col, semantics).astype(np.float64)
    null_mask = col.is_null().to_numpy()
    codes[null_mask] = np.nan
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


def _check_clinical_kernel_inputs(x, y, x_is_narwhals_dataframe):
    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y have different number of features")
    if x_is_narwhals_dataframe:
        if not is_narwhals_dataframe(y):
            raise TypeError("x and y must use the same dataframe library")
        if list(nw.from_native(x).columns) != list(nw.from_native(y).columns):
            raise ValueError("columns do not match")
    else:
        if is_narwhals_dataframe(y):
            raise TypeError("x and y must use the same dataframe library")
        if not x.columns.equals(y.columns):
            raise ValueError("columns do not match")


def _normalize_nominal_nulls_for_kernel(arr):
    if arr.dtype != object:
        return arr
    # Element-wise ``is None`` over an object array; ``== None`` matches only
    # Python ``None`` (not ``NaN``), which is exactly what we normalize here.
    mask = arr == None  # noqa: E711
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


def _extract_nominal_kernel_arrays(x, y, nominal_columns, x_is_narwhals_dataframe):
    if x_is_narwhals_dataframe:
        nw_x = to_narwhals_dataframe(x)
        nw_y = to_narwhals_dataframe(y)
        return _extract_nominal_dataframe_array(nw_x, nominal_columns), _extract_nominal_dataframe_array(
            nw_y, nominal_columns
        )
    return (
        _normalize_nominal_nulls_for_kernel(x.loc[:, nominal_columns].to_numpy()),
        _normalize_nominal_nulls_for_kernel(y.loc[:, nominal_columns].to_numpy()),
    )
