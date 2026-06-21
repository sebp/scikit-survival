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
"""Semantics-based dataframe encoding helpers."""

import narwhals.stable.v2 as nw
import numpy as np

__all__ = [
    "capture_pandas_index",
    "column_to_category_codes",
    "column_to_one_hot_matrix",
    "detach_pandas_index",
    "expand_dataframe_with_one_hot_columns",
    "get_one_hot_column_names",
    "reattach_pandas_index",
]


def capture_pandas_index(nw_obj):
    """Return the pandas row index of a frame/series, or ``None`` for index-less backends.

    Narwhals rebuilds columns positionally, so values reconstructed from numpy
    lose the caller's pandas index. Capture it here and restore it afterwards
    with :func:`reattach_pandas_index` so the returned pandas frame keeps the
    caller's original row labels. No-op for index-less backends (e.g. polars),
    which yield ``None``.
    """
    if nw_obj.implementation.is_pandas_like():
        return nw_obj.to_native().index


def detach_pandas_index(nw_frame):
    """Detach a pandas frame's row index so rebuilt columns concat positionally.

    Narwhals concatenates pandas frames by index *label*, but columns rebuilt
    from numpy (one-hot blocks, standardized columns, integer codes) carry a
    fresh ``RangeIndex``. Mixing those with index-preserving passthrough columns
    makes a horizontal concat realign by label, which corrupts rows when the
    source index is non-default or non-unique. Resetting the source index up
    front forces positional alignment; restore it afterwards with
    :func:`reattach_pandas_index`. No-op for index-less backends (e.g. polars).

    Returns a ``(frame, original_index)`` pair; ``original_index`` is ``None``
    for non-pandas backends.
    """
    original_index = capture_pandas_index(nw_frame)
    if original_index is not None:
        return nw.from_native(nw_frame.to_native().reset_index(drop=True)), original_index
    return nw_frame, None


def reattach_pandas_index(result_native, original_index):
    """Restore an index captured by :func:`detach_pandas_index` (no-op if ``None``)."""
    if original_index is not None:
        result_native.index = original_index
    return result_native


def column_to_category_codes(column, semantics, unknown_value=-1):
    s = nw.from_native(column, series_only=True)
    if semantics.kind == "numeric":
        return s.to_numpy().astype(np.float64)

    categories = semantics.categories or ()
    mapping = {value: idx for idx, value in enumerate(categories)}
    codes = s.replace_strict(mapping, default=unknown_value, return_dtype=nw.Int64).to_numpy()
    return codes.astype(np.float64)


def get_one_hot_column_names(semantics, drop_first=True):
    categories = semantics.categories or ()
    if drop_first:
        categories = categories[1:]
    return [f"{semantics.name}={level}" for level in categories]


def column_to_one_hot_matrix(column, semantics, drop_first=True, unknown_value=-1):
    categories = semantics.categories or ()
    codes = column_to_category_codes(column, semantics, unknown_value=unknown_value)
    n = codes.shape[0]
    n_cat = len(categories)

    if n_cat == 0:
        return np.empty((n, 0), dtype=float)

    encoded = np.zeros((n, n_cat), dtype=float)
    valid = (codes >= 0) & (codes < n_cat)
    if np.any(valid):
        encoded[valid] = np.eye(n_cat, dtype=float)[codes[valid].astype(int)]
    encoded[~valid] = np.nan

    if drop_first:
        encoded = encoded[:, 1:]
    return encoded


def expand_dataframe_with_one_hot_columns(
    nw_X,
    *,
    columns_to_encode,
    allow_drop,
    implementation,
    on_empty="raise",
    logger=None,
):
    nw_X, original_index = detach_pandas_index(nw_X)
    output_frames = []
    for col_name in nw_X.columns:
        if col_name not in columns_to_encode:
            output_frames.append(nw_X.select(col_name))
            continue

        col, semantics = columns_to_encode[col_name]
        n_cat = len(semantics.categories or ())
        if allow_drop and n_cat < 2:
            if logger is not None:
                logger.warning(f"dropped categorical variable {col_name!r}, because it has only {n_cat} values")
            continue
        if not allow_drop and n_cat <= 1:
            output_frames.append(nw_X.select(col_name))
            continue

        encoded = column_to_one_hot_matrix(col, semantics, drop_first=True)
        new_names = get_one_hot_column_names(semantics, drop_first=True)
        encoded_frame = nw.from_numpy(encoded, schema=new_names, backend=implementation)
        output_frames.append(encoded_frame)

    if len(output_frames) == 0:
        if on_empty == "raise":
            raise ValueError("No objects to concatenate")
        if on_empty == "empty_frame":
            return nw.from_dict({}, backend=implementation).to_native()
        raise ValueError(f"on_empty must be 'raise' or 'empty_frame', got {on_empty!r}")

    result = nw.concat(output_frames, how="horizontal").to_native()
    return reattach_pandas_index(result, original_index)
