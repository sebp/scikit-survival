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
"""Dataframe input normalization helpers."""

import narwhals.stable.v2 as nw
from sklearn.utils.validation import validate_data as _sklearn_validate_data

from . import _polars

__all__ = [
    "to_narwhals_dataframe",
    "collect_lazy_dataframe",
    "get_dataframe_library",
    "is_non_numeric_cast_error",
    "is_supported_dataframe",
    "is_supported_dataframe_or_series",
    "unsupported_dataframe_error",
    "is_narwhals_dataframe_or_series",
    "is_narwhals_dataframe",
    "validate_data_with_eager_dataframe",
]

EXTERNAL_DATAFRAME_LIBRARIES = (_polars.LIBRARY,)


def _oxford_join(items):
    items = list(items)
    if len(items) <= 2:
        return " or ".join(items)
    return ", ".join(items[:-1]) + ", or " + items[-1]


SUPPORTED_DATAFRAME_INPUT_TYPES = _oxford_join(
    ["pandas.DataFrame"] + [name for lib in EXTERNAL_DATAFRAME_LIBRARIES for name in lib.dataframe_display_names]
)


def get_dataframe_library(obj, *, allow_series=False):
    for library in EXTERNAL_DATAFRAME_LIBRARIES:
        if library.is_dataframe(obj) or (allow_series and library.is_series(obj)):
            return library
    return None


def is_supported_dataframe(obj):
    """Return whether ``obj`` is a dataframe type explicitly supported by sksurv."""
    return nw.dependencies.is_pandas_dataframe(obj) or get_dataframe_library(obj) is not None


def is_supported_dataframe_or_series(obj):
    return (
        is_supported_dataframe(obj)
        or nw.dependencies.is_pandas_series(obj)
        or get_dataframe_library(obj, allow_series=True) is not None
    )


def unsupported_dataframe_error(obj):
    return TypeError(f"expected {SUPPORTED_DATAFRAME_INPUT_TYPES}, but got {type(obj)!r}")


def is_non_numeric_cast_error(exc):
    """Dispatch backend-specific predicates for "string-to-numeric cast failed".

    Used by helpers that opportunistically cast a string column to a numeric
    dtype and need to know whether the resulting exception came from the
    column being non-numeric (intentional fallthrough) or from an unrelated
    failure (reraise).
    """
    return any(lib.is_non_numeric_cast_error(exc) for lib in EXTERNAL_DATAFRAME_LIBRARIES)


def is_narwhals_dataframe(obj):
    # Extend this predicate, not individual call sites, when another
    # Narwhals-backed dataframe library is supported.
    return get_dataframe_library(obj) is not None


def is_narwhals_dataframe_or_series(obj):
    return get_dataframe_library(obj, allow_series=True) is not None


def collect_lazy_dataframe(obj):
    library = get_dataframe_library(obj)
    if library is None:
        return obj
    return library.collect_lazy(obj)


def to_narwhals_dataframe(obj):
    nw_obj = nw.from_native(obj)
    if isinstance(nw_obj, nw.LazyFrame):
        nw_obj = nw_obj.collect()
    return nw_obj


def validate_data_with_eager_dataframe(estimator, X, *args, **kwargs):
    return _sklearn_validate_data(estimator, collect_lazy_dataframe(X), *args, **kwargs)
