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

from . import _pandas, _polars

__all__ = [
    "ensure_eager_dataframe",
    "get_dataframe_library",
    "is_non_numeric_cast_error",
    "is_supported_dataframe",
    "is_supported_dataframe_or_series",
    "is_supported_series",
    "unsupported_dataframe_error",
]

DATAFRAME_LIBRARIES = (_pandas.LIBRARY, _polars.LIBRARY)


def _oxford_join(items):
    items = list(items)
    if len(items) <= 2:
        return " or ".join(items)
    return ", ".join(items[:-1]) + ", or " + items[-1]  # pragma: no cover


SUPPORTED_DATAFRAME_INPUT_TYPES = _oxford_join(
    [name for lib in DATAFRAME_LIBRARIES for name in lib.dataframe_display_names]
)


def get_dataframe_library(obj, *, allow_series=False):
    for library in DATAFRAME_LIBRARIES:
        if library.is_dataframe(obj) or (allow_series and library.is_series(obj)):
            return library


def is_supported_dataframe(obj):
    """Return whether ``obj`` is a dataframe type explicitly supported by sksurv."""
    return get_dataframe_library(obj) is not None


def is_supported_dataframe_or_series(obj):
    return get_dataframe_library(obj, allow_series=True) is not None


def is_supported_series(obj):
    """Return whether ``obj`` is a supported native Series (and not a DataFrame)."""
    return is_supported_dataframe_or_series(obj) and not is_supported_dataframe(obj)


def unsupported_dataframe_error(obj):
    return TypeError(f"expected {SUPPORTED_DATAFRAME_INPUT_TYPES}, but got {type(obj)!r}")


def is_non_numeric_cast_error(exc):
    """Dispatch backend-specific predicates for "string-to-numeric cast failed".

    Used by helpers that opportunistically cast a string column to a numeric
    dtype and need to know whether the resulting exception came from the
    column being non-numeric (intentional fallthrough) or from an unrelated
    failure (reraise).
    """
    return any(lib.is_non_numeric_cast_error(exc) for lib in DATAFRAME_LIBRARIES)


_LAZYFRAME_NOT_SUPPORTED_MSG = "polars.LazyFrame is not supported; call .collect() before passing to scikit-survival."


def _reject_polars_lazyframe(obj):
    if nw.dependencies.is_polars_lazyframe(obj):
        raise TypeError(_LAZYFRAME_NOT_SUPPORTED_MSG)


def ensure_eager_dataframe(obj):
    """Reject a polars ``LazyFrame`` input; return any other object unchanged.

    scikit-survival does not support lazy frames. scikit-learn's input
    validation requires eager dataframes, so a ``LazyFrame`` would have to be
    collected before anything useful could happen. Rather than collecting
    implicitly, callers are asked to materialize the frame explicitly via
    ``.collect()``.
    """
    _reject_polars_lazyframe(obj)
    return obj
