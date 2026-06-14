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
"""Polars input predicates for the dataframe boundary.

This module only decides whether a native object is a polars frame or
series. Dataframe-library-neutral normalization and processing belong in
``_input.py``, ``_categorical_semantics.py``, or ``_categorical_encoding.py``
after values enter Narwhals.
"""

import narwhals.stable.v2 as nw

__all__ = ["LIBRARY", "PolarsDataFrameLibrary"]


class PolarsDataFrameLibrary:
    name = "polars"
    dataframe_display_names = ("polars.DataFrame",)
    series_display_names = ("polars.Series",)
    dataframe_display_name = " or ".join(dataframe_display_names)
    series_display_name = " or ".join(series_display_names)

    @staticmethod
    def is_dataframe(obj):
        return nw.dependencies.is_polars_dataframe(obj)

    @staticmethod
    def is_series(obj):
        return nw.dependencies.is_polars_series(obj)

    @staticmethod
    def is_non_numeric_cast_error(exc):
        """Return True iff ``exc`` was raised by polars rejecting a
        non-numeric string ``cast`` (e.g. ``"foo".cast(Int64)``).

        Implemented as a class-name + module-name check instead of an
        ``isinstance`` so this module doesn't import ``polars`` eagerly
        and stays importable when polars isn't installed.
        """
        return exc.__class__.__name__ == "InvalidOperationError" and exc.__class__.__module__.startswith("polars.")

    @staticmethod
    def ordinal_categories(native):
        """No auto-detected ordinal columns for polars.

        polars ``Enum`` is an ordered dtype, but it is also polars' only
        dtype that can carry a declared closed category set, so nominal
        columns use it as well (e.g. ARFF nominal attributes, which load as
        unordered ``pandas.Categorical`` on the pandas side). Auto-detecting
        ``Enum`` as statistically ordinal would therefore make kernel
        semantics depend on the dataframe library for the same data.
        Instead, ordinal columns must be declared explicitly via
        ``ordinal_categories=``.
        """
        return {}


LIBRARY = PolarsDataFrameLibrary()
