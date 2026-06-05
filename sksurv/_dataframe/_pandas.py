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
"""Pandas input predicates for the dataframe boundary.

This module decides whether a native object is a pandas frame or series and
holds the pandas-specific semantics that Narwhals cannot express (notably the
ordered-categorical flag). Dataframe-library-neutral normalization and
processing belong in ``_input.py``, ``_categorical_semantics.py``, or
``_categorical_encoding.py`` after values enter Narwhals.
"""

import narwhals.stable.v2 as nw

__all__ = ["LIBRARY", "PandasDataFrameLibrary"]


class PandasDataFrameLibrary:
    name = "pandas"
    dataframe_display_names = ("pandas.DataFrame",)
    series_display_names = ("pandas.Series",)
    dataframe_display_name = " or ".join(dataframe_display_names)
    series_display_name = " or ".join(series_display_names)

    @staticmethod
    def is_dataframe(obj):
        return nw.dependencies.is_pandas_dataframe(obj)

    @staticmethod
    def is_series(obj):
        return nw.dependencies.is_pandas_series(obj)

    @staticmethod
    def is_non_numeric_cast_error(exc):
        """Return True iff ``exc`` was raised by pandas rejecting a
        non-numeric string ``cast`` (e.g. casting ``"foo"`` to ``Int64``).

        pandas raises a builtin ``ValueError`` ("invalid literal for int()")
        in that case, unlike polars' ``InvalidOperationError``. Matched by
        type + message so this module does not import ``pandas`` eagerly.
        """
        return isinstance(exc, ValueError) and "invalid literal" in str(exc).lower()

    @staticmethod
    def ordinal_categories(native):
        """Ordered-categorical columns declared via pandas' ``ordered=True``.

        Returns a mapping of column name to its declared category order for
        every ordered ``pandas.Categorical`` column. This is the pandas-only
        signal that Narwhals does not carry; the clinical kernel uses it to
        auto-detect ordinal columns. Other libraries (e.g. polars) declare
        ordinal columns explicitly instead and return ``{}``.
        """
        from pandas.api.types import CategoricalDtype

        ordinal = {}
        for name, dtype in native.dtypes.items():
            if isinstance(dtype, CategoricalDtype) and dtype.ordered:
                ordinal[str(name)] = tuple(dtype.categories.tolist())
        return ordinal


LIBRARY = PandasDataFrameLibrary()
