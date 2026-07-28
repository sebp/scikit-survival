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
"""Helpers for tests that run against every supported dataframe library.

Test data is described once, as plain Python values, and a
:class:`PandasBackend` or :class:`PolarsBackend` instance builds the native
frame for its library, so each backend receives input constructed the same
way a user of that library would construct it.
"""

import pandas as pd
import pandas.testing as tm
import pytest

__all__ = [
    "COMPARISON_OUTPUT_TYPES",
    "CROSS_LIBRARY_PAIRS",
    "PANDAS_BACKEND",
    "POLARS_BACKEND",
    "PandasBackend",
    "PolarsBackend",
]


class PandasBackend:
    """Builds and compares pandas dataframes in backend-parametrized tests."""

    #: this library's name in loader ``output_type`` arguments
    name = "pandas"
    dataframe_type = pd.DataFrame
    series_type = pd.Series
    #: exception raised when a requested column does not exist
    missing_column_error = KeyError

    @staticmethod
    def make_frame(data, categories=None):
        """Build a dataframe from column values.

        Parameters
        ----------
        data : dict
            Maps column names to sequences of values. ``None`` marks a
            missing value.
        categories : dict, optional
            Maps column names to their declared list of categories.
        """
        categories = categories or {}
        columns = {}
        for name, values in data.items():
            if name in categories:
                columns[name] = pd.Categorical(values, categories=categories[name])
            else:
                columns[name] = values
        return pd.DataFrame(columns)

    @staticmethod
    def make_series(name, values, categories=None):
        if categories is not None:
            return pd.Series(pd.Categorical(values, categories=categories), name=name)
        return pd.Series(values, name=name)

    @staticmethod
    def copy_frame(frame):
        return frame.copy()

    @staticmethod
    def assert_frame_equal(actual, expected, *, abs_tol=None):
        if abs_tol is None:
            tm.assert_frame_equal(actual, expected)
        else:
            tm.assert_frame_equal(actual, expected, check_exact=False, atol=abs_tol)


class PolarsBackend:
    """Builds and compares polars dataframes in backend-parametrized tests."""

    #: this library's name in loader ``output_type`` arguments
    name = "polars"

    @property
    def dataframe_type(self):
        import polars as pl

        return pl.DataFrame

    @property
    def series_type(self):
        import polars as pl

        return pl.Series

    @property
    def missing_column_error(self):
        from polars.exceptions import ColumnNotFoundError

        return ColumnNotFoundError

    @staticmethod
    def make_frame(data, categories=None):
        """Build a dataframe from column values.

        Columns with declared categories become ``pl.Enum``, the dtype that
        preserves a declared category order.
        """
        import polars as pl

        categories = categories or {}
        columns = []
        for name, values in data.items():
            declared = categories.get(name)
            if declared is not None:
                columns.append(pl.Series(name, values, dtype=pl.Enum(declared)))
            else:
                columns.append(pl.Series(name, values))
        return pl.DataFrame(columns)

    @staticmethod
    def make_series(name, values, categories=None):
        import polars as pl

        if categories is not None:
            return pl.Series(name, values, dtype=pl.Enum(categories))
        return pl.Series(name, values)

    @staticmethod
    def copy_frame(frame):
        return frame.clone()

    @staticmethod
    def assert_frame_equal(actual, expected, *, abs_tol=None):
        import polars.testing as pt

        if abs_tol is None:
            pt.assert_frame_equal(actual, expected)
        else:
            pt.assert_frame_equal(actual, expected, check_exact=False, abs_tol=abs_tol)


PANDAS_BACKEND = PandasBackend()
POLARS_BACKEND = PolarsBackend()

#: ordered pairs of distinct backends, with pandas on one side of each pair.
#: The library-mismatch checks compare the two libraries generically, so one
#: pair per direction and backend pins the contract without enumerating every
#: combination.
CROSS_LIBRARY_PAIRS = [
    pytest.param(PANDAS_BACKEND, POLARS_BACKEND, id="pandas-to-polars"),
    pytest.param(POLARS_BACKEND, PANDAS_BACKEND, id="polars-to-pandas"),
]

#: dataset-loader ``output_type`` names of the libraries compared against pandas
COMPARISON_OUTPUT_TYPES = ["polars"]
