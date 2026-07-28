"""Contract tests for the backends behind backend-parametrized tests.

The backend-parametrized tests build their inputs through these backends, so
the rules the backends define themselves are pinned here directly: how a
neutral column spec maps to each library's dtypes, and which exception a
missing column raises. A wrong mapping would otherwise surface only as
cascading failures downstream — or, if both backends drifted the same way,
not at all.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from sksurv.testing.dataframe import PANDAS_BACKEND, POLARS_BACKEND


class TestPandasBackend:
    @staticmethod
    def test_make_frame_plain_columns():
        df = PANDAS_BACKEND.make_frame({"a": [1.0, None], "b": np.array([1, 2]), "c": ["x", "y"]})

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["a", "b", "c"]
        assert df["a"].dtype == np.float64
        assert np.isnan(df["a"].iloc[1])
        assert df["b"].dtype == np.int64

    @staticmethod
    def test_make_frame_declared_categories():
        df = PANDAS_BACKEND.make_frame({"c": ["x", "x", None]}, categories={"c": ["x", "y"]})

        assert df["c"].dtype == pd.CategoricalDtype(["x", "y"], ordered=False)
        assert df["c"].cat.categories.tolist() == ["x", "y"]
        assert df["c"].isna().tolist() == [False, False, True]

    @staticmethod
    def test_make_series_declared_categories():
        s = PANDAS_BACKEND.make_series("c", ["x", "x", None], categories=["x", "y"])

        assert isinstance(s, pd.Series)
        assert s.name == "c"
        assert s.dtype == pd.CategoricalDtype(["x", "y"], ordered=False)
        assert s.isna().tolist() == [False, False, True]

    @staticmethod
    def test_missing_column_error_matches_lookup():
        df = PANDAS_BACKEND.make_frame({"a": [1]})
        with pytest.raises(PANDAS_BACKEND.missing_column_error):
            _ = df["no_such_column"]


class TestPolarsBackend:
    @staticmethod
    def test_make_frame_plain_columns():
        df = POLARS_BACKEND.make_frame({"a": [1.0, None], "b": np.array([1, 2]), "c": ["x", "y"]})

        assert isinstance(df, pl.DataFrame)
        assert df.columns == ["a", "b", "c"]
        assert df["a"].dtype == pl.Float64
        assert df["a"].null_count() == 1
        assert df["b"].dtype == pl.Int64

    @staticmethod
    def test_make_frame_declared_categories():
        df = POLARS_BACKEND.make_frame({"c": ["x", "x", None]}, categories={"c": ["x", "y"]})

        assert df["c"].dtype == pl.Enum(["x", "y"])
        assert df["c"].null_count() == 1

    @staticmethod
    def test_make_series_declared_categories():
        s = POLARS_BACKEND.make_series("c", ["x", "x", None], categories=["x", "y"])

        assert isinstance(s, pl.Series)
        assert s.name == "c"
        assert s.dtype == pl.Enum(["x", "y"])
        assert s.null_count() == 1

    @staticmethod
    def test_missing_column_error_matches_lookup():
        df = POLARS_BACKEND.make_frame({"a": [1]})
        with pytest.raises(POLARS_BACKEND.missing_column_error):
            _ = df["no_such_column"]
