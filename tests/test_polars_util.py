"""Tests for ``sksurv.util`` (Surv, safe_concat) with polars input."""

from contextlib import nullcontext as does_not_raise

import numpy as np
from numpy.testing import assert_array_equal
import polars as pl
from polars.exceptions import ColumnNotFoundError
import pytest

from sksurv.testing import FixtureParameterFactory
from sksurv.util import Surv, safe_concat


class SurvDataFramePolarsCases(FixtureParameterFactory):
    @property
    def event_and_time(self):
        rng = np.random.default_rng(0)
        event = rng.binomial(1, 0.5, size=100)
        time = np.exp(rng.standard_normal(100))
        return event, time

    def _expected(self, event, time, event_name="event", time_name="time"):
        expected = np.empty(dtype=[(str(event_name), bool), (str(time_name), float)], shape=event.shape[0])
        expected[str(event_name)] = event.astype(bool)
        expected[str(time_name)] = time
        return expected

    def _make_eager(self, event, time, event_name="event", time_name="time", event_dtype=None):
        data = {event_name: event, time_name: time}
        df = pl.DataFrame(data)
        if event_dtype is not None:
            df = df.with_columns(pl.col(event_name).cast(event_dtype))
        return df

    def data_polars_eager_bool(self):
        event, time = self.event_and_time
        df = self._make_eager(event.astype(bool), time)
        return ("event", "time", df), self._expected(event, time), does_not_raise()

    def data_polars_eager_int(self):
        event, time = self.event_and_time
        df = self._make_eager(event.astype(np.int64), time)
        return ("event", "time", df), self._expected(event, time), does_not_raise()

    def data_polars_eager_float(self):
        event, time = self.event_and_time
        df = self._make_eager(event.astype(float), time)
        return ("event", "time", df), self._expected(event, time), does_not_raise()

    def data_polars_eager_column_names(self):
        event, time = self.event_and_time
        df = self._make_eager(event.astype(bool), time, event_name="death", time_name="time_to_death")
        return (
            ("death", "time_to_death", df),
            self._expected(event, time, event_name="death", time_name="time_to_death"),
            does_not_raise(),
        )

    def data_polars_eager_no_such_column_event(self):
        event, time = self.event_and_time
        df = self._make_eager(event.astype(bool), time)
        err = pytest.raises(ColumnNotFoundError, match="unknown")
        return ("unknown", "time", df), None, err

    def data_polars_eager_no_such_column_time(self):
        event, time = self.event_and_time
        df = self._make_eager(event.astype(bool), time)
        err = pytest.raises(ColumnNotFoundError, match="unknown")
        return ("event", "unknown", df), None, err

    def data_polars_lazy_bool(self):
        event, time = self.event_and_time
        lf = self._make_eager(event.astype(bool), time).lazy()
        return ("event", "time", lf), self._expected(event, time), does_not_raise()

    def data_polars_lazy_int(self):
        event, time = self.event_and_time
        lf = self._make_eager(event.astype(np.int64), time).lazy()
        return ("event", "time", lf), self._expected(event, time), does_not_raise()

    def data_polars_lazy_float(self):
        event, time = self.event_and_time
        lf = self._make_eager(event.astype(float), time).lazy()
        return ("event", "time", lf), self._expected(event, time), does_not_raise()

    def data_polars_lazy_column_names(self):
        event, time = self.event_and_time
        lf = self._make_eager(event.astype(bool), time, event_name="death", time_name="time_to_death").lazy()
        return (
            ("death", "time_to_death", lf),
            self._expected(event, time, event_name="death", time_name="time_to_death"),
            does_not_raise(),
        )

    def data_polars_lazy_no_such_column_event(self):
        event, time = self.event_and_time
        lf = self._make_eager(event.astype(bool), time).lazy()
        err = pytest.raises(ColumnNotFoundError, match="unknown")
        return ("unknown", "time", lf), None, err

    def data_polars_lazy_no_such_column_time(self):
        event, time = self.event_and_time
        lf = self._make_eager(event.astype(bool), time).lazy()
        err = pytest.raises(ColumnNotFoundError, match="unknown")
        return ("event", "unknown", lf), None, err

    def data_polars_series_input(self):
        s = pl.Series("event", [True, False, True])
        err = pytest.raises(TypeError)
        return ("event", "time", s), None, err


@pytest.mark.parametrize("args,expected,expected_error", SurvDataFramePolarsCases().get_cases())
def test_from_dataframe_polars(args, expected, expected_error):
    with expected_error:
        y = Surv.from_dataframe(*args)

    if expected is not None:
        assert_array_equal(y, expected)


class _ConcatPolarsCases(FixtureParameterFactory):
    @property
    def rnd(self):
        return np.random.default_rng(14)

    def numeric_series(self, name):
        return pl.Series(name, self.rnd.standard_normal(100))

    def categorical_enum_series(self, name, num_cats):
        cat_names = [f"C{i + 1}" for i in range(num_cats)]
        codes = self.rnd.binomial(num_cats - 1, 0.6, 100)
        values = [cat_names[c] for c in codes]
        return pl.Series(name, values, dtype=pl.Enum(cat_names))


class ConcatPolarsAxes1Cases(_ConcatPolarsCases):
    def data_numeric(self):
        a = self.numeric_series("col_A")
        b = self.numeric_series("col_B")
        expected = pl.DataFrame({a.name: a, b.name: b})
        return (a, b), expected, does_not_raise()

    def data_numeric_categorical(self):
        a = self.numeric_series("col_A")
        b = self.categorical_enum_series("col_B", 5)
        expected = pl.DataFrame({a.name: a, b.name: b})
        return (a, b), expected, does_not_raise()

    def data_frame_numeric_categorical(self):
        numeric_df = pl.DataFrame(
            {
                "col_A": self.numeric_series("col_A"),
                "col_B": self.categorical_enum_series("col_B", 5),
            }
        )
        cat_series = self.categorical_enum_series("col_C", 5)
        expected = numeric_df.with_columns(cat_series)
        return (numeric_df, cat_series), expected, does_not_raise()

    def data_duplicate_columns(self):
        df1 = pl.DataFrame(
            {
                "col_N": self.numeric_series("col_N"),
                "col_B": self.categorical_enum_series("col_B", 5),
                "col_A": self.categorical_enum_series("col_A", 5),
            }
        )
        df2 = pl.DataFrame(
            {
                "col_A": self.categorical_enum_series("col_A", 5),
                "col_C": self.categorical_enum_series("col_C", 2),
            }
        )
        err = pytest.raises(ValueError, match="duplicate columns col_A")
        return (df1, df2), None, err


class ConcatPolarsAxes0Cases(_ConcatPolarsCases):
    def data_categorical_match(self):
        a = pl.DataFrame(
            {
                "col_A": self.categorical_enum_series("col_A", 3),
                "col_B": self.numeric_series("col_B"),
            }
        )
        b = pl.DataFrame(
            {
                "col_A": self.categorical_enum_series("col_A", 3),
                "col_B": self.numeric_series("col_B"),
            }
        )
        expected = pl.concat([a, b], how="vertical")
        return (a, b), expected, does_not_raise()

    def data_categorical_mismatch(self):
        a = pl.DataFrame(
            {
                "col_A": self.categorical_enum_series("col_A", 3),
                "col_B": self.numeric_series("col_B"),
            }
        )
        b = pl.DataFrame(
            {
                "col_A": self.categorical_enum_series("col_A", 4),
                "col_B": self.numeric_series("col_B"),
            }
        )
        err = pytest.raises(ValueError, match="categories for column col_A do not match")
        return (a, b), None, err

    def data_lazy_categorical_match(self):
        a = pl.DataFrame(
            {
                "col_A": self.categorical_enum_series("col_A", 3),
                "col_B": self.numeric_series("col_B"),
            }
        ).lazy()
        b = pl.DataFrame(
            {
                "col_A": self.categorical_enum_series("col_A", 3),
                "col_B": self.numeric_series("col_B"),
            }
        ).lazy()
        expected = pl.concat([a.collect(), b.collect()], how="vertical")
        return (a, b), expected, does_not_raise()


def _frames_equal(actual, expected):
    import polars.testing as pt

    pt.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("inputs,expected,expected_error", ConcatPolarsAxes1Cases().get_cases())
def test_safe_concat_polars_axis_1(inputs, expected, expected_error):
    with expected_error:
        actual = safe_concat(inputs, axis=1)
    if expected is not None:
        _frames_equal(actual, expected)


@pytest.mark.parametrize("inputs,expected,expected_error", ConcatPolarsAxes0Cases().get_cases())
def test_safe_concat_polars_axis_0(inputs, expected, expected_error):
    with expected_error:
        actual = safe_concat(inputs, axis=0)
    if expected is not None:
        _frames_equal(actual, expected)


def test_safe_concat_polars_mixed_backend_rejected():
    pd_df = __import__("pandas").DataFrame({"col_A": [1.0, 2.0, 3.0]})
    pl_df = pl.DataFrame({"col_A": [4.0, 5.0, 6.0]})
    with pytest.raises(TypeError, match="mixed backends"):
        safe_concat([pl_df, pd_df], axis=0)


def test_safe_concat_empty():
    with pytest.raises(ValueError, match="No objects to concatenate"):
        safe_concat([], axis=0)


def test_safe_concat_polars_categorical_mismatch_raises():
    df1 = pl.DataFrame({"c": pl.Series(["A", "B"], dtype=pl.Categorical)})
    df2 = pl.DataFrame({"c": pl.Series(["C", "D"], dtype=pl.Categorical)})
    with pytest.raises(ValueError, match="categories for column c do not match"):
        safe_concat([df1, df2], axis=0)


def test_safe_concat_polars_categorical_match_succeeds():
    df1 = pl.DataFrame({"c": pl.Series(["A", "B"], dtype=pl.Categorical)})
    df2 = pl.DataFrame({"c": pl.Series(["A", "B"], dtype=pl.Categorical)})
    out = safe_concat([df1, df2], axis=0)
    assert out.shape == (4, 1)


def test_safe_concat_polars_enum_mismatch_raises():
    df1 = pl.DataFrame({"c": pl.Series(["A"], dtype=pl.Enum(["A", "B"]))})
    df2 = pl.DataFrame({"c": pl.Series(["A"], dtype=pl.Enum(["A", "B", "C"]))})
    with pytest.raises(ValueError, match="categories for column c do not match"):
        safe_concat([df1, df2], axis=0)


class TestSafeConcatAxisValidation:
    @staticmethod
    @pytest.mark.parametrize("bad_axis", [2, -1, 3])
    def test_invalid_axis_polars_raises(bad_axis):
        from sksurv.util import safe_concat

        with pytest.raises(ValueError, match="axis must be 0 or 1"):
            safe_concat([pl.DataFrame({"x": [1]}), pl.DataFrame({"x": [2]})], axis=bad_axis)

    @staticmethod
    @pytest.mark.parametrize("bad_axis", [2, -1, 3])
    def test_invalid_axis_pandas_raises(bad_axis):
        import pandas as pd

        from sksurv.util import safe_concat

        with pytest.raises(ValueError, match="axis must be 0 or 1"):
            safe_concat([pd.DataFrame({"x": [1]}), pd.DataFrame({"x": [2]})], axis=bad_axis)

    @staticmethod
    def test_string_axis_index_accepted():
        import pandas as pd

        from sksurv.util import safe_concat

        result = safe_concat([pd.DataFrame({"x": [1]}), pd.DataFrame({"x": [2]})], axis="index")
        assert result.shape == (2, 1)

    @staticmethod
    def test_string_axis_columns_accepted():
        import pandas as pd

        from sksurv.util import safe_concat

        result = safe_concat([pd.DataFrame({"x": [1]}), pd.DataFrame({"x": [2]})], axis="columns")
        assert result.shape == (1, 2)
