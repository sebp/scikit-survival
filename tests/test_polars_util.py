"""Tests for ``sksurv.util`` with polars input."""

from contextlib import nullcontext as does_not_raise

from dataframe_test_utils import to_polars_via_interchange
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
import polars as pl
from polars.exceptions import ColumnNotFoundError
import pytest

from sksurv.testing import FixtureParameterFactory
from sksurv.util import Surv


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
        df = to_polars_via_interchange(pd.DataFrame(data))
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
