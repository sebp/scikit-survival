from collections import OrderedDict
from contextlib import nullcontext as does_not_raise

import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
import pandas.testing as tm
import pytest

from sksurv.testing import FixtureParameterFactory
from sksurv.util import Surv, _PropertyAvailableIfDescriptor, property_available_if, safe_concat


class ConcatCasesFactory(FixtureParameterFactory):
    @property
    def rnd(self):
        return np.random.RandomState(14)

    def to_data_frame(self, data):
        return pd.DataFrame.from_dict(OrderedDict(data))

    def make_numeric_series(self, name):
        return pd.Series(self.rnd.randn(100), name=name)

    def make_categorical_5_series(self, name):
        return pd.Series(
            pd.Categorical.from_codes(self.rnd.binomial(4, 0.6, 100), ["C1", "C2", "C3", "C4", "C5"]), name=name
        )

    def make_categorical_4_series(self, name):
        return pd.Series(pd.Categorical.from_codes(self.rnd.binomial(3, 0.6, 100), ["C1", "C2", "C3", "C4"]), name=name)

    def make_categorical_3_series(self, name):
        return pd.Series(pd.Categorical.from_codes(self.rnd.binomial(2, 0.6, 100), ["C1", "C2", "C3"]), name=name)

    def make_binary_series(self, name):
        return pd.Series(pd.Categorical.from_codes(self.rnd.binomial(1, 0.6, 100), ["Yes", "No"]), name=name)


class ConcatCasesAxes1(ConcatCasesFactory):
    def data_numeric(self):
        a = self.make_numeric_series("col_A")
        b = self.make_numeric_series("col_B")

        expected = [(a.name, a), (b.name, b)]
        return (a, b), self.to_data_frame(expected), does_not_raise()

    def data_numeric_categorical(self):
        a = self.make_numeric_series("col_A")
        b = self.make_categorical_5_series("col_B")

        expected = [(a.name, a), (b.name, b)]
        return (a, b), self.to_data_frame(expected), does_not_raise()

    def data_frame_numeric_categorical(self):
        numeric_df = self.to_data_frame(
            [
                ("col_A", self.make_numeric_series("col_A")),
                ("col_B", self.make_categorical_5_series("col_B")),
            ]
        )

        cat_series = self.make_categorical_5_series("col_C")

        expected_df = numeric_df.copy()
        expected_df.loc[:, "col_C"] = cat_series
        return (numeric_df, cat_series), expected_df, does_not_raise()

    def data_duplicate_columns(self):
        numeric_df = self.to_data_frame(
            [
                ("col_N", self.make_numeric_series("col_N")),
                ("col_B", self.make_categorical_5_series("col_B")),
                ("col_A", self.make_categorical_5_series("col_A")),
            ]
        )

        cat_df = pd.DataFrame.from_dict(
            OrderedDict(
                [
                    ("col_A", self.make_categorical_5_series("col_A")),
                    ("col_C", self.make_binary_series("col_C")),
                ]
            )
        )

        err = pytest.raises(ValueError, match="duplicate columns col_A")
        return (numeric_df, cat_df), None, err


class ConcatCasesAxes0(ConcatCasesFactory):
    def data_categorical(self):
        a = self.to_data_frame(
            [
                ("col_A", self.make_categorical_3_series("col_A")),
                ("col_B", self.make_numeric_series("col_B")),
            ]
        )
        b = self.to_data_frame(
            [("col_A", self.make_categorical_3_series("col_A")), ("col_B", self.make_numeric_series("col_B"))]
        )

        expected = [
            (
                "col_A",
                pd.Series(
                    pd.Categorical.from_codes(
                        np.r_[a.col_A.cat.codes.values, b.col_A.cat.codes.values], ["C1", "C2", "C3"]
                    )
                ),
            ),
            ("col_B", np.r_[a.col_B.values, b.col_B.values]),
        ]
        expected_df = self.to_data_frame(expected)
        expected_df.index = pd.Index(a.index.tolist() + b.index.tolist())

        return (a, b), expected_df, does_not_raise()

    def data_categorical_mismatch(self):
        a = self.to_data_frame(
            [
                ("col_A", self.make_categorical_3_series("col_A")),
                ("col_B", self.make_numeric_series("col_B")),
            ]
        )
        b = self.to_data_frame(
            [
                ("col_A", self.make_categorical_4_series("col_A")),
                ("col_B", self.make_numeric_series("col_B")),
            ]
        )

        err = pytest.raises(ValueError, match="categories for column col_A do not match")
        return (a, b), None, err


@pytest.mark.parametrize("inputs,expected_df,expected_error", ConcatCasesAxes1().get_cases())
def test_concat_axis_1(inputs, expected_df, expected_error):
    with expected_error:
        actual_df = safe_concat(inputs, axis=1)

    if expected_df is not None:
        tm.assert_frame_equal(actual_df, expected_df)


@pytest.mark.parametrize("inputs,expected_df,expected_error", ConcatCasesAxes0().get_cases())
def test_concat_axis_0(inputs, expected_df, expected_error):
    with expected_error:
        actual_df = safe_concat(inputs, axis=0)

    if expected_df is not None:
        tm.assert_frame_equal(actual_df, expected_df)


class SurvCases(FixtureParameterFactory):
    @property
    def event_and_time(self):
        event = np.random.binomial(1, 0.5, size=100)
        time = np.exp(np.random.randn(100))
        return event, time


class SurvArrayCases(SurvCases):
    def get_surv_arrays(self, event_name="event", time_name="time"):
        event, time = self.event_and_time

        expected = np.empty(dtype=[(event_name, bool), (time_name, float)], shape=event.shape[0])
        expected[event_name] = event.astype(bool)
        expected[time_name] = time

        return (event, time), expected

    def data_list(self):
        (event, time), expected = self.get_surv_arrays()

        inputs = (list(event.astype(bool)), list(time))
        return inputs, {}, expected, does_not_raise()

    def data_bool(self):
        (event, time), expected = self.get_surv_arrays()

        inputs = (event.astype(bool), time)
        return inputs, {}, expected, does_not_raise()

    def data_with_names(self):
        (event, time), expected = self.get_surv_arrays("death", "survival_time")

        inputs = (event.astype(bool), time)
        kwargs = {"name_time": "survival_time", "name_event": "death"}
        return inputs, kwargs, expected, does_not_raise()

    def data_with_one_name_1(self):
        (event, time), expected = self.get_surv_arrays("death")

        inputs = (event.astype(bool), time)
        kwargs = {"name_event": "death"}
        return inputs, kwargs, expected, does_not_raise()

    def data_with_one_name_2(self):
        (event, time), expected = self.get_surv_arrays("event", "survival_time")

        inputs = (event.astype(bool), time)
        kwargs = {"name_time": "survival_time"}
        return inputs, kwargs, expected, does_not_raise()

    def data_array_int_event(self):
        inputs, expected = self.get_surv_arrays()

        return inputs, {}, expected, does_not_raise()

    def data_int_time(self):
        event, time = self.event_and_time
        time += 1
        time *= time

        expected = np.empty(dtype=[("event", bool), ("time", float)], shape=event.shape[0])
        expected["event"] = event.astype(bool)
        expected["time"] = time.astype(int)

        inputs = (event.astype(bool), time.astype(int))
        return inputs, {}, expected, does_not_raise()

    def data_float(self):
        (event, time), expected = self.get_surv_arrays()

        inputs = (event.astype(float), time)
        return inputs, {}, expected, does_not_raise()

    def data_shape_mismatch_0(self):
        event, time = self.event_and_time

        msg = "Found input variables with inconsistent numbers of samples"
        err = pytest.raises(ValueError, match=msg)
        inputs = (event[1:], time)
        return inputs, {}, None, err

    def data_shape_mismatch_1(self):
        event, time = self.event_and_time
        msg = "Found input variables with inconsistent numbers of samples"
        err = pytest.raises(ValueError, match=msg)
        inputs = (event, time[1:])
        return inputs, {}, None, err

    def data_event_value_wrong_1(self):
        event, time = self.event_and_time
        event += 1

        err = pytest.raises(ValueError, match="non-boolean event indicator must contain 0 and 1 only")
        return (event, time), {}, None, err

    def data_event_value_wrong_2(self):
        event, time = self.event_and_time
        event -= 1

        err = pytest.raises(ValueError, match="non-boolean event indicator must contain 0 and 1 only")
        return (event, time), {}, None, err

    def data_event_value_wrong_3(self):
        event, time = self.event_and_time
        event[event == 0] = 3

        err = pytest.raises(ValueError, match="non-boolean event indicator must contain 0 and 1 only")

        return (event, time), {}, None, err

    def data_event_value_wrong_4(self):
        event, time = self.event_and_time
        event[1] = 3

        err = pytest.raises(ValueError, match="event indicator must be binary")
        return (event, time), {}, None, err

    def data_event_value_wrong_5(self):
        event, time = self.event_and_time
        event = np.arange(event.shape[0])

        err = pytest.raises(ValueError, match="event indicator must be binary")
        return (event, time), {}, None, err

    def data_names_match(self):
        event, time = self.event_and_time

        err = pytest.raises(ValueError, match="name_time must be different from name_event")
        kwargs = {"name_event": "time_and_event", "name_time": "time_and_event"}
        return (event, time), kwargs, None, err


@pytest.mark.parametrize("args,kwargs,expected,expected_error", SurvArrayCases().get_cases())
def test_from_arrays(args, kwargs, expected, expected_error):
    with expected_error:
        y = Surv.from_arrays(*args, **kwargs)

    if expected is not None:
        assert_array_equal(y, expected)


class SurvDataFrameCases(SurvCases):
    def get_surv_data_frame(self, event_name="event", time_name="time"):
        event, time = self.event_and_time
        df = pd.DataFrame({event_name: event, time_name: time})

        expected = np.empty(dtype=[(str(event_name), bool), (str(time_name), float)], shape=100)
        expected[str(event_name)] = event.astype(bool)
        expected[str(time_name)] = time

        return df, expected

    def data_bool(self):
        data, expected = self.get_surv_data_frame()
        data["event"] = data["event"].astype(bool)

        inputs = ("event", "time", data)
        return inputs, expected, does_not_raise()

    def data_int(self):
        data, expected = self.get_surv_data_frame()
        inputs = ("event", "time", data)
        return inputs, expected, does_not_raise()

    def data_float(self):
        data, expected = self.get_surv_data_frame()
        data["event"] = data["event"].astype(float)
        inputs = ("event", "time", data)
        return inputs, expected, does_not_raise()

    def data_no_str_columns(self):
        data, expected = self.get_surv_data_frame(event_name=0, time_name=1)
        inputs = (0, 1, data)
        return inputs, expected, does_not_raise()

    def data_column_names(self):
        data, expected = self.get_surv_data_frame(event_name="death", time_name="time_to_death")
        inputs = ("death", "time_to_death", data)
        return inputs, expected, does_not_raise()

    def data_no_such_column_0(self):
        data, _ = self.get_surv_data_frame()

        err = pytest.raises(KeyError, match="unknown")
        inputs = ("unknown", "time", data)
        return inputs, None, err

    def data_no_such_column_1(self):
        data, _ = self.get_surv_data_frame()

        err = pytest.raises(KeyError, match="unknown")
        inputs = ("event", "unknown", data)
        return inputs, None, err

    def data_wrong_class_0(self):
        data, _ = self.get_surv_data_frame()

        err = pytest.raises(TypeError, match=r"expected pandas.DataFrame, but got <class 'dict'>")
        inputs = ("event", "time", data.to_dict())
        return inputs, None, err

    def data_wrong_class_1(self):
        data, _ = self.get_surv_data_frame()

        err = pytest.raises(TypeError, match=r"expected pandas.DataFrame, but got <class 'numpy.ndarray'>")
        inputs = ("event", "time", data.values)
        return inputs, None, err


@pytest.mark.parametrize("args,expected,expected_error", SurvDataFrameCases().get_cases())
def test_from_dataframe(args, expected, expected_error):
    with expected_error:
        y = Surv.from_dataframe(*args)

    if expected is not None:
        assert_array_equal(y, expected)


def test_cond_avail_property():
    class WithCondProp:
        def __init__(self, val):
            self.avail = False
            self._prop = val

        @property_available_if(lambda self: self.avail)
        def prop(self):
            return self._prop

        no_prop = _PropertyAvailableIfDescriptor(lambda self: self.avail, fget=None)

    testval = 43
    msg = "has no attribute 'prop'"

    assert WithCondProp.prop is not None

    test_obj = WithCondProp(testval)
    with pytest.raises(AttributeError, match=msg):
        _ = test_obj.prop
    assert test_obj.avail is False

    test_obj.avail = True
    assert test_obj.prop == testval

    test_obj.avail = False
    with pytest.raises(AttributeError, match=msg):
        _ = test_obj.prop

    test_obj.avail = True
    with pytest.raises(AttributeError, match="has no getter"):
        _ = test_obj.no_prop
