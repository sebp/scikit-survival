from collections import OrderedDict

import numpy
from numpy.testing import assert_array_equal
import pandas.util.testing as tm
import pandas
import pytest

from sksurv.util import safe_concat, Surv


class TestUtil(object):
    @staticmethod
    def test_concat_numeric():
        rnd = numpy.random.RandomState(14)
        a = pandas.Series(rnd.randn(100), name="col_A")
        b = pandas.Series(rnd.randn(100), name="col_B")

        expected_df = pandas.DataFrame.from_dict(OrderedDict(
            [(a.name, a), (b.name, b)]
        ))

        actual_df = safe_concat((a, b), axis=1)

        tm.assert_frame_equal(actual_df, expected_df)

    @staticmethod
    def test_concat_numeric_categorical():
        rnd = numpy.random.RandomState(14)
        a = pandas.Series(rnd.randn(100), name="col_A")
        b = pandas.Series(pandas.Categorical.from_codes(
            rnd.binomial(4, 0.6, 100), ["C1", "C2", "C3", "C4", "C5"]), name="col_B")

        expected_df = pandas.DataFrame.from_dict(OrderedDict(
            [(a.name, a), (b.name, b)]
        ))

        actual_df = safe_concat((a, b), axis=1)

        tm.assert_frame_equal(actual_df, expected_df)

    @staticmethod
    def test_concat_categorical():
        rnd = numpy.random.RandomState(14)
        a = pandas.DataFrame.from_dict(OrderedDict([
            ("col_A", pandas.Series(pandas.Categorical.from_codes(
                rnd.binomial(2, 0.6, 100), ["C1", "C2", "C3"]), name="col_A")),
            ("col_B", rnd.randn(100))]))
        b = pandas.DataFrame.from_dict(OrderedDict([
            ("col_A", pandas.Series(pandas.Categorical.from_codes(
                rnd.binomial(2, 0.2, 100), ["C1", "C2", "C3"]), name="col_A")),
            ("col_B", rnd.randn(100))]))

        expected_series = pandas.DataFrame.from_dict(OrderedDict([
            ("col_A", pandas.Series(pandas.Categorical.from_codes(
                numpy.concatenate((a.col_A.cat.codes.values, b.col_A.cat.codes.values)),
                ["C1", "C2", "C3"]
            ))),
            ("col_B", numpy.concatenate((a.col_B.values, b.col_B.values)))
        ]))
        expected_series.index = pandas.Index(a.index.tolist() + b.index.tolist())

        actual_series = safe_concat((a, b), axis=0)

        tm.assert_frame_equal(actual_series, expected_series)

    @staticmethod
    def test_concat_categorical_mismatch():
        rnd = numpy.random.RandomState(14)
        a = pandas.DataFrame.from_dict(OrderedDict([
            ("col_A", pandas.Series(pandas.Categorical.from_codes(
                rnd.binomial(2, 0.6, 100), ["C1", "C2", "C3"]), name="col_A")),
            ("col_B", rnd.randn(100))]))
        b = pandas.DataFrame.from_dict(OrderedDict([
            ("col_A", pandas.Series(pandas.Categorical.from_codes(
                rnd.binomial(3, 0.6, 100), ["C1", "C2", "C3", "C4"]), name="col_A")),
            ("col_B", rnd.randn(100))]))

        with pytest.raises(ValueError, match="categories for column col_A do not match"):
            safe_concat((a, b), axis=0)

    @staticmethod
    def test_concat_dataframe_numeric_categorical():
        rnd = numpy.random.RandomState(14)
        numeric_df = pandas.DataFrame.from_dict(OrderedDict(
            [("col_A", rnd.randn(100)), ("col_B", rnd.randn(100))]
        ))

        cat_series = pandas.Series(pandas.Categorical.from_codes(
            rnd.binomial(4, 0.6, 100), ["C1", "C2", "C3", "C4", "C5"]), name="col_C")

        expected_df = numeric_df.copy()
        expected_df["col_C"] = cat_series

        actual_df = safe_concat((numeric_df, cat_series), axis=1)

        tm.assert_frame_equal(actual_df, expected_df)

    @staticmethod
    def test_concat_duplicate_columns():
        rnd = numpy.random.RandomState(14)
        numeric_df = pandas.DataFrame.from_dict(OrderedDict([
            ("col_N", rnd.randn(100)), ("col_B", rnd.randn(100)),
            ("col_A", pandas.Series(pandas.Categorical.from_codes(
                rnd.binomial(4, 0.2, 100), ["C1", "C2", "C3", "C4", "C5"]), name="col_A")),
        ]))

        cat_df = pandas.DataFrame.from_dict(OrderedDict([
            ("col_A", pandas.Series(pandas.Categorical.from_codes(
                rnd.binomial(4, 0.6, 100), ["C1", "C2", "C3", "C4", "C5"]), name="col_A")),
            ("col_C", pandas.Series(pandas.Categorical.from_codes(
                rnd.binomial(1, 0.6, 100), ["Yes", "No"]), name="col_C")),
        ]))

        with pytest.raises(ValueError, match="duplicate columns col_A"):
            safe_concat((numeric_df, cat_df), axis=1)


@pytest.fixture
def surv_arrays():
    event = numpy.random.binomial(1, 0.5, size=100)
    time = numpy.exp(numpy.random.randn(100))
    return event, time


@pytest.fixture
def surv_data_frame():
    df = pandas.DataFrame({'event': numpy.random.binomial(1, 0.5, size=100),
                           'time': numpy.exp(numpy.random.randn(100))})
    return df


class TestSurv(object):

    @staticmethod
    def test_from_list(surv_arrays):
        event, time = surv_arrays

        expected = numpy.empty(dtype=[('event', bool), ('time', float)], shape=100)
        expected['event'] = event.astype(bool)
        expected['time'] = time

        y = Surv.from_arrays(list(event.astype(bool)), list(time))
        assert_array_equal(y, expected)

    @staticmethod
    def test_from_array_bool(surv_arrays):
        event, time = surv_arrays

        expected = numpy.empty(dtype=[('event', bool), ('time', float)], shape=100)
        expected['event'] = event.astype(bool)
        expected['time'] = time

        y = Surv.from_arrays(event.astype(bool), time)
        assert_array_equal(y, expected)

    @staticmethod
    def test_from_array_with_names(surv_arrays):
        event, time = surv_arrays

        expected = numpy.empty(dtype=[('death', bool), ('survival_time', float)], shape=100)
        expected['death'] = event.astype(bool)
        expected['survival_time'] = time

        y = Surv.from_arrays(event.astype(bool), time, name_time='survival_time', name_event='death')
        assert_array_equal(y, expected)

    @staticmethod
    def test_from_array_with_one_name_1(surv_arrays):
        event, time = surv_arrays

        expected = numpy.empty(dtype=[('death', bool), ('time', float)], shape=100)
        expected['death'] = event.astype(bool)
        expected['time'] = time

        y = Surv.from_arrays(event.astype(bool), time, name_event='death')
        assert_array_equal(y, expected)

    @staticmethod
    def test_from_array_with_one_name_2(surv_arrays):
        event, time = surv_arrays

        expected = numpy.empty(dtype=[('event', bool), ('survival_time', float)], shape=100)
        expected['event'] = event.astype(bool)
        expected['survival_time'] = time

        y = Surv.from_arrays(event.astype(bool), time, name_time='survival_time')
        assert_array_equal(y, expected)

    @staticmethod
    def test_from_array_int_event(surv_arrays):
        event, time = surv_arrays

        expected = numpy.empty(dtype=[('event', bool), ('time', float)], shape=100)
        expected['event'] = event.astype(bool)
        expected['time'] = time

        y = Surv.from_arrays(event, time)
        assert_array_equal(y, expected)

    @staticmethod
    def test_from_array_int_time(surv_arrays):
        event, time = surv_arrays
        time += 1
        time *= time

        expected = numpy.empty(dtype=[('event', bool), ('time', float)], shape=100)
        expected['event'] = event.astype(bool)
        expected['time'] = time.astype(int)

        y = Surv.from_arrays(event.astype(bool), time.astype(int))
        assert_array_equal(y, expected)

    @staticmethod
    def test_from_array_float(surv_arrays):
        event, time = surv_arrays

        expected = numpy.empty(dtype=[('event', bool), ('time', float)], shape=100)
        expected['event'] = event.astype(bool)
        expected['time'] = time

        y = Surv.from_arrays(event.astype(float), time)
        assert_array_equal(y, expected)

    @staticmethod
    def test_from_array_shape_mismatch(surv_arrays):
        event, time = surv_arrays

        msg = "Found input variables with inconsistent numbers of samples"
        with pytest.raises(ValueError, match=msg):
            Surv.from_arrays(event[1:], time)

        with pytest.raises(ValueError, match=msg):
            Surv.from_arrays(event, time[1:])

    @staticmethod
    def test_from_array_event_value_wrong_1(surv_arrays):
        event, time = surv_arrays
        event += 1

        with pytest.raises(ValueError,
                           match="non-boolean event indicator must contain 0 and 1 only"):
            Surv.from_arrays(event, time)

    @staticmethod
    def test_from_array_event_value_wrong_2(surv_arrays):
        event, time = surv_arrays
        event -= 1

        with pytest.raises(ValueError,
                           match="non-boolean event indicator must contain 0 and 1 only"):
            Surv.from_arrays(event, time)

    @staticmethod
    def test_from_array_event_value_wrong_3(surv_arrays):
        event, time = surv_arrays
        event[event == 0] = 3

        with pytest.raises(ValueError,
                           match="non-boolean event indicator must contain 0 and 1 only"):
            Surv.from_arrays(event, time)

    @staticmethod
    def test_from_array_event_value_wrong_4(surv_arrays):
        event, time = surv_arrays
        event[1] = 3

        with pytest.raises(ValueError,
                           match="event indicator must be binary"):
            Surv.from_arrays(event, time)

    @staticmethod
    def test_from_array_event_value_wrong_5(surv_arrays):
        event, time = surv_arrays
        event = numpy.arange(event.shape[0])

        with pytest.raises(ValueError,
                           match="event indicator must be binary"):
            Surv.from_arrays(event, time)

    @staticmethod
    def test_from_array_names_match(surv_arrays):
        event, time = surv_arrays

        with pytest.raises(ValueError,
                           match="name_time must be different from name_event"):
            Surv.from_arrays(event, time,
                             name_event='time_and_event', name_time='time_and_event')

    @staticmethod
    def test_from_dataframe_bool(surv_data_frame):
        data = surv_data_frame
        data['event'] = data['event'].astype(bool)

        expected = numpy.empty(dtype=[('event', bool), ('time', float)], shape=100)
        expected['event'] = data['event']
        expected['time'] = data['time']

        y = Surv.from_dataframe('event', 'time', data)
        assert_array_equal(y, expected)

    @staticmethod
    def test_from_dataframe_int(surv_data_frame):
        data = surv_data_frame

        expected = numpy.empty(dtype=[('event', bool), ('time', float)], shape=100)
        expected['event'] = data['event'].astype(bool)
        expected['time'] = data['time']

        y = Surv.from_dataframe('event', 'time', data)
        assert_array_equal(y, expected)

    @staticmethod
    def test_from_dataframe_float(surv_data_frame):
        data = surv_data_frame
        data['event'] = data['event'].astype(float)

        expected = numpy.empty(dtype=[('event', bool), ('time', float)], shape=100)
        expected['event'] = data['event'].astype(bool)
        expected['time'] = data['time']

        y = Surv.from_dataframe('event', 'time', data)
        assert_array_equal(y, expected)

    @staticmethod
    def test_from_dataframe_no_str_columns(surv_data_frame):
        data = surv_data_frame
        data['event'] = data['event'].astype(bool)

        expected = numpy.empty(dtype=[('0', bool), ('1', float)], shape=100)
        expected['0'] = data['event']
        expected['1'] = data['time']

        y = Surv.from_dataframe(0, 1, data.rename(columns={'event': 0, 'time': 1}))
        assert_array_equal(y, expected)

    @staticmethod
    def test_from_dataframe_column_names(surv_data_frame):
        data = surv_data_frame.rename(columns={'event': 'death', 'time': 'time_to_death'})
        data['death'] = data['death'].astype(bool)

        expected = numpy.empty(dtype=[('death', bool), ('time_to_death', float)], shape=100)
        expected['death'] = data['death']
        expected['time_to_death'] = data['time_to_death']

        y = Surv.from_dataframe('death', 'time_to_death', data)
        assert_array_equal(y, expected)

    @staticmethod
    def test_from_dataframe_no_such_column(surv_data_frame, pandas_version_under_0p24):
        data = surv_data_frame
        data['event'] = data['event'].astype(bool)

        expected = numpy.empty(dtype=[('event', bool), ('time', float)], shape=100)
        expected['event'] = data['event']
        expected['time'] = data['time']

        if pandas_version_under_0p24:
            match = r'the label \[unknown\] is not in the \[columns\]'
        else:
            match = 'unknown'

        with pytest.raises(KeyError,
                           match=match):
            Surv.from_dataframe('unknown', 'time', data)

        with pytest.raises(KeyError,
                           match=match):
            Surv.from_dataframe('event', 'unknown', data)

    @staticmethod
    def test_from_dataframe_wrong_class(surv_data_frame):
        data = surv_data_frame

        with pytest.raises(TypeError,
                           match=r"exepected pandas.DataFrame, but got <class 'dict'>"):
            Surv.from_dataframe('event', 'time', data.to_dict())

        with pytest.raises(TypeError,
                           match=r"exepected pandas.DataFrame, but got <class 'numpy.ndarray'>"):
            Surv.from_dataframe('event', 'time', data.values)
