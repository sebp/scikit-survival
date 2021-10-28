from collections import OrderedDict

import numpy
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pandas
import pandas.testing as tm
import pytest

from sksurv import column


@pytest.fixture()
def numeric_data():
    data = pandas.DataFrame(numpy.arange(50, dtype=float).reshape(10, 5))

    expected = numpy.array([[-1.486301, -1.486301, -1.486301, -1.486301, -1.486301],
                            [-1.156012, -1.156012, -1.156012, -1.156012, -1.156012],
                            [-0.825723, -0.825723, -0.825723, -0.825723, -0.825723],
                            [-0.495434, -0.495434, -0.495434, -0.495434, -0.495434],
                            [-0.165145, -0.165145, -0.165145, -0.165145, -0.165145],
                            [0.165145, 0.165145, 0.165145, 0.165145, 0.165145],
                            [0.495434, 0.495434, 0.495434, 0.495434, 0.495434],
                            [0.825723, 0.825723, 0.825723, 0.825723, 0.825723],
                            [1.156012, 1.156012, 1.156012, 1.156012, 1.156012],
                            [1.486301, 1.486301, 1.486301, 1.486301, 1.486301]])
    return data, expected


@pytest.fixture()
def non_numeric_data_frame():
    data = pandas.DataFrame({'q1': ['no', 'no', 'yes', 'yes', 'no', 'no', None, 'yes', 'no', None],
                             'q2': ['maybe', 'no', 'yes', 'maybe', 'yes', 'no', None, 'maybe', 'no',
                                    'yes'],
                             'q3': [1, 2, 1, 3, 1, 2, numpy.nan, numpy.nan, 3, 2]})

    data['q3'] = data['q3'].astype('category')
    return data


class TestColumn:
    @staticmethod
    def test_standardize_numeric(numeric_data):
        numeric_data_frame, expected = numeric_data
        result = column.standardize(numeric_data_frame)

        assert isinstance(result, pandas.DataFrame)
        assert_array_almost_equal(expected, result)

    @staticmethod
    def test_standardize_float_numpy_array(numeric_data):
        numeric_data_frame, expected = numeric_data
        result = column.standardize(numeric_data_frame.values)

        assert isinstance(result, numpy.ndarray)
        assert_array_almost_equal(expected, result)

    @staticmethod
    def test_standardize_int_numpy_array(numeric_data):
        numeric_data_frame, expected = numeric_data
        result = column.standardize(numeric_data_frame.values.astype(int))

        assert isinstance(result, numpy.ndarray)
        assert_array_almost_equal(expected, result)

    @staticmethod
    def test_standardize_not_inplace(numeric_data):
        numeric_data_frame, expected = numeric_data
        numeric_array = numeric_data_frame.values

        before = numeric_array.copy()
        result = column.standardize(numeric_array)
        assert_array_almost_equal(expected, result)
        assert_array_almost_equal(before, numeric_array)

    @staticmethod
    def test_standardize_non_numeric(non_numeric_data_frame):
        result = column.standardize(non_numeric_data_frame)

        assert isinstance(result, pandas.DataFrame)
        tm.assert_frame_equal(non_numeric_data_frame, result)

    @staticmethod
    def test_standardize_non_numeric_numpy_array(non_numeric_data_frame):
        result = column.standardize(non_numeric_data_frame.values)

        assert isinstance(result, numpy.ndarray)

        assert_array_equal(pandas.isnull(non_numeric_data_frame),
                           pandas.isnull(result))

        non_nan_idx = [0, 1, 2, 3, 4, 5, 8, 9]

        assert_array_equal(non_numeric_data_frame.iloc[non_nan_idx, :].values,
                           result[non_nan_idx, :])

    @staticmethod
    def test_standardize_mixed(numeric_data, non_numeric_data_frame):
        numeric_data_frame, expected = numeric_data
        mixed_data_frame = pandas.concat((numeric_data_frame, non_numeric_data_frame), axis=1)
        result = column.standardize(mixed_data_frame)

        assert isinstance(result, pandas.DataFrame)
        assert_array_almost_equal(expected, result.iloc[:, :numeric_data_frame.shape[1]].values)

        tm.assert_frame_equal(non_numeric_data_frame, result.iloc[:, numeric_data_frame.shape[1]:])

    @staticmethod
    def test_standardize_mixed_numpy_array(numeric_data, non_numeric_data_frame):
        numeric_data_frame, _ = numeric_data
        mixed_data_frame = pandas.concat((numeric_data_frame, non_numeric_data_frame), axis=1)
        result = column.standardize(mixed_data_frame.values)

        assert_array_equal(pandas.isnull(mixed_data_frame),
                           pandas.isnull(result))

        assert_array_almost_equal(numeric_data_frame, result[:, :numeric_data_frame.shape[1]])

        non_nan_idx = [0, 1, 2, 3, 4, 5, 8, 9]

        assert_array_equal(non_numeric_data_frame.iloc[non_nan_idx, :].values,
                           result[:, numeric_data_frame.shape[1]:][non_nan_idx, :])


class TestEncodeCategorical:
    @staticmethod
    def test_series_categorical():
        input_series = pandas.Series(pandas.Categorical.from_codes([1, 1, 0, 2, 0, 1, 2, 1, 2, 0, 0, 1, 2, 2],
                                                                   ["small", "medium", "large"], ordered=False),
                                     name="a_series")
        expected_df = pandas.DataFrame.from_dict(OrderedDict(
            [("a_series=medium", numpy.array([1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0], dtype=float)),
             ("a_series=large", numpy.array([0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1], dtype=float))
             ])
        )

        actual_df = column.encode_categorical(input_series)

        tm.assert_frame_equal(actual_df, expected_df, check_exact=True)

    @staticmethod
    def test_series_numeric():
        input_series = pandas.Series([0.5, 0.1, 10, 25, 3.8, 11, 2256, -1, -0.2, 3.14], name="a_series")

        with pytest.raises(TypeError, match="series must be of categorical dtype, but was float"):
            column.encode_categorical(input_series)

    @staticmethod
    def test_case1():
        a = numpy.r_[
            numpy.repeat(["large"], 10),
            numpy.repeat(["small"], 5),
            numpy.repeat(["tiny"], 13),
            numpy.repeat(["medium"], 3)]
        b = numpy.r_[
            numpy.repeat(["yes"], 8),
            numpy.repeat(["no"], 23)]

        rnd = numpy.random.RandomState(0)
        c = rnd.randn(len(a))

        df = pandas.DataFrame.from_dict(
            OrderedDict([
                ("a_category", a),
                ("a_binary", b),
                ("a_number", c.copy())])
        )

        actual_df = column.encode_categorical(df)

        eb = numpy.r_[
            numpy.repeat([1.], 8),
            numpy.repeat([0.], 23)]

        a_tiny = numpy.zeros(31, dtype=float)
        a_tiny[15:28] = 1

        a_small = numpy.zeros(31, dtype=float)
        a_small[10:15] = 1

        a_medium = numpy.zeros(31, dtype=float)
        a_medium[-3:] = 1

        expected_df = pandas.DataFrame.from_dict(
            OrderedDict([
                ("a_category=medium", a_medium),
                ("a_category=small", a_small),
                ("a_category=tiny", a_tiny),
                ("a_binary=yes", eb),
                ("a_number", c.copy())])
        )

        assert actual_df.shape == expected_df.shape
        tm.assert_frame_equal(actual_df, expected_df, check_exact=True)

    @staticmethod
    def test_duplicate_index():
        a = numpy.r_[
            numpy.repeat(["large"], 10),
            numpy.repeat(["small"], 6),
            numpy.repeat(["tiny"], 13),
            numpy.repeat(["medium"], 3)]
        rnd = numpy.random.RandomState(0)
        c = rnd.randn(len(a))

        index = numpy.ceil(numpy.arange(0, len(a) // 2, 0.5))
        df = pandas.DataFrame.from_dict(OrderedDict([
            ("a_category", pandas.Series(a, index=index)),
            ("a_number", pandas.Series(c, index=index, copy=True))
        ]))

        actual_df = column.encode_categorical(df)

        expected_df = pandas.DataFrame(numpy.zeros((32, 3), dtype=float),
                                       index=index,
                                       columns=["a_category=medium", "a_category=small", "a_category=tiny"])
        # tiny
        expected_df.iloc[16:29, 2] = 1
        # small
        expected_df.iloc[10:16, 1] = 1
        # medium
        expected_df.iloc[-3:, 0] = 1

        expected_df["a_number"] = c

        assert actual_df.shape == expected_df.shape
        tm.assert_frame_equal(actual_df, expected_df, check_exact=True)

    @staticmethod
    def test_case_numeric():
        a = numpy.array([0, 1, 1, 0, 1, 0, 0, 1, 0, 1], dtype=object)
        b = numpy.array([1, 2, 1, 3, 2, 1, 3, 2, 3, 1], dtype=object)
        c = numpy.array([1./128, 1./32, 1., 1./8, 1./32, 1., 1./128, 1./8, 1., 1./32], dtype=object)

        df = pandas.DataFrame({"a_binary_int": a.copy(),
                               "a_three_int": b.copy(),
                               "a_four_float": c.copy()})

        actual_df = column.encode_categorical(df)

        expected_df = pandas.DataFrame({
            "a_binary_int=1": a.astype(float),
            "a_three_int=2": (b == 2).astype(float),
            "a_three_int=3": (b == 3).astype(float),
            "a_four_float={}".format(1. / 32): (c == 1. / 32).astype(float),
            "a_four_float={}".format(1. / 8): (c == 1. / 8).astype(float),
            "a_four_float={}".format(1.): (c == 1.).astype(float),
        })

        assert actual_df.shape == expected_df.shape
        tm.assert_frame_equal(actual_df, expected_df, check_exact=True)

    @staticmethod
    def test_with_missing():
        b = numpy.r_[
            numpy.repeat(["yes"], 5),
            numpy.repeat([None], 10),
            numpy.repeat(["no"], 16)]

        rnd = numpy.random.RandomState(0)
        c = rnd.randn(len(b))

        df = pandas.DataFrame(OrderedDict([("a_binary", b),
                                          ("a_number", c.copy())]))

        actual_df = column.encode_categorical(df)

        eb = numpy.r_[
            numpy.repeat([1.], 5),
            numpy.repeat([numpy.nan], 10),
            numpy.repeat([0.], 16)]

        d = OrderedDict()
        d['a_binary=yes'] = eb
        d['a_number'] = c.copy()
        expected_df = pandas.DataFrame(d)

        assert actual_df.shape == expected_df.shape
        tm.assert_frame_equal(actual_df.isnull(), expected_df.isnull())
        tm.assert_frame_equal(actual_df.dropna(), expected_df.dropna(), check_exact=True)

    @staticmethod
    def test_drop_all_missing():
        b = numpy.r_[
            numpy.repeat(["yes"], 5),
            numpy.repeat([None], 10),
            numpy.repeat(["no"], 16)]

        all_missing = numpy.repeat([None], len(b))

        df = pandas.DataFrame({"a_binary": b,
                               "bogus": all_missing})

        actual_df = column.encode_categorical(df)

        eb = numpy.r_[
            numpy.repeat([1.], 5),
            numpy.repeat([numpy.nan], 10),
            numpy.repeat([0.], 16)]

        expected_df = pandas.DataFrame({"a_binary=yes": eb})

        assert actual_df.shape == expected_df.shape
        tm.assert_frame_equal(actual_df.isnull(), expected_df.isnull())
        tm.assert_frame_equal(actual_df.dropna(), expected_df.dropna(), check_exact=True)

    @staticmethod
    def test_retain_all_missing():
        b = numpy.r_[
            numpy.repeat(["yes"], 5),
            numpy.repeat([None], 10),
            numpy.repeat(["no"], 16)]

        all_missing = numpy.repeat([None], len(b))

        df = pandas.DataFrame({"a_binary": b,
                               "bogus": all_missing})

        actual_df = column.encode_categorical(df, allow_drop=False)

        eb = numpy.r_[
            numpy.repeat([1.], 5),
            numpy.repeat([numpy.nan], 10),
            numpy.repeat([0.], 16)]

        expected_df = pandas.DataFrame({"a_binary=yes": eb,
                                        "bogus": all_missing.copy()})

        assert actual_df.shape == expected_df.shape
        tm.assert_frame_equal(actual_df.isnull(), expected_df.isnull())
        tm.assert_frame_equal(actual_df.dropna(), expected_df.dropna(), check_exact=True)

    @staticmethod
    def test_retain_only_one_level():
        b = numpy.r_[numpy.repeat(["yes"], 10)]

        df = pandas.DataFrame({"categorical_col_with_only_one_level": b})

        expected_df = df.copy(deep=True)

        actual_df = column.encode_categorical(df, allow_drop=False)

        assert actual_df.shape == expected_df.shape
        tm.assert_frame_equal(actual_df.isnull(), expected_df.isnull())
        tm.assert_frame_equal(actual_df.dropna(), expected_df.dropna(), check_exact=True)


def test_categorical_series_to_numeric():
    input_series = pandas.Series(["a", "a", "b", "b", "b", "c"], name="Thr33",
                                 index=["Alpha", "Beta", "Gamma", "Delta", "Eta", "Mu"])
    expected = pandas.Series([0, 0, 1, 1, 1, 2], name="Thr33",
                             index=["Alpha", "Beta", "Gamma", "Delta", "Eta", "Mu"],
                             dtype=numpy.int64)

    actual = column.categorical_to_numeric(input_series)

    tm.assert_series_equal(actual, expected, check_exact=True)


def test_bool_series_to_numeric():
    input_series = pandas.Series([True, True, False, False, True, False, True], name="human",
                                 index=["Alpha", "Beta", "Gamma", "Delta", "Eta", "Mu", "Zeta"])
    expected = pandas.Series([1, 1, 0, 0, 1, 0, 1], name="human",
                             index=["Alpha", "Beta", "Gamma", "Delta", "Eta", "Mu", "Zeta"],
                             dtype=numpy.int64)

    actual = column.categorical_to_numeric(input_series)

    tm.assert_series_equal(actual, expected, check_exact=True)


def test_data_frame_to_numeric():
    a = numpy.r_[
        numpy.repeat(["large"], 10),
        numpy.repeat(["small"], 5),
        numpy.repeat(["tiny"], 13),
        numpy.repeat(["medium"], 3)]
    b = numpy.r_[
        numpy.repeat(["yes"], 8),
        numpy.repeat(["no"], 23)]

    rnd = numpy.random.RandomState(0)
    c = rnd.randn(len(a))

    input_df = pandas.DataFrame({"a_category": a,
                                 "a_binary": b,
                                 "a_number": c.copy()})

    a_num = numpy.r_[
        numpy.repeat([0], 10),
        numpy.repeat([2], 5),
        numpy.repeat([3], 13),
        numpy.repeat([1], 3)].astype(numpy.int64)
    b_num = numpy.r_[
        numpy.repeat([1], 8),
        numpy.repeat([0], 23)].astype(numpy.int64)
    expected = pandas.DataFrame({"a_category": a_num,
                                 "a_binary": b_num,
                                 "a_number": c.copy()})

    actual = column.categorical_to_numeric(input_df)

    tm.assert_frame_equal(actual, expected, check_exact=True)
