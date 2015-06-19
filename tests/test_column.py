from numpy.testing import TestCase, run_module_suite, assert_array_almost_equal, assert_array_equal
import pandas.util.testing as tm
import pandas
import numpy

from survival import column

NUMERIC_DATA_FRAME = pandas.DataFrame(numpy.arange(50).reshape(10, 5))

NON_NUMERIC_DATA_FRAME = pandas.DataFrame({'q1': ['no', 'no', 'yes', 'yes', 'no', 'no', None, 'yes', 'no', None],
                                           'q2': ['maybe', 'no', 'yes', 'maybe', 'yes', 'no', None, 'maybe', 'no',
                                                  'yes'],
                                           'q3': [1, 2, 1, 3, 1, 2, numpy.nan, numpy.nan, 3, 2]})

MIXED_DATA_FRAME = pandas.concat((NUMERIC_DATA_FRAME, NON_NUMERIC_DATA_FRAME), axis=1)

NON_NUMERIC_DATA_FRAME['q3'] = NON_NUMERIC_DATA_FRAME['q3'].astype('category')
MIXED_DATA_FRAME['q3'] = MIXED_DATA_FRAME['q3'].astype('category')


class TestColumn(TestCase):
    def test_standardize_numeric(self):
        result = column.standardize(NUMERIC_DATA_FRAME)

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

        self.assertTrue(isinstance(result, pandas.DataFrame))
        assert_array_almost_equal(expected, result)

    def test_standardize_non_numeric(self):
        result = column.standardize(NON_NUMERIC_DATA_FRAME)

        self.assertTrue(isinstance(result, pandas.DataFrame))
        tm.assert_frame_equal(NON_NUMERIC_DATA_FRAME, result)

    def test_standardize_mixed(self):
        result = column.standardize(MIXED_DATA_FRAME)

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

        self.assertTrue(isinstance(result, pandas.DataFrame))
        assert_array_almost_equal(expected, result.iloc[:, :NUMERIC_DATA_FRAME.shape[1]].values)

        tm.assert_frame_equal(NON_NUMERIC_DATA_FRAME, result.iloc[:, NUMERIC_DATA_FRAME.shape[1]:])

    def test_standardize_numpy_array(self):
        result = column.standardize(MIXED_DATA_FRAME.values)

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

        self.assertTrue(isinstance(result, numpy.ndarray))
        assert_array_almost_equal(expected, result[:, :NUMERIC_DATA_FRAME.shape[1]])

        assert_array_equal(pandas.isnull(NON_NUMERIC_DATA_FRAME),
                           pandas.isnull(result[:, NUMERIC_DATA_FRAME.shape[1]:]))

        non_nan_idx = [0, 1, 2, 3, 4, 5, 8, 9]

        assert_array_equal(NON_NUMERIC_DATA_FRAME.iloc[non_nan_idx, :].values,
                           result[:, NUMERIC_DATA_FRAME.shape[1]:][non_nan_idx, :])


class TestEncodeCategorical(TestCase):
    def test_case1(self):
        a = numpy.concatenate((
            numpy.repeat(["large"], 10),
            numpy.repeat(["small"], 5),
            numpy.repeat(["tiny"], 13),
            numpy.repeat(["medium"], 3)))
        b = numpy.concatenate((
            numpy.repeat(["yes"], 8),
            numpy.repeat(["no"], 23)))

        rnd = numpy.random.RandomState(0)
        c = rnd.randn(len(a))

        df = pandas.DataFrame({"a_category": a,
                               "a_binary": b,
                               "a_number": c.copy()})

        actual_df = column.encode_categorical(df)

        eb = numpy.concatenate((
            numpy.repeat([1.], 8),
            numpy.repeat([0.], 23)))

        a_tiny = numpy.zeros(31, dtype=float)
        a_tiny[15:28] = 1

        a_small = numpy.zeros(31, dtype=float)
        a_small[10:15] = 1

        a_medium = numpy.zeros(31, dtype=float)
        a_medium[-3:] = 1

        expected_df = pandas.DataFrame({"a_number": c.copy(),
                                        "a_binary=yes": eb,
                                        "a_category=medium": a_medium,
                                        "a_category=small": a_small,
                                        "a_category=tiny": a_tiny})

        self.assertTupleEqual(actual_df.shape, expected_df.shape)
        tm.assert_frame_equal(actual_df, expected_df, check_exact=True)

    def test_with_missing(self):
        b = numpy.concatenate((
            numpy.repeat(["yes"], 5),
            numpy.repeat([None], 10),
            numpy.repeat(["no"], 16)))

        rnd = numpy.random.RandomState(0)
        c = rnd.randn(len(b))

        df = pandas.DataFrame({"a_binary": b,
                               "a_number": c.copy()})

        actual_df = column.encode_categorical(df)

        eb = numpy.concatenate((
            numpy.repeat([1.], 5),
            numpy.repeat([numpy.nan], 10),
            numpy.repeat([0.], 16)))

        expected_df = pandas.DataFrame({"a_number": c.copy(),
                                        "a_binary=yes": eb})

        self.assertTupleEqual(actual_df.shape, expected_df.shape)
        tm.assert_frame_equal(actual_df.isnull(), expected_df.isnull())
        tm.assert_frame_equal(actual_df.dropna(), expected_df.dropna(), check_exact=True)

    def test_drop_all_missing(self):
        b = numpy.concatenate((
            numpy.repeat(["yes"], 5),
            numpy.repeat([None], 10),
            numpy.repeat(["no"], 16)))

        all_missing = numpy.repeat([None], len(b))

        df = pandas.DataFrame({"a_binary": b,
                               "bogus": all_missing})

        actual_df = column.encode_categorical(df)

        eb = numpy.concatenate((
            numpy.repeat([1.], 5),
            numpy.repeat([numpy.nan], 10),
            numpy.repeat([0.], 16)))

        expected_df = pandas.DataFrame({"a_binary=yes": eb})

        self.assertTupleEqual(actual_df.shape, expected_df.shape)
        tm.assert_frame_equal(actual_df.isnull(), expected_df.isnull())
        tm.assert_frame_equal(actual_df.dropna(), expected_df.dropna(), check_exact=True)

    def test_retain_all_missing(self):
        b = numpy.concatenate((
            numpy.repeat(["yes"], 5),
            numpy.repeat([None], 10),
            numpy.repeat(["no"], 16)))

        all_missing = numpy.repeat([None], len(b))

        df = pandas.DataFrame({"a_binary": b,
                               "bogus": all_missing})

        actual_df = column.encode_categorical(df, allow_drop=False)

        eb = numpy.concatenate((
            numpy.repeat([1.], 5),
            numpy.repeat([numpy.nan], 10),
            numpy.repeat([0.], 16)))

        expected_df = pandas.DataFrame({"a_binary=yes": eb,
                                        "bogus": all_missing.copy()})

        self.assertTupleEqual(actual_df.shape, expected_df.shape)
        tm.assert_frame_equal(actual_df.isnull(), expected_df.isnull())
        tm.assert_frame_equal(actual_df.dropna(), expected_df.dropna(), check_exact=True)


class TestCategoricalToNumeric(TestCase):

    def test_series(self):
        input_series = pandas.Series(["a", "a", "b", "b", "b", "c"], name="Thr33",
                               index=["Alpha", "Beta", "Gamma", "Delta", "Eta", "Mu"])
        expected = pandas.Series([0, 0, 1, 1, 1, 2], name="Thr33",
                                 index=["Alpha", "Beta", "Gamma", "Delta", "Eta", "Mu"])

        actual = column.categorical_to_numeric(input_series)

        tm.assert_series_equal(actual, expected, check_exact=True)

    def test_data_frame(self):
        a = numpy.concatenate((
            numpy.repeat(["large"], 10),
            numpy.repeat(["small"], 5),
            numpy.repeat(["tiny"], 13),
            numpy.repeat(["medium"], 3)))
        b = numpy.concatenate((
            numpy.repeat(["yes"], 8),
            numpy.repeat(["no"], 23)))

        rnd = numpy.random.RandomState(0)
        c = rnd.randn(len(a))

        input_df = pandas.DataFrame({"a_category": a,
                                     "a_binary": b,
                                     "a_number": c.copy()})

        a_num = numpy.concatenate((
            numpy.repeat([0], 10),
            numpy.repeat([2], 5),
            numpy.repeat([3], 13),
            numpy.repeat([1], 3)))
        b_num = numpy.concatenate((
            numpy.repeat([1], 8),
            numpy.repeat([0], 23)))
        expected = pandas.DataFrame({"a_category": a_num,
                                     "a_binary": b_num,
                                     "a_number": c.copy()})

        actual = column.categorical_to_numeric(input_df)

        tm.assert_frame_equal(actual, expected, check_exact=True)


if __name__ == '__main__':
    run_module_suite()
