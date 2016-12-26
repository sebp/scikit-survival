from numpy.testing import TestCase, run_module_suite, assert_array_almost_equal, assert_array_equal
import pandas.util.testing as tm
import pandas
import numpy

from sksurv.util import safe_concat


class TestUtil(TestCase):
    def test_concat_numeric(self):
        rnd = numpy.random.RandomState(14)
        a = pandas.Series(rnd.randn(100), name="col_A")
        b = pandas.Series(rnd.randn(100), name="col_B")

        expected_df = pandas.DataFrame.from_items(
            [(a.name, a), (b.name, b)]
        )

        actual_df = safe_concat((a, b), axis=1)

        tm.assert_frame_equal(actual_df, expected_df)

    def test_concat_numeric_categorical(self):
        rnd = numpy.random.RandomState(14)
        a = pandas.Series(rnd.randn(100), name="col_A")
        b = pandas.Series(pandas.Categorical.from_codes(
            rnd.binomial(4, 0.6, 100), ["C1", "C2", "C3", "C4", "C5"]), name="col_B")

        expected_df = pandas.DataFrame.from_items(
            [(a.name, a), (b.name, b)]
        )

        actual_df = safe_concat((a, b), axis=1)

        tm.assert_frame_equal(actual_df, expected_df)

    def test_concat_categorical(self):
        rnd = numpy.random.RandomState(14)
        a = pandas.DataFrame.from_items([
            ("col_A", pandas.Series(pandas.Categorical.from_codes(
                rnd.binomial(2, 0.6, 100), ["C1", "C2", "C3"]), name="col_A")),
            ("col_B", rnd.randn(100))])
        b = pandas.DataFrame.from_items([
            ("col_A", pandas.Series(pandas.Categorical.from_codes(
                rnd.binomial(2, 0.2, 100), ["C1", "C2", "C3"]), name="col_A")),
            ("col_B", rnd.randn(100))])

        expected_series = pandas.DataFrame.from_items([
            ("col_A", pandas.Series(pandas.Categorical.from_codes(
                numpy.concatenate((a.col_A.cat.codes.values, b.col_A.cat.codes.values)),
                ["C1", "C2", "C3", "C4", "C5"]
            ))),
            ("col_B", numpy.concatenate((a.col_B.values, b.col_B.values)))
        ])
        expected_series.index = pandas.Index(a.index.tolist() + b.index.tolist())

        actual_series = safe_concat((a, b), axis=0)

        tm.assert_frame_equal(actual_series, expected_series)

    def test_concat_categorical_mismatch(self):
        rnd = numpy.random.RandomState(14)
        a = pandas.DataFrame.from_items([
            ("col_A", pandas.Series(pandas.Categorical.from_codes(
                rnd.binomial(2, 0.6, 100), ["C1", "C2", "C3"]), name="col_A")),
            ("col_B", rnd.randn(100))])
        b = pandas.DataFrame.from_items([
            ("col_A", pandas.Series(pandas.Categorical.from_codes(
                rnd.binomial(3, 0.6, 100), ["C1", "C2", "C3", "C4"]), name="col_A")),
            ("col_B", rnd.randn(100))])

        self.assertRaisesRegex(ValueError, "categories for column col_A do not match",
                               safe_concat, (a, b), axis=0)

    def test_concat_dataframe_numeric_categorical(self):
        rnd = numpy.random.RandomState(14)
        numeric_df = pandas.DataFrame.from_items(
            [("col_A", rnd.randn(100)), ("col_B", rnd.randn(100))]
        )

        cat_series = pandas.Series(pandas.Categorical.from_codes(
            rnd.binomial(4, 0.6, 100), ["C1", "C2", "C3", "C4", "C5"]), name="col_C")

        expected_df = numeric_df.copy()
        expected_df["col_C"] = cat_series

        actual_df = safe_concat((numeric_df, cat_series), axis=1)

        tm.assert_frame_equal(actual_df, expected_df)

    def test_concat_duplicate_columns(self):
        rnd = numpy.random.RandomState(14)
        numeric_df = pandas.DataFrame.from_items([
            ("col_N", rnd.randn(100)), ("col_B", rnd.randn(100)),
            ("col_A", pandas.Series(pandas.Categorical.from_codes(
                rnd.binomial(4, 0.2, 100), ["C1", "C2", "C3", "C4", "C5"]), name="col_A")),
        ])

        cat_df = pandas.DataFrame.from_items([
            ("col_A", pandas.Series(pandas.Categorical.from_codes(
                rnd.binomial(4, 0.6, 100), ["C1", "C2", "C3", "C4", "C5"]), name="col_A")),
            ("col_C", pandas.Series(pandas.Categorical.from_codes(
                rnd.binomial(1, 0.6, 100), ["Yes", "No"]), name="col_C")),
        ])

        self.assertRaisesRegex(ValueError, "duplicate columns col_A",
                               safe_concat, (numeric_df, cat_df), axis=1)


if __name__ == '__main__':
    run_module_suite()
