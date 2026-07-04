from contextlib import nullcontext as does_not_raise

from dataframe_test_utils import to_polars_dataframe
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
import pandas.testing as tm
import polars as pl
import pytest

from sksurv import column
from sksurv._dataframe import infer_column_semantics
from sksurv.preprocessing import OneHotEncoder
from sksurv.testing import FixtureParameterFactory, get_pandas_infer_string_context


class StandardizeCase(FixtureParameterFactory):
    @property
    def numeric_data(self):
        return pd.DataFrame(np.arange(50, dtype=float).reshape(10, 5))

    @property
    def expected(self):
        return pd.DataFrame(
            np.array(
                [
                    [-1.486301, -1.486301, -1.486301, -1.486301, -1.486301],
                    [-1.156012, -1.156012, -1.156012, -1.156012, -1.156012],
                    [-0.825723, -0.825723, -0.825723, -0.825723, -0.825723],
                    [-0.495434, -0.495434, -0.495434, -0.495434, -0.495434],
                    [-0.165145, -0.165145, -0.165145, -0.165145, -0.165145],
                    [0.165145, 0.165145, 0.165145, 0.165145, 0.165145],
                    [0.495434, 0.495434, 0.495434, 0.495434, 0.495434],
                    [0.825723, 0.825723, 0.825723, 0.825723, 0.825723],
                    [1.156012, 1.156012, 1.156012, 1.156012, 1.156012],
                    [1.486301, 1.486301, 1.486301, 1.486301, 1.486301],
                ]
            )
        )

    @property
    def non_numeric_data(self):
        data = pd.DataFrame.from_dict(
            {
                "q1": ["no", "no", "yes", "yes", "no", "no", None, "yes", "no", None],
                "q2": ["maybe", "no", "yes", "maybe", "yes", "no", None, "maybe", "no", "yes"],
                "q3": [1, 2, 1, 3, 1, 2, np.nan, np.nan, 3, 2],
            }
        )

        data = data.astype({"q3": "category"})
        return data

    @property
    def numeric_data_with_missing(self):
        return pd.DataFrame(
            {"a": [1.0, 2.0, np.nan, 4.0], "b": [1.0, 2.0, 3.0, 4.0]},
            index=[10, 20, 30, 40],
        )

    @property
    def expected_with_missing(self):
        return pd.DataFrame(
            {
                "a": [-0.872872, -0.218218, np.nan, 1.091089],
                "b": [-1.161895, -0.387298, 0.387298, 1.161895],
            },
            index=[10, 20, 30, 40],
        )

    def data_numeric(self):
        return self.numeric_data, self.expected

    def data_numeric_with_missing(self):
        return self.numeric_data_with_missing, self.expected_with_missing

    def data_float_numpy_array(self):
        return self.numeric_data.to_numpy(), self.expected

    def data_int_numpy_array(self):
        return self.numeric_data.to_numpy(dtype=int), self.expected

    def data_non_numeric(self):
        return self.non_numeric_data, self.non_numeric_data

    def data_non_numeric_numpy_array(self):
        data = self.non_numeric_data.to_numpy()
        return data, pd.DataFrame(data)

    def data_mixed(self):
        mixed_data_frame = pd.concat((self.numeric_data, self.non_numeric_data), axis=1)
        expected = pd.concat((self.expected, self.non_numeric_data), axis=1)
        return mixed_data_frame, expected

    def data_mixed_numpy_array(self):
        data, _ = self.data_mixed()
        data = data.to_numpy()
        return data, pd.DataFrame(data)


@pytest.mark.parametrize("in_data,expected", StandardizeCase().get_cases())
def test_standardize(in_data, expected):
    before = in_data.copy()
    result = column.standardize(in_data)

    # check that data wasn't modified inplace
    if isinstance(before, np.ndarray) and np.issubdtype(before.dtype, float):
        assert_array_equal(before, in_data, strict=True)
    elif isinstance(before, pd.DataFrame):
        tm.assert_frame_equal(before, in_data)

    if isinstance(result, np.ndarray):
        result = pd.DataFrame(result, columns=expected.columns)

    tm.assert_frame_equal(pd.isna(result), pd.isna(expected))
    tm.assert_frame_equal(result, expected)


def test_standardize_with_missing_no_std():
    data = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0]})

    result = column.standardize(data, with_std=False)

    expected = pd.DataFrame({"a": [-4.0 / 3, -1.0 / 3, np.nan, 5.0 / 3]})
    tm.assert_frame_equal(result, expected)


def test_standardize_numpy_without_std():
    data = np.arange(12, dtype=float).reshape(4, 3)

    result = column.standardize(data, with_std=False)

    expected = data - data.mean(axis=0)
    np.testing.assert_allclose(result, expected)


class CategoricalCases(FixtureParameterFactory):
    def _make_randn(self, shape):
        return np.random.default_rng(0).standard_normal(shape)

    @property
    def mixed_data_frame(self):
        a = np.r_[np.repeat(["large"], 10), np.repeat(["small"], 5), np.repeat(["tiny"], 13), np.repeat(["medium"], 3)]
        b = np.r_[np.repeat(["yes"], 8), np.repeat(["no"], 23)]

        c = self._make_randn(len(a))

        df = pd.DataFrame.from_dict({"a_category": a, "a_binary": b, "a_number": c.copy()})
        return df


class EncodeCategoricalCases(CategoricalCases):
    @property
    def binary_with_missing(self):
        inputs = np.r_[
            np.repeat(["yes"], 5),
            np.repeat([None], 10),
            np.repeat(["no"], 16),
        ]
        expected = np.r_[
            np.repeat([1.0], 5),
            np.repeat([np.nan], 10),
            np.repeat([0.0], 16),
        ]
        return inputs, expected

    def data_series_categorical(self):
        input_series = pd.Series(
            pd.Categorical.from_codes(
                [1, 1, 0, 2, 0, 1, 2, 1, 2, 0, 0, 1, 2, 2], ["small", "medium", "large"], ordered=False
            ),
            name="a_series",
        )

        expected_df = pd.DataFrame.from_dict(
            {
                "a_series=medium": np.array([1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0], dtype=float),
                "a_series=large": np.array([0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1], dtype=float),
            }
        )

        return input_series, {}, expected_df

    def data_case_1(self):
        input_df = self.mixed_data_frame

        eb = np.r_[np.repeat([1.0], 8), np.repeat([0.0], 23)]

        a_tiny = np.zeros(31, dtype=float)
        a_tiny[15:28] = 1

        a_small = np.zeros(31, dtype=float)
        a_small[10:15] = 1

        a_medium = np.zeros(31, dtype=float)
        a_medium[-3:] = 1

        expected_df = pd.DataFrame.from_dict(
            {
                "a_category=medium": a_medium,
                "a_category=small": a_small,
                "a_category=tiny": a_tiny,
                "a_binary=yes": eb,
                "a_number": input_df.loc[:, "a_number"].to_numpy(copy=True),
            }
        )

        return input_df, {}, expected_df

    def data_duplicate_index(self):
        input_df = self.mixed_data_frame.drop("a_binary", axis=1)
        input_df = pd.concat((input_df.iloc[:11], input_df.iloc[[11]], input_df.iloc[11:]))

        index = np.ceil(np.arange(0, input_df.shape[0] // 2, 0.5))
        input_df.index = index

        expected_df = pd.DataFrame(
            np.zeros((32, 3), dtype=float),
            index=index,
            columns=["a_category=medium", "a_category=small", "a_category=tiny"],
        )
        # tiny
        expected_df.iloc[16:29, 2] = 1
        # small
        expected_df.iloc[10:16, 1] = 1
        # medium
        expected_df.iloc[-3:, 0] = 1

        expected_df.loc[:, "a_number"] = input_df.loc[:, "a_number"].to_numpy(copy=True)

        return input_df, {}, expected_df

    def data_numeric(self):
        a = np.array([0, 1, 1, 0, 1, 0, 0, 1, 0, 1], dtype=object)
        b = np.array([1, 2, 1, 3, 2, 1, 3, 2, 3, 1], dtype=object)
        c = np.array(
            [1.0 / 128, 1.0 / 32, 1.0, 1.0 / 8, 1.0 / 32, 1.0, 1.0 / 128, 1.0 / 8, 1.0, 1.0 / 32], dtype=object
        )

        input_df = pd.DataFrame.from_dict({"a_binary_int": a.copy(), "a_three_int": b.copy(), "a_four_float": c.copy()})

        expected_df = pd.DataFrame(
            {
                "a_binary_int=1": a.astype(float),
                "a_three_int=2": (b == 2).astype(float),
                "a_three_int=3": (b == 3).astype(float),
                f"a_four_float={1.0 / 32}": (c == 1.0 / 32).astype(float),
                f"a_four_float={1.0 / 8}": (c == 1.0 / 8).astype(float),
                f"a_four_float={1.0}": (c == 1.0).astype(float),
            }
        )

        return input_df, {}, expected_df

    def data_with_missing(self):
        b, eb = self.binary_with_missing

        c = self._make_randn(len(b))

        input_df = pd.DataFrame({"a_binary": b, "a_number": c.copy()})

        expected_df = pd.DataFrame.from_dict({"a_binary=yes": eb, "a_number": c.copy()})

        return input_df, {}, expected_df

    def data_drop_all_missing(self):
        b, eb = self.binary_with_missing

        all_missing = pd.Series([np.nan] * len(b), dtype=object)

        input_df = pd.DataFrame({"a_binary": b, "bogus": all_missing})

        expected_df = pd.DataFrame({"a_binary=yes": eb})

        return input_df, {}, expected_df

    def data_retain_all_missing(self):
        input_df, _, expected_df = self.data_drop_all_missing()
        kwargs = {"allow_drop": False}
        expected_df.loc[:, "bogus"] = pd.Series([np.nan] * expected_df.shape[0], index=expected_df.index, dtype=object)

        return input_df, kwargs, expected_df

    def data_retain_only_one_level(self):
        b = np.r_[np.repeat(["yes"], 10)]

        input_df = pd.DataFrame({"categorical_col_with_only_one_level": b})
        expected_df = input_df.copy(deep=True)
        kwargs = {"allow_drop": False}

        return input_df, kwargs, expected_df


@pytest.mark.parametrize("infer_string_context", get_pandas_infer_string_context())
@pytest.mark.parametrize("make_data_fn", EncodeCategoricalCases().get_cases_func())
def test_encode_categorical(make_data_fn, infer_string_context):
    with infer_string_context:
        inputs, kwargs, expected_df = make_data_fn()
        actual_df = column.encode_categorical(inputs, **kwargs)
        tm.assert_frame_equal(actual_df.isnull(), expected_df.isnull())
        tm.assert_frame_equal(actual_df, expected_df, check_exact=True)


def test_encode_categorical_series_preserves_index():
    input_series = pd.Series(["a", "b", "a"], name="letter", index=["r0", "r0", "r2"])
    expected = pd.DataFrame({"letter=b": [0.0, 1.0, 0.0]}, index=input_series.index)

    actual = column.encode_categorical(input_series)

    tm.assert_frame_equal(actual, expected, check_exact=True)


def test_encode_categorical_drops_single_category_series_preserves_index():
    input_series = pd.Series(pd.Categorical(["a", "a", "a"]), name="c", index=["r0", "r1", "r2"])

    actual = column.encode_categorical(input_series)

    assert isinstance(actual, pd.DataFrame)
    assert actual.shape == (3, 0)
    assert list(actual.index) == ["r0", "r1", "r2"]


def test_series_numeric():
    input_series = pd.Series([0.5, 0.1, 10, 25, 3.8, 11, 2256, -1, -0.2, 3.14], name="a_series")

    with pytest.raises(TypeError, match="series must be of categorical dtype"):
        column.encode_categorical(input_series)


class CategoricalToNumeric(CategoricalCases):
    def data_categorical_series_to_numeric(self):
        input_series = pd.Series(
            ["a", "a", "b", "b", "b", "c"], name="Thr33", index=["Alpha", "Beta", "Gamma", "Delta", "Eta", "Mu"]
        )
        expected = pd.Series(
            [0, 0, 1, 1, 1, 2], name="Thr33", index=["Alpha", "Beta", "Gamma", "Delta", "Eta", "Mu"], dtype=np.int64
        )
        return input_series, expected

    def data_bool_series_to_numeric(self):
        input_series = pd.Series(
            [True, True, False, False, True, False, True],
            name="human",
            index=["Alpha", "Beta", "Gamma", "Delta", "Eta", "Mu", "Zeta"],
        )
        expected = pd.Series(
            [1, 1, 0, 0, 1, 0, 1],
            name="human",
            index=["Alpha", "Beta", "Gamma", "Delta", "Eta", "Mu", "Zeta"],
            dtype=np.int64,
        )
        return input_series, expected

    def data_object_numeric_series_to_numeric(self):
        input_series = pd.Series([1, 2, 1], name="x", dtype=object)
        expected = pd.Series([1, 2, 1], name="x", dtype=np.int64)
        return input_series, expected

    def data_frame_to_numeric(self):
        input_df = self.mixed_data_frame

        a_num = np.r_[np.repeat([0], 10), np.repeat([2], 5), np.repeat([3], 13), np.repeat([1], 3)].astype(np.int64)
        b_num = np.r_[np.repeat([1], 8), np.repeat([0], 23)].astype(np.int64)

        expected = pd.DataFrame.from_dict({"a_category": a_num, "a_binary": b_num})
        expected.loc[:, "a_number"] = input_df.loc[:, "a_number"].to_numpy(copy=True)

        return input_df, expected

    def data_object_numeric_frame_to_numeric(self):
        input_df = pd.DataFrame({"x": pd.Series([1, 2, 1], dtype=object)})
        expected = pd.DataFrame({"x": pd.Series([1, 2, 1], dtype=np.int64)})
        return input_df, expected


@pytest.mark.parametrize("infer_string_context", get_pandas_infer_string_context())
@pytest.mark.parametrize("make_data_fn", CategoricalToNumeric().get_cases_func())
def test_categorical_to_numeric(make_data_fn, infer_string_context):
    with infer_string_context:
        input_df, expected = make_data_fn()
        actual = column.categorical_to_numeric(input_df)

        if isinstance(expected, pd.Series):
            tm.assert_series_equal(actual, expected, check_exact=True)
        else:
            tm.assert_frame_equal(actual, expected, check_exact=True)


class TestStandardizeEmptyParity:
    @staticmethod
    def test_empty_polars_returns_empty():
        from sksurv.column import standardize

        out = standardize(pl.DataFrame())
        assert out.shape == (0, 0)

    @staticmethod
    def test_categorical_to_numeric_empty_polars_parity():
        import pandas as pd

        from sksurv.column import categorical_to_numeric

        assert categorical_to_numeric(pl.DataFrame()).shape == (0, 0)
        assert categorical_to_numeric(pd.DataFrame()).shape == (0, 0)

    @staticmethod
    def test_standardize_without_std_and_mixed_columns():
        from sksurv.column import standardize

        df = pl.DataFrame(
            {
                "x": [1.0, 2.0, 3.0],
                "label": pl.Series(["A", "B", "A"], dtype=pl.Enum(["A", "B"])),
            }
        )
        out = standardize(df, with_std=False)
        np.testing.assert_allclose(out["x"].to_numpy(), np.array([-1.0, 0.0, 1.0]))
        assert out["label"].to_list() == ["A", "B", "A"]


class TestCategoricalToNumericPandasParity:
    @staticmethod
    def test_string_numeric_dataframe_parses_ints():
        import pandas as pd

        from sksurv.column import categorical_to_numeric

        values = ["1", "2", "10"]
        pd_df = pd.DataFrame({"x": values})
        pd_out = categorical_to_numeric(pd_df)["x"].tolist()
        pl_out = categorical_to_numeric(to_polars_dataframe(pd_df))["x"].to_list()
        assert pd_out == pl_out == [1, 2, 10]

    @staticmethod
    def test_string_non_numeric_falls_back_to_codes():
        import pandas as pd

        from sksurv.column import categorical_to_numeric

        values = ["a", "b", "a"]
        pd_df = pd.DataFrame({"x": values})
        pd_out = categorical_to_numeric(pd_df)["x"].tolist()
        pl_out = categorical_to_numeric(to_polars_dataframe(pd_df))["x"].to_list()
        assert pd_out == pl_out

    @staticmethod
    def test_string_non_numeric_null_maps_to_nan():
        import pandas as pd

        from sksurv.column import categorical_to_numeric

        values = ["b", None, "a"]
        pd_df = pd.DataFrame({"x": values})
        pd_out = categorical_to_numeric(pd_df)["x"].to_numpy()
        pl_out = categorical_to_numeric(to_polars_dataframe(pd_df))["x"].to_numpy()
        np.testing.assert_allclose(pd_out, pl_out, equal_nan=True)

    @staticmethod
    def test_float_series_pass_through():
        from sksurv.column import categorical_to_numeric

        result = categorical_to_numeric(pl.Series("x", [1.2, 2.8]))
        assert result.to_list() == [1.2, 2.8]
        assert result.dtype == pl.Float64


class TestEncodeCategoricalExplicitColumnsParity:
    @staticmethod
    def test_explicit_numeric_column_polars_matches_pandas():
        import pandas as pd

        from sksurv.column import encode_categorical

        pd_df = pd.DataFrame({"x": [1, 2, 1], "z": [10, 20, 30]})
        pl_df = to_polars_dataframe(pd_df)
        pd_out = encode_categorical(pd_df, columns=["x"])
        pl_out = encode_categorical(pl_df, columns=["x"])
        assert list(pd_out.columns) == list(pl_out.columns)
        np.testing.assert_array_equal(pd_out.to_numpy(), pl_out.to_numpy(), strict=True)

    @staticmethod
    def test_explicit_boolean_column_polars_matches_pandas():
        import pandas as pd

        from sksurv.column import encode_categorical

        pd_df = pd.DataFrame({"b": [True, False, True, False, True]})
        pl_df = to_polars_dataframe(pd_df)
        pd_out = encode_categorical(pd_df, columns=["b"])
        pl_out = encode_categorical(pl_df, columns=["b"])
        assert list(pd_out.columns) == list(pl_out.columns) == ["b=True"]
        np.testing.assert_array_equal(pd_out.to_numpy(), pl_out.to_numpy(), strict=True)

    @staticmethod
    def test_explicit_numeric_column_preserves_value_ordering():
        import pandas as pd

        from sksurv.column import encode_categorical

        pd_df = pd.DataFrame({"x": [1, 2, 10, 1]})
        pd_out = encode_categorical(pd_df, columns=["x"])
        pl_out = encode_categorical(to_polars_dataframe(pd_df), columns=["x"])
        assert list(pd_out.columns) == list(pl_out.columns) == ["x=2", "x=10"]
        np.testing.assert_array_equal(pd_out.to_numpy(), pl_out.to_numpy(), strict=True)


class StandardizePolarsCases(FixtureParameterFactory):
    @property
    def numeric_data(self):
        return np.arange(50, dtype=float).reshape(10, 5)

    @property
    def expected_numeric(self):
        return np.array(
            [
                [-1.486301, -1.486301, -1.486301, -1.486301, -1.486301],
                [-1.156012, -1.156012, -1.156012, -1.156012, -1.156012],
                [-0.825723, -0.825723, -0.825723, -0.825723, -0.825723],
                [-0.495434, -0.495434, -0.495434, -0.495434, -0.495434],
                [-0.165145, -0.165145, -0.165145, -0.165145, -0.165145],
                [0.165145, 0.165145, 0.165145, 0.165145, 0.165145],
                [0.495434, 0.495434, 0.495434, 0.495434, 0.495434],
                [0.825723, 0.825723, 0.825723, 0.825723, 0.825723],
                [1.156012, 1.156012, 1.156012, 1.156012, 1.156012],
                [1.486301, 1.486301, 1.486301, 1.486301, 1.486301],
            ]
        )

    def _to_polars(self, data, columns):
        return pl.DataFrame(data, schema=columns)

    def data_polars_numeric(self):
        cols = [f"V{i}" for i in range(5)]
        df = self._to_polars(self.numeric_data, cols)
        expected = self._to_polars(self.expected_numeric, cols)
        return df, expected, does_not_raise()

    def data_polars_mixed(self):
        cols = [f"V{i}" for i in range(5)]
        numeric = self._to_polars(self.numeric_data, cols)
        cat = pl.Series("q", ["a", "b", "c", "a", "b", "c", "a", "b", "c", "a"], dtype=pl.Enum(["a", "b", "c"]))
        df = numeric.with_columns(cat)
        expected_numeric = self._to_polars(self.expected_numeric, cols)
        expected = expected_numeric.with_columns(cat)
        return df, expected, does_not_raise()

    @property
    def expected_standardized_missing(self):
        return [-0.872872, -0.218218, None, 1.091089]

    @property
    def expected_standardized_other(self):
        return [-1.161895, -0.387298, 0.387298, 1.161895]

    def data_polars_missing_null(self):
        # The statistics skip nulls (matching pandas' NaN handling) and the
        # output keeps null as null.
        df = pl.DataFrame({"a": [1.0, 2.0, None, 4.0], "b": [1.0, 2.0, 3.0, 4.0]})
        expected = pl.DataFrame({"a": self.expected_standardized_missing, "b": self.expected_standardized_other})
        return df, expected, does_not_raise()

    def data_polars_missing_nan(self):
        # Float NaN must be skipped by the statistics as well (it would
        # otherwise poison the whole column) and stays NaN in the output.
        df = pl.DataFrame({"a": [1.0, 2.0, float("nan"), 4.0], "b": [1.0, 2.0, 3.0, 4.0]})
        expected_a = [float("nan") if v is None else v for v in self.expected_standardized_missing]
        expected = pl.DataFrame({"a": expected_a, "b": self.expected_standardized_other})
        return df, expected, does_not_raise()


@pytest.mark.parametrize("in_data,expected,expected_error", StandardizePolarsCases().get_cases())
def test_standardize_polars(in_data, expected, expected_error):
    import polars.testing as pt

    with expected_error:
        result = column.standardize(in_data)
    if expected is not None:
        assert isinstance(result, pl.DataFrame), f"expected polars.DataFrame, got {type(result)!r}"
        pt.assert_frame_equal(result, expected, check_exact=False, abs_tol=1e-6)


@pytest.mark.parametrize(
    "polars_missing",
    [
        pl.Series([None] * 3, dtype=pl.Float64),
        [float("nan")] * 3,
    ],
    ids=["null", "nan"],
)
def test_standardize_all_missing_column_matches_pandas_via_numpy(polars_missing):
    import pandas as pd

    pd_out = column.standardize(pd.DataFrame({"a": [np.nan] * 3, "b": [1.0, 2.0, 3.0]}))
    # The all-missing column must be a float dtype, not Null, or standardize
    # would skip it as non-numeric.
    pl_out = column.standardize(pl.DataFrame({"a": polars_missing, "b": [1.0, 2.0, 3.0]}))

    # The dataframe-level representation differs (pandas NaN vs polars null) but
    # both normalize to NaN at the numpy boundary that feeds the estimators.
    np.testing.assert_allclose(pd_out.to_numpy(), pl_out.to_numpy(), equal_nan=True)


def test_standardize_all_missing_polars_column_stays_null():
    # Pin the polars dataframe-level behavior so a future change to NaN is noticed.
    out = column.standardize(pl.DataFrame({"a": pl.Series([None] * 3, dtype=pl.Float64), "b": [1.0, 2.0, 3.0]}))
    assert out["a"].dtype == pl.Float64
    assert out["a"].null_count() == 3


class EncodeCategoricalPolarsCases(FixtureParameterFactory):
    def _make_randn(self, shape):
        return np.random.default_rng(0).standard_normal(shape)

    @property
    def mixed_data_frame(self):
        a = np.r_[
            np.repeat(["large"], 10),
            np.repeat(["small"], 5),
            np.repeat(["tiny"], 13),
            np.repeat(["medium"], 3),
        ]
        b = np.r_[np.repeat(["yes"], 8), np.repeat(["no"], 23)]
        c = self._make_randn(len(a))
        return pl.DataFrame({"a_category": a, "a_binary": b, "a_number": c})

    def data_polars_mixed(self):
        df = self.mixed_data_frame
        eb = np.r_[np.repeat([1.0], 8), np.repeat([0.0], 23)]

        a_tiny = np.zeros(31, dtype=float)
        a_tiny[15:28] = 1
        a_small = np.zeros(31, dtype=float)
        a_small[10:15] = 1
        a_medium = np.zeros(31, dtype=float)
        a_medium[-3:] = 1

        expected = pl.DataFrame(
            {
                "a_category=medium": a_medium,
                "a_category=small": a_small,
                "a_category=tiny": a_tiny,
                "a_binary=yes": eb,
                "a_number": df.get_column("a_number"),
            }
        )
        return df, {}, expected


@pytest.mark.parametrize("inputs,kwargs,expected", EncodeCategoricalPolarsCases().get_cases())
def test_encode_categorical_polars(inputs, kwargs, expected):
    import polars.testing as pt

    actual = column.encode_categorical(inputs, **kwargs)
    assert isinstance(actual, pl.DataFrame), f"expected polars.DataFrame, got {type(actual)!r}"
    pt.assert_frame_equal(actual, expected, check_exact=False, abs_tol=1e-9)


def test_encode_categorical_polars_series_nominal():
    import polars.testing as pt

    s = pl.Series(
        "a_series",
        ["medium", "medium", "small", "large", "small", "medium", "large", "medium", "large", "small"],
        dtype=pl.Enum(["small", "medium", "large"]),
    )
    expected = pl.DataFrame(
        {
            "a_series=medium": np.array([1, 1, 0, 0, 0, 1, 0, 1, 0, 0], dtype=float),
            "a_series=large": np.array([0, 0, 0, 1, 0, 0, 1, 0, 1, 0], dtype=float),
        }
    )
    actual = column.encode_categorical(s)
    pt.assert_frame_equal(actual, expected, check_exact=False, abs_tol=1e-9)


def test_encode_categorical_polars_series_numeric_rejected():
    s = pl.Series("a_series", [0.5, 0.1, 10.0, 25.0, 3.8, 11.0])
    with pytest.raises(TypeError):
        column.encode_categorical(s)


def test_encode_categorical_polars_dataframe_drop_emits_warning(caplog):
    df = pl.DataFrame({"single": pl.Series(["only", "only"], dtype=pl.Enum(["only"]))})
    result = column.encode_categorical(df)
    assert result.shape == (0, 0)
    assert "dropped categorical variable 'single'" in caplog.text


def test_encode_categorical_polars_single_category_series_drop_policy():
    s = pl.Series("a_series", ["only", "only"], dtype=pl.Enum(["only"]))

    dropped = column.encode_categorical(s)
    assert dropped.shape == (0, 0)

    preserved = column.encode_categorical(s, allow_drop=False)
    assert preserved.to_list() == ["only", "only"]


def test_categorical_to_numeric_polars_bool_series():
    result = column.categorical_to_numeric(pl.Series("flag", [True, False, True]))
    assert result.to_list() == [1, 0, 1]
    assert result.dtype == pl.Int64


def test_categorical_to_numeric_polars_unsupported_column_passes_through():
    df = pl.DataFrame({"items": [[1], [2]]})
    result = column.categorical_to_numeric(df)
    assert result.to_dict(as_series=False) == {"items": [[1], [2]]}


def test_categorical_to_numeric_polars_numeric_string_series():
    result = column.categorical_to_numeric(pl.Series("digits", ["1", "2", "10"]))
    assert result.to_list() == [1, 2, 10]


class CategoricalToNumericPolarsCases(FixtureParameterFactory):
    def _make_randn(self, shape):
        return np.random.default_rng(0).standard_normal(shape)

    def data_polars_mixed(self):
        n = 16
        cat = pl.Series(
            "a_cat",
            np.repeat(["b", "a", "c", "a"], n // 4),
            dtype=pl.Enum(["a", "b", "c"]),
        )
        boolean = pl.Series("a_bool", np.r_[np.repeat([True], 6), np.repeat([False], 10)])
        numeric = pl.Series("a_num", self._make_randn(n))
        df = pl.DataFrame({"a_cat": cat, "a_bool": boolean, "a_num": numeric})

        cat_codes = np.array([1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0], dtype=np.int64)
        bool_codes = np.r_[np.repeat([1], 6), np.repeat([0], 10)].astype(np.int64)
        expected = pl.DataFrame(
            {
                "a_cat": cat_codes,
                "a_bool": bool_codes,
                "a_num": numeric,
            }
        )
        return df, expected


@pytest.mark.parametrize("inputs,expected", CategoricalToNumericPolarsCases().get_cases())
def test_categorical_to_numeric_polars(inputs, expected):
    import polars.testing as pt

    actual = column.categorical_to_numeric(inputs)
    assert isinstance(actual, pl.DataFrame), f"expected polars.DataFrame, got {type(actual)!r}"
    pt.assert_frame_equal(actual, expected, check_exact=False, abs_tol=1e-9)


class TestPolarsCategoricalGlobalPoolBug:
    """Polars ``pl.Categorical`` categories must be column-scoped."""

    @staticmethod
    def test_infer_isolates_categories_per_column():
        df = pl.DataFrame(
            {
                "size": pl.Series(["medium", "small", "large", "xlarge", "small"], dtype=pl.Categorical),
                "answer": pl.Series(["yes", "no", "yes", "yes", "no"], dtype=pl.Categorical),
            }
        )
        sem_size = infer_column_semantics(df.get_column("size"))
        sem_answer = infer_column_semantics(df.get_column("answer"))
        assert set(sem_size.categories) == {
            "medium",
            "small",
            "large",
            "xlarge",
        }, f"polars Categorical leak: {sem_size.categories}"
        assert set(sem_answer.categories) == {"yes", "no"}, f"polars Categorical leak: {sem_answer.categories}"

    @staticmethod
    def test_encode_categorical_isolates_categories_per_column():
        df = pl.DataFrame(
            {
                "size": pl.Series(["medium", "small", "large", "xlarge", "small"], dtype=pl.Categorical),
                "answer": pl.Series(["yes", "no", "yes", "yes", "no"], dtype=pl.Categorical),
            }
        )
        encoded = column.encode_categorical(df)
        size_cols = [c for c in encoded.columns if c.startswith("size=")]
        answer_cols = [c for c in encoded.columns if c.startswith("answer=")]
        assert len(size_cols) == 3, f"expected 3 size= columns, got {size_cols}"
        assert len(answer_cols) == 1, f"expected 1 answer= column, got {answer_cols}"
        for c in size_cols:
            assert c.split("=", 1)[1] in {"medium", "small", "large", "xlarge"}
        for c in answer_cols:
            assert c.split("=", 1)[1] in {"yes", "no"}

    @staticmethod
    def test_one_hot_encoder_categories_isolated():
        df = pl.DataFrame(
            {
                "size": pl.Series(["medium", "small", "large", "xlarge", "small"], dtype=pl.Categorical),
                "answer": pl.Series(["yes", "no", "yes", "yes", "no"], dtype=pl.Categorical),
            }
        )
        enc = OneHotEncoder()
        enc.fit(df)
        assert set(enc.categories_["size"]) == {
            "medium",
            "small",
            "large",
            "xlarge",
        }, f"size leak: {enc.categories_['size'].tolist()}"
        assert set(enc.categories_["answer"]) == {"yes", "no"}, f"answer leak: {enc.categories_['answer'].tolist()}"

    @staticmethod
    def test_pl_enum_categories_still_dtype_based():
        s = pl.Series("x", ["mid", "low", "high"], dtype=pl.Enum(["low", "mid", "high"]))
        sem = infer_column_semantics(s)
        assert sem.kind == "nominal"
        assert sem.categories == ("low", "mid", "high")
        assert sem.ordered is False


class TestPolarsCategoryOrderPolicy:
    @staticmethod
    def test_enum_preserves_declared_order():
        s = pl.Series("grade", ["mid", "low", "high"], dtype=pl.Enum(["low", "mid", "high"]))
        sem = infer_column_semantics(s)
        assert sem.categories == ("low", "mid", "high")

    @staticmethod
    def test_categorical_uses_sorted_observed_values():
        s = pl.Series("grade", ["mid", "low", "high", "low"], dtype=pl.Categorical)
        sem = infer_column_semantics(s)
        assert sem.categories == ("high", "low", "mid")

    @staticmethod
    def test_string_uses_sorted_observed_values():
        s = pl.Series("grade", ["mid", "low", "high", None])
        sem = infer_column_semantics(s)
        assert sem.categories == ("high", "low", "mid")

    @staticmethod
    def test_one_hot_encoder_follows_category_order_policy():
        df = pl.DataFrame({"grade": pl.Series(["mid", "low", "high"], dtype=pl.Categorical)})
        enc = OneHotEncoder().fit(df)
        assert enc.categories_["grade"].tolist() == ["high", "low", "mid"]
        assert enc.get_feature_names_out().tolist() == ["grade=low", "grade=mid"]


class TestCategoricalDataInferredParity:
    @staticmethod
    def test_categorical_to_numeric_pl_categorical_matches_pandas():
        import pandas as pd

        from sksurv.column import categorical_to_numeric

        values = ["banana", "apple", "cherry", "apple", "banana"]
        df_pd = pd.DataFrame({"fruit": pd.Categorical(values)})
        df_pl = pl.DataFrame({"fruit": pl.Series(values, dtype=pl.Categorical)})
        out_pd = categorical_to_numeric(df_pd).to_numpy()
        out_pl = categorical_to_numeric(df_pl).to_numpy()
        np.testing.assert_array_equal(out_pd, out_pl, strict=True)

    @staticmethod
    def test_categorical_to_numeric_enum_matches_pandas_explicit_order():
        import pandas as pd

        from sksurv.column import categorical_to_numeric

        values = ["c", "a", "b", "c"]
        categories = ["c", "a", "b"]
        df_pd = pd.DataFrame({"x": pd.Categorical(values, categories=categories)})
        df_pl = pl.DataFrame({"x": pl.Series("x", values, dtype=pl.Enum(categories))})
        out_pd = categorical_to_numeric(df_pd).to_numpy()
        out_pl = categorical_to_numeric(df_pl).to_numpy()
        np.testing.assert_array_equal(out_pd, out_pl, strict=True)

    @staticmethod
    def test_encode_categorical_pl_categorical_matches_pandas():
        import pandas as pd

        from sksurv.column import encode_categorical

        values = ["banana", "apple", "cherry", "apple", "banana"]
        df_pd = pd.DataFrame({"fruit": pd.Categorical(values)})
        df_pl = pl.DataFrame({"fruit": pl.Series(values, dtype=pl.Categorical)})
        out_pd = encode_categorical(df_pd).to_numpy()
        out_pl = encode_categorical(df_pl).to_numpy()
        np.testing.assert_array_equal(out_pd, out_pl, strict=True)


class TestLazyFramePaths:
    @staticmethod
    def _eager_lazy_pair():
        df = pl.DataFrame(
            {
                "age": [40.0, 50.0, 60.0, 70.0],
                "grade": pl.Series(["I", "II", "III", "I"], dtype=pl.Enum(["I", "II", "III", "IV"])),
            }
        )
        return df, df.lazy()

    @staticmethod
    def test_standardize_lazyframe_rejected():
        from sksurv.column import standardize

        _, df_lazy = TestLazyFramePaths._eager_lazy_pair()
        with pytest.raises(TypeError, match=r"polars\.LazyFrame is not supported"):
            standardize(df_lazy)

    @staticmethod
    def test_categorical_to_numeric_lazyframe_rejected():
        from sksurv.column import categorical_to_numeric

        _, df_lazy = TestLazyFramePaths._eager_lazy_pair()
        with pytest.raises(TypeError, match=r"polars\.LazyFrame is not supported"):
            categorical_to_numeric(df_lazy)

    @staticmethod
    def test_encode_categorical_lazyframe_rejected():
        from sksurv.column import encode_categorical

        _, df_lazy = TestLazyFramePaths._eager_lazy_pair()
        with pytest.raises(TypeError, match=r"polars\.LazyFrame is not supported"):
            encode_categorical(df_lazy)

    @staticmethod
    def test_encode_categorical_polars_series():
        from sksurv.column import encode_categorical

        df, _ = TestLazyFramePaths._eager_lazy_pair()
        out = encode_categorical(df["grade"])
        assert hasattr(out, "shape")
        assert out.shape[0] == df.shape[0]
