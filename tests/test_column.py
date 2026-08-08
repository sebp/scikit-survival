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


def _standardize_numeric_input():
    return np.arange(50, dtype=float).reshape(10, 5)


def _standardize_expected_output():
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


class StandardizeCase(FixtureParameterFactory):
    """Inputs that only exist for pandas or numpy."""

    @property
    def numeric_data(self):
        return pd.DataFrame(_standardize_numeric_input())

    @property
    def expected(self):
        return pd.DataFrame(_standardize_expected_output())

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

    def data_numeric_with_missing_keeps_index(self):
        data = pd.DataFrame(
            {"a": [1.0, 2.0, np.nan, 4.0], "b": [1.0, 2.0, 3.0, 4.0]},
            index=[10, 20, 30, 40],
        )
        expected = pd.DataFrame(
            {
                "a": [-0.872872, -0.218218, np.nan, 1.091089],
                "b": [-1.161895, -0.387298, 0.387298, 1.161895],
            },
            index=[10, 20, 30, 40],
        )
        return data, expected

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


class StandardizeFrameCases(FixtureParameterFactory):
    @property
    def column_names(self):
        return [f"V{i}" for i in range(5)]

    def _make_numeric(self, backend, values):
        return backend.make_frame({name: values[:, i] for i, name in enumerate(self.column_names)})

    def data_numeric(self, backend):
        data = self._make_numeric(backend, _standardize_numeric_input())
        expected = self._make_numeric(backend, _standardize_expected_output())
        return data, expected

    def data_numeric_with_missing(self, backend):
        data = backend.make_frame({"a": [1.0, 2.0, None, 4.0], "b": [1.0, 2.0, 3.0, 4.0]})
        expected = backend.make_frame(
            {"a": [-0.872872, -0.218218, None, 1.091089], "b": [-1.161895, -0.387298, 0.387298, 1.161895]}
        )
        return data, expected

    def data_mixed(self, backend):
        q_values = ["a", "b", "c", "a", "b", "c", "a", "b", "c", "a"]
        categories = {"q": ["a", "b", "c"]}

        input_array = _standardize_numeric_input()
        data_columns = {name: input_array[:, i] for i, name in enumerate(self.column_names)}
        data_columns["q"] = q_values
        expected_array = _standardize_expected_output()
        expected_columns = {name: expected_array[:, i] for i, name in enumerate(self.column_names)}
        expected_columns["q"] = q_values

        data = backend.make_frame(data_columns, categories=categories)
        expected = backend.make_frame(expected_columns, categories=categories)
        return data, expected


@pytest.mark.parametrize("case", StandardizeFrameCases().get_cases_func())
def test_standardize_frame(case, dataframe_backend):
    in_data, expected = case(dataframe_backend)
    before = dataframe_backend.copy_frame(in_data)

    result = column.standardize(in_data)

    # the input frame must not be modified in place
    dataframe_backend.assert_frame_equal(in_data, before)
    dataframe_backend.assert_frame_equal(result, expected, abs_tol=1e-6)


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


def _mixed_categorical_values():
    a = np.r_[np.repeat(["large"], 10), np.repeat(["small"], 5), np.repeat(["tiny"], 13), np.repeat(["medium"], 3)]
    b = np.r_[np.repeat(["yes"], 8), np.repeat(["no"], 23)]
    c = np.random.default_rng(0).standard_normal(len(a))
    return a, b, c


class EncodeCategoricalCases(FixtureParameterFactory):
    def _make_randn(self, shape):
        return np.random.default_rng(0).standard_normal(shape)

    @property
    def mixed_data_frame(self):
        a, b, c = _mixed_categorical_values()
        df = pd.DataFrame.from_dict({"a_category": a, "a_binary": b, "a_number": c.copy()})
        return df

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


class EncodeCategoricalFrameCases(FixtureParameterFactory):
    def data_mixed(self, backend):
        a, b, c = _mixed_categorical_values()
        data = backend.make_frame({"a_category": a, "a_binary": b, "a_number": c})

        eb = np.r_[np.repeat([1.0], 8), np.repeat([0.0], 23)]
        a_tiny = np.zeros(31, dtype=float)
        a_tiny[15:28] = 1
        a_small = np.zeros(31, dtype=float)
        a_small[10:15] = 1
        a_medium = np.zeros(31, dtype=float)
        a_medium[-3:] = 1

        expected = backend.make_frame(
            {
                "a_category=medium": a_medium,
                "a_category=small": a_small,
                "a_category=tiny": a_tiny,
                "a_binary=yes": eb,
                "a_number": c,
            }
        )
        return data, {}, expected


@pytest.mark.parametrize("case", EncodeCategoricalFrameCases().get_cases_func())
def test_encode_categorical_frame(case, dataframe_backend_with_pandas_options):
    backend = dataframe_backend_with_pandas_options
    data, kwargs, expected = case(backend)

    actual = column.encode_categorical(data, **kwargs)

    backend.assert_frame_equal(actual, expected)


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


class CategoricalToNumeric(FixtureParameterFactory):
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


class CategoricalToNumericFrameCases(FixtureParameterFactory):
    def data_mixed(self, backend):
        a, b, c = _mixed_categorical_values()
        flags = np.r_[np.repeat([True], 6), np.repeat([False], 25)]
        data = backend.make_frame({"a_category": a, "a_binary": b, "a_flag": flags, "a_number": c})

        a_num = np.r_[np.repeat([0], 10), np.repeat([2], 5), np.repeat([3], 13), np.repeat([1], 3)].astype(np.int64)
        b_num = np.r_[np.repeat([1], 8), np.repeat([0], 23)].astype(np.int64)
        expected = backend.make_frame(
            {
                "a_category": a_num,
                "a_binary": b_num,
                "a_flag": flags.astype(np.int64),
                "a_number": c,
            }
        )
        return data, expected


@pytest.mark.parametrize("case", CategoricalToNumericFrameCases().get_cases_func())
def test_categorical_to_numeric_frame(case, dataframe_backend_with_pandas_options):
    backend = dataframe_backend_with_pandas_options
    data, expected = case(backend)

    actual = column.categorical_to_numeric(data)

    backend.assert_frame_equal(actual, expected)


def test_standardize_empty_frame_returns_empty(dataframe_backend):
    out = column.standardize(dataframe_backend.make_frame({}))
    assert out.shape == (0, 0)


def test_categorical_to_numeric_empty_frame_returns_empty(dataframe_backend):
    out = column.categorical_to_numeric(dataframe_backend.make_frame({}))
    assert out.shape == (0, 0)


def test_standardize_without_std_keeps_categorical(dataframe_backend):
    df = dataframe_backend.make_frame(
        {"x": [1.0, 2.0, 3.0], "label": ["A", "B", "A"]}, categories={"label": ["A", "B"]}
    )

    out = column.standardize(df, with_std=False)

    np.testing.assert_allclose(out["x"].to_numpy(), np.array([-1.0, 0.0, 1.0]))
    assert list(out["label"]) == ["A", "B", "A"]


class TestCategoricalToNumericPandasParity:
    @staticmethod
    def test_string_numeric_dataframe_parses_ints():

        from sksurv.column import categorical_to_numeric

        values = ["1", "2", "10"]
        pd_out = categorical_to_numeric(pd.DataFrame({"x": values}))["x"].tolist()
        pl_out = categorical_to_numeric(pl.DataFrame({"x": values}))["x"].to_list()
        assert pd_out == pl_out == [1, 2, 10]

    @staticmethod
    def test_string_non_numeric_falls_back_to_codes():

        from sksurv.column import categorical_to_numeric

        values = ["a", "b", "a"]
        pd_out = categorical_to_numeric(pd.DataFrame({"x": values}))["x"].tolist()
        pl_out = categorical_to_numeric(pl.DataFrame({"x": values}))["x"].to_list()
        assert pd_out == pl_out

    @staticmethod
    def test_string_non_numeric_null_maps_to_nan():

        from sksurv.column import categorical_to_numeric

        values = ["b", None, "a"]
        pd_out = categorical_to_numeric(pd.DataFrame({"x": values}))["x"].to_numpy()
        pl_out = categorical_to_numeric(pl.DataFrame({"x": values}))["x"].to_numpy()
        np.testing.assert_allclose(pd_out, pl_out, equal_nan=True)


class TestEncodeCategoricalExplicitColumnsParity:
    @staticmethod
    def test_explicit_numeric_column_polars_matches_pandas():

        from sksurv.column import encode_categorical

        data = {"x": [1, 2, 1], "z": [10, 20, 30]}
        pd_out = encode_categorical(pd.DataFrame(data), columns=["x"])
        pl_out = encode_categorical(pl.DataFrame(data), columns=["x"])
        assert list(pd_out.columns) == list(pl_out.columns)
        np.testing.assert_array_equal(pd_out.to_numpy(), pl_out.to_numpy(), strict=True)

    @staticmethod
    def test_explicit_boolean_column_polars_matches_pandas():

        from sksurv.column import encode_categorical

        data = {"b": [True, False, True, False, True]}
        pd_out = encode_categorical(pd.DataFrame(data), columns=["b"])
        pl_out = encode_categorical(pl.DataFrame(data), columns=["b"])
        assert list(pd_out.columns) == list(pl_out.columns) == ["b=True"]
        np.testing.assert_array_equal(pd_out.to_numpy(), pl_out.to_numpy(), strict=True)

    @staticmethod
    def test_explicit_numeric_column_preserves_value_ordering():

        from sksurv.column import encode_categorical

        data = {"x": [1, 2, 10, 1]}
        pd_out = encode_categorical(pd.DataFrame(data), columns=["x"])
        pl_out = encode_categorical(pl.DataFrame(data), columns=["x"])
        assert list(pd_out.columns) == list(pl_out.columns) == ["x=2", "x=10"]
        np.testing.assert_array_equal(pd_out.to_numpy(), pl_out.to_numpy(), strict=True)


def test_standardize_polars_nan_skipped_like_null():
    import polars.testing as pt

    # Float NaN must be skipped by the statistics (it would otherwise poison
    # the whole column) and stays NaN in the output, unlike null which stays null.
    data = pl.DataFrame({"a": [1.0, 2.0, float("nan"), 4.0], "b": [1.0, 2.0, 3.0, 4.0]})

    result = column.standardize(data)

    expected = pl.DataFrame(
        {"a": [-0.872872, -0.218218, float("nan"), 1.091089], "b": [-1.161895, -0.387298, 0.387298, 1.161895]}
    )
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


def test_encode_categorical_series_numeric_rejected(dataframe_backend):
    s = dataframe_backend.make_series("a_series", [0.5, 0.1, 10.0, 25.0, 3.8, 11.0])
    with pytest.raises(TypeError, match="series must be of categorical dtype"):
        column.encode_categorical(s)


def test_encode_categorical_series(dataframe_backend):
    values = ["medium", "medium", "small", "large", "small", "medium", "large", "medium", "large", "small"]
    s = dataframe_backend.make_series("a_series", values, categories=["small", "medium", "large"])

    actual = column.encode_categorical(s)

    expected = dataframe_backend.make_frame(
        {
            "a_series=medium": [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            "a_series=large": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        }
    )
    dataframe_backend.assert_frame_equal(actual, expected)


def test_encode_categorical_series_declared_unseen(dataframe_backend):
    s = dataframe_backend.make_series("grade", ["I", "II", "III", "I"], categories=["I", "II", "III", "IV"])

    actual = column.encode_categorical(s)

    assert list(actual.columns) == ["grade=II", "grade=III", "grade=IV"]
    assert actual["grade=IV"].to_numpy().tolist() == [0.0, 0.0, 0.0, 0.0]


def test_encode_categorical_single_category_series_drop_policy(dataframe_backend):
    s = dataframe_backend.make_series("c", ["only", "only"], categories=["only"])

    dropped = column.encode_categorical(s)
    assert dropped.shape[1] == 0

    preserved = column.encode_categorical(s, allow_drop=False)
    assert list(preserved) == ["only", "only"]


def test_categorical_to_numeric_bool_series(dataframe_backend):
    s = dataframe_backend.make_series("flag", [True, False, True])

    result = column.categorical_to_numeric(s)

    np.testing.assert_array_equal(result.to_numpy(), np.array([1, 0, 1]), strict=True)


def test_categorical_to_numeric_numeric_string_series(dataframe_backend):
    s = dataframe_backend.make_series("digits", ["1", "2", "10"])

    result = column.categorical_to_numeric(s)

    np.testing.assert_array_equal(result.to_numpy(), np.array([1, 2, 10]), strict=True)


def test_categorical_to_numeric_float_series_passes_through(dataframe_backend):
    s = dataframe_backend.make_series("x", [1.2, 2.8])

    result = column.categorical_to_numeric(s)

    np.testing.assert_array_equal(result.to_numpy(), np.array([1.2, 2.8]), strict=True)


def test_encode_categorical_polars_dataframe_drop_emits_warning(caplog):
    df = pl.DataFrame({"single": pl.Series(["only", "only"], dtype=pl.Enum(["only"]))})
    result = column.encode_categorical(df)
    assert result.shape == (0, 0)
    assert "dropped categorical variable 'single'" in caplog.text


def test_categorical_to_numeric_polars_unsupported_column_passes_through():
    df = pl.DataFrame({"items": [[1], [2]]})
    result = column.categorical_to_numeric(df)
    assert result.to_dict(as_series=False) == {"items": [[1], [2]]}


def _size_answer_categorical_frame():
    return pl.DataFrame(
        {
            "size": pl.Series(["medium", "small", "large", "xlarge", "small"], dtype=pl.Categorical),
            "answer": pl.Series(["yes", "no", "yes", "yes", "no"], dtype=pl.Categorical),
        }
    )


class TestPolarsCategoricalGlobalPoolBug:
    """Polars ``pl.Categorical`` categories must be column-scoped."""

    @staticmethod
    def test_infer_isolates_categories_per_column():
        df = _size_answer_categorical_frame()
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
        df = _size_answer_categorical_frame()
        encoded = column.encode_categorical(df)
        assert encoded.columns == ["size=medium", "size=small", "size=xlarge", "answer=yes"]

    @staticmethod
    def test_one_hot_encoder_categories_isolated():
        df = _size_answer_categorical_frame()
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
    def test_categorical_to_numeric_enum_matches_pandas_explicit_order():

        from sksurv.column import categorical_to_numeric

        values = ["c", "a", "b", "c"]
        categories = ["c", "a", "b"]
        df_pd = pd.DataFrame({"x": pd.Categorical(values, categories=categories)})
        df_pl = pl.DataFrame({"x": pl.Series("x", values, dtype=pl.Enum(categories))})
        out_pd = categorical_to_numeric(df_pd).to_numpy()
        out_pl = categorical_to_numeric(df_pl).to_numpy()
        np.testing.assert_array_equal(out_pd, out_pl, strict=True)

    @pytest.mark.parametrize("func", [column.categorical_to_numeric, column.encode_categorical])
    @staticmethod
    def test_pl_categorical_matches_pandas(func):
        values = ["banana", "apple", "cherry", "apple", "banana"]
        df_pd = pd.DataFrame({"fruit": pd.Categorical(values)})
        df_pl = pl.DataFrame({"fruit": pl.Series(values, dtype=pl.Categorical)})
        out_pd = func(df_pd).to_numpy()
        out_pl = func(df_pl).to_numpy()
        np.testing.assert_array_equal(out_pd, out_pl, strict=True)


class TestLazyFramePaths:
    @staticmethod
    def test_standardize_lazyframe_rejected(polars_grade_enum_frame):
        from sksurv.column import standardize

        with pytest.raises(TypeError, match=r"polars\.LazyFrame is not supported"):
            standardize(polars_grade_enum_frame.lazy())

    @staticmethod
    def test_categorical_to_numeric_lazyframe_rejected(polars_grade_enum_frame):
        from sksurv.column import categorical_to_numeric

        with pytest.raises(TypeError, match=r"polars\.LazyFrame is not supported"):
            categorical_to_numeric(polars_grade_enum_frame.lazy())

    @staticmethod
    def test_encode_categorical_lazyframe_rejected(polars_grade_enum_frame):
        from sksurv.column import encode_categorical

        with pytest.raises(TypeError, match=r"polars\.LazyFrame is not supported"):
            encode_categorical(polars_grade_enum_frame.lazy())
