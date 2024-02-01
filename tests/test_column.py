import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
import pandas.testing as tm
import pytest

from sksurv import column
from sksurv.testing import FixtureParameterFactory


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

        data["q3"] = data.loc[:, "q3"].astype("category")
        return data

    def data_numeric(self):
        return self.numeric_data, self.expected

    def data_float_numpy_array(self):
        return self.numeric_data.values, self.expected

    def data_int_numpy_array(self):
        return self.numeric_data.values.astype(int), self.expected

    def data_non_numeric(self):
        return self.non_numeric_data, self.non_numeric_data

    def data_non_numeric_numpy_array(self):
        data = self.non_numeric_data.values
        return data, pd.DataFrame(data)

    def data_mixed(self):
        mixed_data_frame = pd.concat((self.numeric_data, self.non_numeric_data), axis=1)
        expected = pd.concat((self.expected, self.non_numeric_data), axis=1)
        return mixed_data_frame, expected

    def data_mixed_numpy_array(self):
        data, _ = self.data_mixed()
        data = data.values
        return data, pd.DataFrame(data)


@pytest.mark.parametrize("in_data,expected", StandardizeCase().get_cases())
def test_standardize(in_data, expected):
    before = in_data.copy()
    result = column.standardize(in_data)

    # check that data wasn't modified inplace
    if isinstance(before, np.ndarray) and np.issubdtype(before.dtype, float):
        assert_array_equal(before, in_data)
    elif isinstance(before, pd.DataFrame):
        tm.assert_frame_equal(before, in_data)

    if isinstance(result, np.ndarray):
        result = pd.DataFrame(result, columns=expected.columns)

    tm.assert_frame_equal(pd.isnull(result), pd.isnull(expected))
    tm.assert_frame_equal(result, expected)


class CategoricalCases(FixtureParameterFactory):
    def _make_randn(self, shape):
        return np.random.RandomState(0).randn(shape)

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
                "a_number": input_df.loc[:, "a_number"].values.copy(),
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

        expected_df.loc[:, "a_number"] = input_df.loc[:, "a_number"].values.copy()

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


@pytest.mark.parametrize("inputs,kwargs,expected_df", EncodeCategoricalCases().get_cases())
@pytest.mark.filterwarnings(
    "ignore:In a future version, the Index constructor will not infer numeric dtypes when "
    "passed object-dtype sequences \\(matching Series behavior\\):FutureWarning"
)  # deprecated in pandas 1.4.0
def test_encode_categorical(inputs, kwargs, expected_df):
    actual_df = column.encode_categorical(inputs, **kwargs)
    tm.assert_frame_equal(actual_df.isnull(), expected_df.isnull())
    tm.assert_frame_equal(actual_df, expected_df, check_exact=True)


def test_series_numeric():
    input_series = pd.Series([0.5, 0.1, 10, 25, 3.8, 11, 2256, -1, -0.2, 3.14], name="a_series")

    with pytest.raises(TypeError, match="series must be of categorical dtype, but was float"):
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

    def data_frame_to_numeric(self):
        input_df = self.mixed_data_frame

        a_num = np.r_[np.repeat([0], 10), np.repeat([2], 5), np.repeat([3], 13), np.repeat([1], 3)].astype(np.int64)
        b_num = np.r_[np.repeat([1], 8), np.repeat([0], 23)].astype(np.int64)

        expected = pd.DataFrame.from_dict({"a_category": a_num, "a_binary": b_num})
        expected.loc[:, "a_number"] = input_df.loc[:, "a_number"].values.copy()

        return input_df, expected


@pytest.mark.parametrize("input_df,expected", CategoricalToNumeric().get_cases())
def test_categorical_to_numeric(input_df, expected):
    actual = column.categorical_to_numeric(input_df)

    if isinstance(expected, pd.Series):
        tm.assert_series_equal(actual, expected, check_exact=True)
    else:
        tm.assert_frame_equal(actual, expected, check_exact=True)
