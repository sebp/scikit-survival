"""Tests for ``sksurv.column`` with polars input."""

from contextlib import nullcontext as does_not_raise

from dataframe_test_utils import to_polars_dataframe
import numpy as np
import polars as pl
import pytest

from sksurv import column as _sksurv_column
from sksurv._dataframe import infer_column_semantics
from sksurv.preprocessing import OneHotEncoder
from sksurv.testing import FixtureParameterFactory


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
        result = _sksurv_column.standardize(in_data)
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

    pd_out = _sksurv_column.standardize(pd.DataFrame({"a": [np.nan] * 3, "b": [1.0, 2.0, 3.0]}))
    # The all-missing column must be a float dtype, not Null, or standardize
    # would skip it as non-numeric.
    pl_out = _sksurv_column.standardize(pl.DataFrame({"a": polars_missing, "b": [1.0, 2.0, 3.0]}))

    # The dataframe-level representation differs (pandas NaN vs polars null) but
    # both normalize to NaN at the numpy boundary that feeds the estimators.
    np.testing.assert_allclose(pd_out.to_numpy(), pl_out.to_numpy(), equal_nan=True)


def test_standardize_all_missing_polars_column_stays_null():
    # Pin the polars dataframe-level behavior so a future change to NaN is noticed.
    out = _sksurv_column.standardize(pl.DataFrame({"a": pl.Series([None] * 3, dtype=pl.Float64), "b": [1.0, 2.0, 3.0]}))
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

    actual = _sksurv_column.encode_categorical(inputs, **kwargs)
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
    actual = _sksurv_column.encode_categorical(s)
    pt.assert_frame_equal(actual, expected, check_exact=False, abs_tol=1e-9)


def test_encode_categorical_polars_series_numeric_rejected():
    s = pl.Series("a_series", [0.5, 0.1, 10.0, 25.0, 3.8, 11.0])
    with pytest.raises(TypeError):
        _sksurv_column.encode_categorical(s)


def test_encode_categorical_polars_dataframe_drop_emits_warning(caplog):
    df = pl.DataFrame({"single": pl.Series(["only", "only"], dtype=pl.Enum(["only"]))})
    result = _sksurv_column.encode_categorical(df)
    assert result.shape == (0, 0)
    assert "dropped categorical variable 'single'" in caplog.text


def test_encode_categorical_polars_single_category_series_drop_policy():
    s = pl.Series("a_series", ["only", "only"], dtype=pl.Enum(["only"]))

    dropped = _sksurv_column.encode_categorical(s)
    assert dropped.shape == (0, 0)

    preserved = _sksurv_column.encode_categorical(s, allow_drop=False)
    assert preserved.to_list() == ["only", "only"]


def test_categorical_to_numeric_polars_bool_series():
    result = _sksurv_column.categorical_to_numeric(pl.Series("flag", [True, False, True]))
    assert result.to_list() == [1, 0, 1]
    assert result.dtype == pl.Int64


def test_categorical_to_numeric_polars_unsupported_column_passes_through():
    df = pl.DataFrame({"items": [[1], [2]]})
    result = _sksurv_column.categorical_to_numeric(df)
    assert result.to_dict(as_series=False) == {"items": [[1], [2]]}


def test_categorical_to_numeric_polars_numeric_string_series():
    result = _sksurv_column.categorical_to_numeric(pl.Series("digits", ["1", "2", "10"]))
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

    actual = _sksurv_column.categorical_to_numeric(inputs)
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
        encoded = _sksurv_column.encode_categorical(df)
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
