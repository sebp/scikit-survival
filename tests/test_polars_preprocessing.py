"""Tests for ``sksurv.preprocessing.OneHotEncoder`` with polars input."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from sksurv.preprocessing import OneHotEncoder


class TestOneHotEncoderAllDroppedParity:
    @staticmethod
    def test_all_dropped_polars_raises():
        df = pl.DataFrame({"cat": pl.Series(["x", "x"], dtype=pl.Categorical)})
        with pytest.raises(ValueError, match="No objects to concatenate"):
            OneHotEncoder().fit_transform(df)

    @staticmethod
    def test_all_dropped_pandas_raises():
        df = pd.DataFrame({"cat": pd.Categorical(["x", "x"])})
        with pytest.raises(ValueError, match="No objects to concatenate"):
            OneHotEncoder().fit_transform(df)


class _OneHotEncoderPolarsFactory:
    def _create(self, n_samples=117):
        rnd = np.random.default_rng(51365192)
        numeric = {f"N{i}": rnd.random(n_samples) for i in range(5)}

        binary_1 = pl.Series(
            "binary_1",
            np.array(["Yes", "No"])[rnd.binomial(1, 0.6, n_samples)],
            dtype=pl.Enum(["Yes", "No"]),
        )
        binary_2 = pl.Series(
            "binary_2",
            np.array(["East", "West"])[rnd.binomial(1, 0.376, n_samples)],
            dtype=pl.Enum(["East", "West"]),
        )
        trinary = pl.Series(
            "trinary",
            np.array(["Green", "Blue", "Red"])[rnd.binomial(2, 0.76, n_samples)],
            dtype=pl.Enum(["Green", "Blue", "Red"]),
        )
        many = pl.Series(
            "many",
            np.array(["One", "Two", "Three", "Four", "Five", "Six"])[rnd.binomial(5, 0.47, n_samples)],
            dtype=pl.Enum(["One", "Two", "Three", "Four", "Five", "Six"]),
        )

        data = pl.DataFrame(numeric).with_columns([binary_1, binary_2, trinary, many])

        expected_cols = {}
        for n in [f"N{i}" for i in range(5)]:
            expected_cols[n] = data.get_column(n)
        for nam, series in [("binary_1", binary_1), ("binary_2", binary_2), ("trinary", trinary), ("many", many)]:
            cats = series.dtype.categories
            for cat in list(cats)[1:]:
                expected_cols[f"{nam}={cat}"] = (series == cat).cast(pl.Float64).to_numpy()
        expected = pl.DataFrame(expected_cols)
        return data, expected


@pytest.fixture()
def polars_categorical_data():
    return _OneHotEncoderPolarsFactory()._create


class TestOneHotEncoderPolars:
    @staticmethod
    def test_fit(polars_categorical_data):
        data, _ = polars_categorical_data()
        t = OneHotEncoder().fit(data)
        assert isinstance(t.feature_names_, pd.Index)
        assert isinstance(t.encoded_columns_, pd.Index)
        assert all(isinstance(categories, pd.Index) for categories in t.categories_.values())
        assert t.feature_names_.tolist() == ["binary_1", "binary_2", "trinary", "many"]
        assert t.categories_["binary_1"].tolist() == ["Yes", "No"]
        assert t.categories_["binary_2"].tolist() == ["East", "West"]
        assert t.categories_["trinary"].tolist() == ["Green", "Blue", "Red"]
        assert t.categories_["many"].tolist() == ["One", "Two", "Three", "Four", "Five", "Six"]

    @staticmethod
    def test_fit_transform(polars_categorical_data):
        import polars.testing as pt

        data, expected = polars_categorical_data()
        actual = OneHotEncoder().fit_transform(data)
        assert isinstance(actual, pl.DataFrame)
        pt.assert_frame_equal(actual, expected, check_exact=False, abs_tol=1e-9)

    @staticmethod
    def test_fit_transform_lazy(polars_categorical_data):
        import polars.testing as pt

        data, expected = polars_categorical_data()
        actual = OneHotEncoder().fit_transform(data.lazy())
        assert isinstance(actual, pl.DataFrame)
        pt.assert_frame_equal(actual, expected, check_exact=False, abs_tol=1e-9)

    @staticmethod
    def test_transform_lazy_after_fit():
        data = pl.DataFrame(
            {
                "age": [40.0, 50.0, 60.0, 70.0],
                "grade": pl.Series(["I", "II", "III", "I"], dtype=pl.Enum(["I", "II", "III", "IV"])),
            }
        )
        enc = OneHotEncoder().fit(data)
        out_eager = enc.transform(data)
        out_lazy = enc.transform(data.lazy())
        np.testing.assert_array_equal(out_eager.to_numpy(), out_lazy.to_numpy())

    @staticmethod
    def test_transform(polars_categorical_data):
        import polars.testing as pt

        data, _ = polars_categorical_data()
        t = OneHotEncoder().fit(data)
        new_data, expected = polars_categorical_data(165)
        actual = t.transform(new_data)
        assert isinstance(actual, pl.DataFrame)
        pt.assert_frame_equal(actual, expected, check_exact=False, abs_tol=1e-9)

    @staticmethod
    def test_get_feature_names_out(polars_categorical_data):
        data, expected = polars_categorical_data()
        encoder = OneHotEncoder().fit(data)
        names = encoder.get_feature_names_out()
        assert list(names) == list(expected.columns)


class TestOneHotEncoderUnseenAndCrossDataframeLibrary:
    @staticmethod
    def _make_fit_test_pair():
        fit = ["red", "green", "blue", "red"]
        test = ["red", "yellow", "blue"]

        df_fit_pd = pd.DataFrame({"color": pd.Categorical(fit)})
        df_test_pd = pd.DataFrame({"color": pd.Categorical(test)})
        df_fit_pl = pl.DataFrame({"color": pl.Series(fit, dtype=pl.Categorical)})
        df_test_pl = pl.DataFrame({"color": pl.Series(test, dtype=pl.Categorical)})
        return df_fit_pd, df_test_pd, df_fit_pl, df_test_pl

    def test_unseen_label_emits_nan_in_both_backends(self):
        df_fit_pd, df_test_pd, df_fit_pl, df_test_pl = self._make_fit_test_pair()
        out_pd = OneHotEncoder().fit(df_fit_pd).transform(df_test_pd).to_numpy()
        out_pl = OneHotEncoder().fit(df_fit_pl).transform(df_test_pl).to_numpy()
        assert np.isnan(out_pd[1]).all()
        assert np.isnan(out_pl[1]).all()
        np.testing.assert_array_equal(out_pd, out_pl)

    def test_fit_pandas_transform_polars_raises(self):
        df_fit_pd, _, _, _ = self._make_fit_test_pair()
        df_test_pl = pl.DataFrame({"color": pl.Series(["red", "blue"], dtype=pl.Categorical)})
        enc = OneHotEncoder().fit(df_fit_pd)
        with pytest.raises(TypeError, match="same dataframe library"):
            enc.transform(df_test_pl)

    def test_fit_polars_transform_pandas_raises(self):
        _, _, df_fit_pl, _ = self._make_fit_test_pair()
        df_test_pd = pd.DataFrame({"color": pd.Categorical(["red", "blue"])})
        enc = OneHotEncoder().fit(df_fit_pl)
        with pytest.raises(TypeError, match="same dataframe library"):
            enc.transform(df_test_pd)
