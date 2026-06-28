"""Tests for ``sksurv.preprocessing.OneHotEncoder`` with polars input."""

from dataframe_test_utils import make_one_hot_categorical_data, to_polars_dataframe
import numpy as np
import pandas as pd
import polars as pl
import pytest

from sksurv.preprocessing import OneHotEncoder


class TestOneHotEncoderAllDroppedParity:
    @staticmethod
    @pytest.mark.parametrize("dataframe_library", ["pandas", "polars"])
    def test_all_dropped_raises(dataframe_library):
        df = pd.DataFrame({"cat": pd.Categorical(["x", "x"])})
        if dataframe_library == "polars":
            df = to_polars_dataframe(df)
        with pytest.raises(ValueError, match="No objects to concatenate"):
            OneHotEncoder().fit_transform(df)


@pytest.fixture()
def polars_categorical_data():
    def _create(n_samples=117):
        data, expected = make_one_hot_categorical_data(n_samples)
        return to_polars_dataframe(data), to_polars_dataframe(expected)

    return _create


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
    def test_fit_transform_lazyframe_rejected(polars_categorical_data):
        data, _ = polars_categorical_data()
        with pytest.raises(TypeError, match=r"polars\.LazyFrame is not supported"):
            OneHotEncoder().fit_transform(data.lazy())

    @staticmethod
    def test_transform_lazyframe_rejected():
        data = pl.DataFrame(
            {
                "age": [40.0, 50.0, 60.0, 70.0],
                "grade": pl.Series(["I", "II", "III", "I"], dtype=pl.Enum(["I", "II", "III", "IV"])),
            }
        )
        enc = OneHotEncoder().fit(data)
        with pytest.raises(TypeError, match=r"polars\.LazyFrame is not supported"):
            enc.transform(data.lazy())

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
        df_fit_pl = pl.DataFrame({"color": pl.Series(df_fit_pd["color"].astype(str).to_list(), dtype=pl.Categorical)})
        df_test_pl = pl.DataFrame({"color": pl.Series(df_test_pd["color"].astype(str).to_list(), dtype=pl.Categorical)})
        return df_fit_pd, df_test_pd, df_fit_pl, df_test_pl

    def test_unseen_label_emits_nan_in_both_backends(self):
        df_fit_pd, df_test_pd, df_fit_pl, df_test_pl = self._make_fit_test_pair()
        out_pd = OneHotEncoder().fit(df_fit_pd).transform(df_test_pd).to_numpy()
        out_pl = OneHotEncoder().fit(df_fit_pl).transform(df_test_pl).to_numpy()
        assert np.isnan(out_pd[1]).all()
        assert np.isnan(out_pl[1]).all()
        np.testing.assert_array_equal(out_pd, out_pl, strict=True)

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
