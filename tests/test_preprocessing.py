from dataframe_test_utils import (
    assert_backend_frame_equal,
    expected_one_hot_data,
    make_one_hot_categorical_data,
    to_polars_dataframe,
)
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
import pandas.testing as tm
import polars as pl
import pytest

from sksurv.preprocessing import OneHotEncoder
from sksurv.testing import get_pandas_infer_string_context


@pytest.fixture()
def create_backend_categorical_data(dataframe_backend):
    def _create(n_samples=117):
        data, expected = make_one_hot_categorical_data(n_samples)
        if dataframe_backend == "polars":
            return to_polars_dataframe(data), to_polars_dataframe(expected)
        return data, expected

    return _create


@pytest.fixture()
def create_categorical_data():
    return make_one_hot_categorical_data


@pytest.fixture()
def create_string_data():
    def _create_data(n_samples=97):
        rnd = np.random.default_rng(882)
        data = pd.DataFrame(
            {
                "answer": np.array(["Yes", "No"])[rnd.binomial(1, 0.6, n_samples)],
                "direction": np.array(["East", "North", "West", "South"])[rnd.integers(4, size=n_samples)],
                "color": np.array(["Green", "Blue", "Red"])[rnd.integers(3, size=n_samples)],
            }
        )

        data_cat = data.astype(dict.fromkeys(data.columns, "category"))
        return data, expected_one_hot_data(data_cat)

    return _create_data


class TestOneHotEncoder:
    @pytest.mark.parametrize("infer_string_context", get_pandas_infer_string_context())
    @staticmethod
    def test_fit(create_categorical_data, infer_string_context):
        with infer_string_context:
            data, expected_data = create_categorical_data()

            t = OneHotEncoder().fit(data)

            assert isinstance(t.feature_names_, pd.Index)
            assert isinstance(t.encoded_columns_, pd.Index)
            assert t.feature_names_.tolist() == ["binary_1", "binary_2", "trinary", "many"]
            assert set(t.encoded_columns_) == set(expected_data.columns)

            expected_categories = {k: data[k].cat.categories for k in ["binary_1", "binary_2", "trinary", "many"]}
            assert set(t.categories_) == set(expected_categories)
            for key, expected_index in expected_categories.items():
                assert isinstance(t.categories_[key], pd.Index)
                assert t.categories_[key].tolist() == expected_index.tolist()

    @staticmethod
    def test_fit_transform(create_backend_categorical_data, dataframe_backend):
        data, expected_data = create_backend_categorical_data()

        actual_data = OneHotEncoder().fit_transform(data)
        assert_backend_frame_equal(actual_data, expected_data, dataframe_backend)

    @pytest.mark.parametrize("infer_string_context", get_pandas_infer_string_context())
    @staticmethod
    def test_transform(create_categorical_data, infer_string_context):
        with infer_string_context:
            data, _ = create_categorical_data()

            t = OneHotEncoder().fit(data)
            data, expected_data = create_categorical_data(165)
            actual_data = t.transform(data)
            tm.assert_frame_equal(actual_data, expected_data)

            data = pd.concat((data.iloc[:, :2], data.iloc[:, 5:], data.iloc[:, 2:5]), axis=1)
            actual_data = t.transform(data)
            tm.assert_frame_equal(actual_data, expected_data)

    @pytest.mark.parametrize("infer_string_context", get_pandas_infer_string_context())
    @staticmethod
    def test_get_feature_names_out(create_categorical_data, infer_string_context):
        with infer_string_context:
            data, expected_data = create_categorical_data()

            t = OneHotEncoder()
            t.fit(data)

            out_names = t.get_feature_names_out()
            assert_array_equal(out_names, expected_data.columns.to_numpy(), strict=True)

    @pytest.mark.parametrize("infer_string_context", get_pandas_infer_string_context())
    @staticmethod
    def test_get_feature_names_out_shuffled(create_categorical_data, infer_string_context):
        with infer_string_context:
            data, _ = create_categorical_data()
            order = np.array(["binary_1", "N0", "N3", "trinary", "binary_2", "N1", "N2", "many"])
            expected_columns = np.array(
                [
                    "binary_1=No",
                    "N0",
                    "N3",
                    "trinary=Blue",
                    "trinary=Red",
                    "binary_2=West",
                    "N1",
                    "N2",
                    "many=Two",
                    "many=Three",
                    "many=Four",
                    "many=Five",
                    "many=Six",
                ],
                dtype=object,
            )

            t = OneHotEncoder()
            t.fit(data.loc[:, order])

            out_names = t.get_feature_names_out()
            assert_array_equal(out_names, expected_columns, strict=True)

            with pytest.raises(ValueError, match="input_features is not equal to feature_names_in_"):
                t.get_feature_names_out(data.columns.tolist())

    @pytest.mark.parametrize("infer_string_context", get_pandas_infer_string_context())
    @staticmethod
    def test_transform_other_columns(create_categorical_data, infer_string_context):
        with infer_string_context:
            data, _ = create_categorical_data()

            t = OneHotEncoder().fit(data)
            data, _ = create_categorical_data(125)

            data_renamed = data.rename(columns={"binary_1": "renamed_1"})
            with pytest.raises(ValueError, match=r"1 features are missing from data: \['binary_1'\]"):
                t.transform(data_renamed)

            data_dropped = data.drop("trinary", axis=1)
            with pytest.raises(ValueError, match=r"1 features are missing from data: \['trinary'\]"):
                t.transform(data_dropped)

            data_renamed = data.rename(columns={"binary_1": "renamed_1", "many": "too_many"})
            with pytest.raises(ValueError, match=r"2 features are missing from data: \['binary_1', 'many'\]"):
                t.transform(data_renamed)

    @pytest.mark.parametrize("infer_string_context", get_pandas_infer_string_context())
    @staticmethod
    def test_fit_transform_string_dtype(create_string_data, infer_string_context):
        with infer_string_context:
            data, expected = create_string_data()

            t = OneHotEncoder()
            transformed = t.fit_transform(data)

            assert t.feature_names_in_.tolist() == ["answer", "direction", "color"]

            assert t.get_feature_names_out().tolist() == [
                "answer=Yes",
                "direction=North",
                "direction=South",
                "direction=West",
                "color=Green",
                "color=Red",
            ]

            tm.assert_frame_equal(transformed, expected)

    @pytest.mark.parametrize("infer_string_context", get_pandas_infer_string_context())
    @pytest.mark.parametrize("n_rows_transform", [1, 2, 3, 4, 5, 10, 15, 20, 39])
    @staticmethod
    def test_fit_transform_mixed_dtype(
        create_categorical_data, create_string_data, n_rows_transform, infer_string_context
    ):
        with infer_string_context:
            data_cat, expected_cat = create_categorical_data(101)
            data_obj, expected_obj = create_string_data(101)

            data = pd.concat((data_cat, data_obj), axis=1)
            expected = pd.concat((expected_cat, expected_obj), axis=1)

            data_fit = data.iloc[n_rows_transform:]
            data_transform = data.iloc[:n_rows_transform]
            expected_transformed = expected.iloc[:n_rows_transform]

            t = OneHotEncoder().fit(data_fit)

            assert t.feature_names_in_.tolist() == [
                "N0",
                "N1",
                "N2",
                "N3",
                "N4",
                "binary_1",
                "binary_2",
                "trinary",
                "many",
                "answer",
                "direction",
                "color",
            ]

            transformed = t.transform(data_transform)
            assert transformed.shape[0] == n_rows_transform

            tm.assert_frame_equal(transformed, expected_transformed)


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
