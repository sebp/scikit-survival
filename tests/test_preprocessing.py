from collections import OrderedDict

import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
import pandas.testing as tm
import polars as pl
import pytest

from sksurv.preprocessing import OneHotEncoder
from sksurv.testing import get_pandas_infer_string_context
from sksurv.testing.dataframe import CROSS_LIBRARY_PAIRS, PANDAS_BACKEND, POLARS_BACKEND


def expected_one_hot_data(data):
    expected = []
    for nam, col in data.items():
        if hasattr(col, "cat"):
            for cat in col.cat.categories[1:]:
                name = f"{nam}={cat}"
                s = pd.Series(col == cat, dtype=np.float64)
                expected.append((name, s))
        else:
            expected.append((nam, col))

    return pd.DataFrame.from_dict(OrderedDict(expected))


ONE_HOT_CATEGORIES = {
    "binary_1": ["Yes", "No"],
    "binary_2": ["East", "West"],
    "trinary": ["Green", "Blue", "Red"],
    "many": ["One", "Two", "Three", "Four", "Five", "Six"],
}


def make_one_hot_data(backend, n_samples=117):
    """One-hot encoder test data built by `backend`.

    Returns the input frame, whose categorical columns declare the
    :data:`ONE_HOT_CATEGORIES` category lists, and the expected encoded frame
    with one indicator column per category except the first.
    """
    rnd = np.random.default_rng(51365192)
    numeric = rnd.random((n_samples, 5))
    codes = {
        "binary_1": rnd.binomial(1, 0.6, n_samples),
        "binary_2": rnd.binomial(1, 0.376, n_samples),
        "trinary": rnd.binomial(2, 0.76, n_samples),
        "many": rnd.binomial(5, 0.47, n_samples),
    }

    data = {f"N{i}": numeric[:, i] for i in range(5)}
    expected = {f"N{i}": numeric[:, i] for i in range(5)}
    for name, declared in ONE_HOT_CATEGORIES.items():
        values = [declared[code] for code in codes[name]]
        data[name] = values
        for category in declared[1:]:
            expected[f"{name}={category}"] = np.array([value == category for value in values], dtype=float)

    input_frame = backend.make_frame(data, categories=ONE_HOT_CATEGORIES)
    expected_frame = backend.make_frame(expected)
    return input_frame, expected_frame


@pytest.fixture()
def create_backend_categorical_data(dataframe_backend_with_pandas_options):
    def _create(n_samples=117):
        return make_one_hot_data(dataframe_backend_with_pandas_options, n_samples)

    return _create


@pytest.fixture()
def create_categorical_data():
    def _create(n_samples=117):
        return make_one_hot_data(PANDAS_BACKEND, n_samples)

    return _create


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
    @staticmethod
    def test_fit(create_backend_categorical_data):
        data, expected_data = create_backend_categorical_data()

        t = OneHotEncoder().fit(data)

        assert isinstance(t.feature_names_, pd.Index)
        assert isinstance(t.encoded_columns_, pd.Index)
        assert t.feature_names_.tolist() == list(ONE_HOT_CATEGORIES)
        assert set(t.encoded_columns_) == set(expected_data.columns)

        assert set(t.categories_) == set(ONE_HOT_CATEGORIES)
        for key, expected_categories in ONE_HOT_CATEGORIES.items():
            assert isinstance(t.categories_[key], pd.Index)
            assert t.categories_[key].tolist() == expected_categories

    @staticmethod
    def test_fit_transform(create_backend_categorical_data, dataframe_backend_with_pandas_options):
        data, expected_data = create_backend_categorical_data()

        actual_data = OneHotEncoder().fit_transform(data)
        dataframe_backend_with_pandas_options.assert_frame_equal(actual_data, expected_data)

    @staticmethod
    def test_transform(create_backend_categorical_data, dataframe_backend_with_pandas_options):
        data, _ = create_backend_categorical_data()

        t = OneHotEncoder().fit(data)
        data, expected_data = create_backend_categorical_data(165)
        actual_data = t.transform(data)
        dataframe_backend_with_pandas_options.assert_frame_equal(actual_data, expected_data)

        columns = list(data.columns)
        reordered = data[columns[:2] + columns[5:] + columns[2:5]]
        actual_data = t.transform(reordered)
        dataframe_backend_with_pandas_options.assert_frame_equal(actual_data, expected_data)

    @staticmethod
    def test_get_feature_names_out(create_backend_categorical_data):
        data, expected_data = create_backend_categorical_data()

        t = OneHotEncoder()
        t.fit(data)

        out_names = t.get_feature_names_out()
        expected_names = np.asarray(list(expected_data.columns), dtype=object)
        assert_array_equal(out_names, expected_names, strict=True)

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
    def test_all_dropped_raises(dataframe_backend):
        df = dataframe_backend.make_frame({"cat": ["x", "x"]}, categories={"cat": ["x"]})
        with pytest.raises(ValueError, match="No objects to concatenate"):
            OneHotEncoder().fit_transform(df)


class TestOneHotEncoderLazyFrame:
    @staticmethod
    def test_fit_transform_lazyframe_rejected():
        data, _ = make_one_hot_data(POLARS_BACKEND)
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


class TestOneHotEncoderDeclaredUnseenCategory:
    @staticmethod
    def test_declared_unseen_category_produces_column(dataframe_backend):
        """A category that is declared but absent from the data still yields
        an encoded column, filled with zeros."""
        data = {"grade": ["I", "II", "III", "I"]}
        categories = {"grade": ["I", "II", "III", "IV"]}
        df = dataframe_backend.make_frame(data, categories=categories)

        enc = OneHotEncoder().fit(df)
        assert enc.categories_["grade"].tolist() == ["I", "II", "III", "IV"]
        assert "grade=IV" in list(enc.encoded_columns_)

        encoded = enc.transform(df)
        assert list(encoded.columns) == ["grade=II", "grade=III", "grade=IV"]
        assert encoded["grade=IV"].to_numpy().tolist() == [0.0, 0.0, 0.0, 0.0]


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

    @staticmethod
    @pytest.mark.parametrize("fit_backend,transform_backend", CROSS_LIBRARY_PAIRS)
    def test_transform_library_mismatch_raises(fit_backend, transform_backend):
        categories = {"color": ["blue", "green", "red"]}
        data_fit = fit_backend.make_frame({"color": ["red", "green", "blue", "red"]}, categories=categories)
        data_transform = transform_backend.make_frame({"color": ["red", "blue"]}, categories=categories)

        enc = OneHotEncoder().fit(data_fit)
        with pytest.raises(TypeError, match="same dataframe library"):
            enc.transform(data_transform)
