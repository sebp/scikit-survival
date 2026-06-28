from dataframe_test_utils import expected_one_hot_data, make_one_hot_categorical_data
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
import pandas.testing as tm
import pytest

from sksurv.preprocessing import OneHotEncoder
from sksurv.testing import get_pandas_infer_string_context


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

    @pytest.mark.parametrize("infer_string_context", get_pandas_infer_string_context())
    @staticmethod
    def test_fit_transform(create_categorical_data, infer_string_context):
        with infer_string_context:
            data, expected_data = create_categorical_data()

            actual_data = OneHotEncoder().fit_transform(data)
            tm.assert_frame_equal(actual_data, expected_data)

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
