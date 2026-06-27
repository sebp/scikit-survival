from dataframe_test_utils import make_clinical_kernel_pandas_data
import numpy as np
from numpy.testing import assert_array_almost_equal
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.metrics.pairwise import pairwise_kernels

from sksurv.kernels import ClinicalKernelTransform, clinical_kernel


@pytest.fixture()
def make_data():
    return make_clinical_kernel_pandas_data


class TestClinicalKernel:
    @staticmethod
    def test_clinical_kernel_1(make_data):
        data, expected = make_data()
        mat = clinical_kernel(data)

        assert_array_almost_equal(expected, mat, 4)

    @staticmethod
    def test_clinical_kernel_no_ordinal(make_data):
        data, expected = make_data(with_ordinal=False)
        mat = clinical_kernel(data)
        assert_array_almost_equal(expected, mat, 4)

    @staticmethod
    def test_clinical_kernel_no_nominal(make_data):
        data, expected = make_data(with_nominal=False)
        mat = clinical_kernel(data)
        assert_array_almost_equal(expected, mat, 4)

    @staticmethod
    def test_clinical_kernel_no_continuous(make_data):
        data, expected = make_data(with_continuous=False)
        mat = clinical_kernel(data)
        assert_array_almost_equal(expected, mat, 4)

    @staticmethod
    def test_clinical_kernel_only_nominal(make_data):
        data, expected = make_data(with_continuous=False, with_ordinal=False)
        mat = clinical_kernel(data)
        assert_array_almost_equal(expected, mat, 4)

    @staticmethod
    def test_clinical_kernel_x_and_y(make_data):
        data, m = make_data()
        mat = clinical_kernel(data.iloc[:3, :], data.iloc[3:, :])
        expected = m[:3:, 3:]

        assert_array_almost_equal(expected, mat, 4)

    @staticmethod
    def test_fit_error_ndim():
        t = ClinicalKernelTransform()
        rng = np.random.default_rng()

        with pytest.raises(ValueError, match="expected 2d array, but got 1"):
            t.fit(rng.standard_normal(31))

        with pytest.raises(ValueError, match="expected 2d array, but got 3"):
            t.fit(rng.standard_normal((31, 20, 2)))

    @staticmethod
    def test_kernel_transform(make_data):
        data, expected = make_data()
        t = ClinicalKernelTransform()

        t.fit(data)
        df_test = pd.DataFrame(t.X_fit_, columns=data.columns)
        mat = t.transform(df_test)

        assert_array_almost_equal(expected, mat, 4)

    @staticmethod
    def test_kernel_transform_x_and_y(make_data):
        data, m = make_data()
        t = ClinicalKernelTransform(fit_once=True)
        t.prepare(data)
        x_num = t.X_fit_.copy()

        t.fit(x_num[:3, :])
        mat = t.transform(x_num[3:, :])

        expected = m[:3, 3:].T

        assert_array_almost_equal(expected, mat, 4)

    @staticmethod
    def test_kernel_transform_with_missing_numeric():
        x = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": [1.0, 1.0, 2.0, 2.0]})

        t = ClinicalKernelTransform().fit(x)

        # The range of "a" is computed from the non-missing values only.
        assert_array_almost_equal(t._numeric_ranges, [3.0, 1.0])

        mat = t.transform(x)
        # Only pairs involving the missing value are NaN.
        expected = np.array(
            [
                [1.0, 5.0 / 6, np.nan, 0.0],
                [5.0 / 6, 1.0, np.nan, 1.0 / 6],
                [np.nan, np.nan, np.nan, np.nan],
                [0.0, 1.0 / 6, np.nan, 1.0],
            ]
        )
        assert_array_almost_equal(mat, expected)

    @staticmethod
    def test_kernel_transform_num_features_mismatch(make_data):
        data, _ = make_data()
        t = ClinicalKernelTransform()
        t.fit(data)

        array_test = np.zeros((2, 17), dtype=float)

        error_msg = r"X has 17 features, but ClinicalKernelTransform is expecting 4 features as input\."
        warn_msg = "X does not have valid feature names, but ClinicalKernelTransform was fitted with feature names"
        with pytest.raises(ValueError, match=error_msg), pytest.warns(UserWarning, match=warn_msg):
            t.transform(array_test)

    @staticmethod
    def test_kernel_transform_feature_names_mismatch(make_data):
        data, _ = make_data()
        t = ClinicalKernelTransform()
        t.fit(data)

        df_test = pd.DataFrame(np.zeros((2, data.shape[1] + 1), dtype=float), columns=data.columns.tolist() + ["XYZ"])

        error_msg = r"""The feature names should match those that were passed during fit\.
Feature names unseen at fit time:
- XYZ
"""
        with pytest.raises(ValueError, match=error_msg):
            t.transform(df_test)

    @staticmethod
    def test_pairwise(make_data):
        data, expected = make_data()
        t = ClinicalKernelTransform()
        t.fit(data)

        mat = pairwise_kernels(t.X_fit_, t.X_fit_, metric=t.pairwise_kernel, n_jobs=1)

        assert_array_almost_equal(expected, mat, 4)

    @staticmethod
    def test_pairwise_x_and_y(make_data):
        data, m = make_data()
        t = ClinicalKernelTransform()
        t.fit(data)

        mat = pairwise_kernels(t.X_fit_[:3, :], t.X_fit_[3:, :], metric=t.pairwise_kernel, n_jobs=1)

        expected = m[:3:, 3:]

        assert_array_almost_equal(expected, mat, 4)

    @staticmethod
    def test_pairwise_x_and_y_error_shape(make_data):
        data, _ = make_data()
        t = ClinicalKernelTransform()
        t.fit(data)

        with pytest.raises(
            ValueError,
            match=r"Incompatible dimension for X and Y matrices: X\.shape\[0\] == 4 while Y\.shape\[0\] == 2",
        ):
            t.pairwise_kernel(data.iloc[0, :], data.iloc[1, :2])

    @staticmethod
    def test_pairwise_no_nominal(make_data):
        data, expected = make_data(with_nominal=False)
        t = ClinicalKernelTransform()
        t.fit(data)

        mat = pairwise_kernels(t.X_fit_[:3, :], t.X_fit_[3:, :], metric=t.pairwise_kernel, n_jobs=1)

        assert_array_almost_equal(expected[:3:, 3:], mat, 4)

    @staticmethod
    def test_call_function(make_data):
        data, expected = make_data()
        t = ClinicalKernelTransform(fit_once=True)
        t.prepare(data)

        mat = t(t.X_fit_, t.X_fit_)
        assert_array_almost_equal(expected, mat, 4)

    @staticmethod
    def test_call_function_x_and_y(make_data):
        data, m = make_data()
        t = ClinicalKernelTransform(fit_once=True)
        t.prepare(data)

        mat = t(t.X_fit_[:3, :], t.X_fit_[3:, :])
        expected = m[:3:, 3:]

        assert_array_almost_equal(expected, mat, 4)

    @staticmethod
    def test_pairwise_feature_mismatch(make_data):
        data, _ = make_data()
        t = ClinicalKernelTransform()
        t.fit(data)

        with pytest.raises(
            ValueError,
            match=r"Incompatible dimension for X and Y matrices: X\.shape\[[0-1]\] == 4 while Y\.shape\[[0-1]\] == 17",
        ):
            pairwise_kernels(t.X_fit_, np.zeros((5, 17), dtype=float), metric=t.pairwise_kernel, n_jobs=1)

    @staticmethod
    def test_prepare(make_data):
        data, expected = make_data()
        t = ClinicalKernelTransform(fit_once=True)
        t.prepare(data)

        mat = clone(t).fit(t.X_fit_).transform(t.X_fit_[:4, :])

        assert_array_almost_equal(expected[:4, :], mat, 4)

    @staticmethod
    def test_fit_once_fit_dataframe_raises(make_data):
        data, _ = make_data()
        t = ClinicalKernelTransform(fit_once=True)
        t.prepare(data)

        with pytest.raises(TypeError, match="fit_once=True expects a numeric array in fit"):
            t.fit(data)

    @staticmethod
    def test_prepare_error_fit_once(make_data):
        data = make_data()
        t = ClinicalKernelTransform(fit_once=False)

        with pytest.raises(ValueError, match="prepare can only be used if fit_once parameter is set to True"):
            t.prepare(data)

    @staticmethod
    def test_prepare_error_type():
        t = ClinicalKernelTransform(fit_once=True)

        with pytest.raises(TypeError, match=r"X must be a pandas DataFrame or supported Narwhals dataframe input"):
            t.prepare([[0, 1], [1, 2], [4, 3], [6, 5]])

    @staticmethod
    def test_prepare_error_dtype():
        t = ClinicalKernelTransform(fit_once=True)
        data = pd.DataFrame.from_dict(
            {
                "age": [12, 61, 18, 21, 57, 17],
                "date": np.array(
                    ["2016-01-01", "1954-06-30", "1999-03-01", "2005-02-25", "2112-12-31", "1731-09-16"],
                    dtype="datetime64",
                ),
            }
        )

        with pytest.raises(TypeError, match=r"unsupported dtype: Datetime"):
            t.prepare(data)

    @staticmethod
    def test_bool_column_treated_as_numeric():
        df_bool = pd.DataFrame(
            {
                "age": [20.0, 23.0, 26.0, 54.0, 100.0],
                "event": [True, False, True, True, False],
            }
        )
        df_uint = df_bool.assign(event=df_bool["event"].astype(np.uint8))

        assert_array_almost_equal(clinical_kernel(df_bool), clinical_kernel(df_uint))

        t_bool = ClinicalKernelTransform().fit(df_bool)
        t_uint = ClinicalKernelTransform().fit(df_uint)
        assert_array_almost_equal(t_bool._numeric_ranges, t_uint._numeric_ranges)
        assert_array_almost_equal(t_bool.X_fit_, t_uint.X_fit_)
        assert list(t_bool._numeric_columns) == [0, 1]
        assert list(t_bool._nominal_columns) == []

    @staticmethod
    def test_object_column_treated_as_nominal():
        df_object = pd.DataFrame(
            {
                "age": [20.0, 23.0, 54.0, 100.0],
                "stage": pd.Series([1, 2, 1, 2], dtype=object),
            }
        )
        df_categorical = df_object.assign(stage=pd.Categorical(df_object["stage"], categories=[1, 2]))

        assert_array_almost_equal(clinical_kernel(df_object), clinical_kernel(df_categorical))

        t_object = ClinicalKernelTransform().fit(df_object)
        t_categorical = ClinicalKernelTransform().fit(df_categorical)
        assert_array_almost_equal(t_object.transform(df_object), t_categorical.transform(df_categorical))
        assert list(t_object._numeric_columns) == [0]
        assert list(t_object._nominal_columns) == [1]

    @staticmethod
    def test_feature_mismatch(make_data):
        data, _ = make_data()
        x = data.iloc[:, :2]
        y = data.iloc[:, 2:]
        with pytest.raises(ValueError, match="columns do not match"):
            clinical_kernel(x, y)

        y = np.zeros((10, 17))
        with pytest.raises(ValueError, match="x and y have different number of features"):
            clinical_kernel(x, y)
