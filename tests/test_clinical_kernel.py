import numpy as np
from numpy.testing import assert_array_almost_equal
import pandas as pd
import polars as pl
import pytest
from sklearn.base import clone
from sklearn.metrics.pairwise import pairwise_kernels

from sksurv.kernels import ClinicalKernelTransform, clinical_kernel
from sksurv.kernels._clinical_dataframe import _ordinal_range
from sksurv.preprocessing import OneHotEncoder
from sksurv.testing.dataframe import CROSS_LIBRARY_PAIRS, PANDAS_BACKEND, POLARS_BACKEND, PolarsBackend


def make_clinical_kernel_expected(with_ordinal=True, with_nominal=True, with_continuous=True):
    mat_age = np.array(
        [
            [1.0, 0.9625, 0.925, 0.575, 0.0],
            [0.9625, 1.0, 0.9625, 0.6125, 0.0375],
            [0.925, 0.9625, 1.0, 0.6500, 0.075],
            [0.575, 0.6125, 0.6500, 1.0, 0.425],
            [0.0, 0.0375, 0.075, 0.425, 1.0],
        ]
    )
    mat_node_size = np.array(
        [
            [1.0, 2 / 3, 2 / 3, 1 / 3, 2 / 3],
            [2 / 3, 1.0, 1 / 3, 0.0, 1.0],
            [2 / 3, 1 / 3, 1.0, 2 / 3, 1 / 3],
            [1 / 3, 0.0, 2 / 3, 1.0, 0.0],
            [2 / 3, 1.0, 1 / 3, 0.0, 1.0],
        ]
    )
    mat_node_spread = np.array(
        [
            [1.0, 0.0, 1.0, 0.5, 0.0],
            [0.0, 1.0, 0.0, 0.5, 1.0],
            [1.0, 0.0, 1.0, 0.5, 0.0],
            [0.5, 0.5, 0.5, 1.0, 0.5],
            [0.0, 1.0, 0.0, 0.5, 1.0],
        ]
    )
    mat_metastasis = np.array(
        [
            [1, 0, 1, 1, 0],
            [0, 1, 0, 0, 1],
            [1, 0, 1, 1, 0],
            [1, 0, 1, 1, 0],
            [0, 1, 0, 0, 1],
        ],
        dtype=float,
    )

    included = []
    if with_continuous:
        included.append(mat_age)
    if with_ordinal:
        included.append(mat_node_size)
        included.append(mat_node_spread)
    if with_nominal:
        included.append(mat_metastasis)

    expected = included[0]
    for i in range(1, len(included)):
        expected += included[i]
    expected /= len(included)
    return expected


def make_clinical_kernel_data(backend, with_ordinal=True, with_nominal=True, with_continuous=True):
    """Clinical kernel test data built by `backend`.

    Returns the dataframe, the expected kernel matrix, and the keyword
    arguments that declare the ordinal columns for that backend: empty when
    an ordered categorical dtype carries the declaration, an
    ``ordinal_categories`` mapping otherwise.
    """
    data = {}
    categories = {}
    kwargs = {}
    if with_continuous:
        data["age"] = [20, 23, 26, 54, 100]
    if with_ordinal:
        ordinal = {
            "lymph node size": ([2, 1, 3, 4, 1], [1, 2, 3, 4]),
            "lymph node spread": (["distant", "none", "distant", "close", "none"], ["none", "close", "distant"]),
        }
        if isinstance(backend, PolarsBackend):
            # polars has no ordered categorical dtype: the columns become
            # pl.Enum (string categories only) and the category order is
            # passed to the kernel via ``ordinal_categories``.
            for name, (values, declared) in ordinal.items():
                data[name] = [str(value) for value in values]
                categories[name] = [str(category) for category in declared]
            kwargs["ordinal_categories"] = {name: categories[name] for name in ordinal}
        else:
            # pandas: an ordered categorical dtype alone marks the column
            # as ordinal.
            for name, (values, declared) in ordinal.items():
                data[name] = pd.Categorical(values, categories=declared, ordered=True)
    if with_nominal:
        data["metastasis"] = ["yes", "no", "yes", "yes", "no"]
        categories["metastasis"] = ["no", "yes"]

    frame = backend.make_frame(data, categories=categories)
    expected = make_clinical_kernel_expected(
        with_ordinal=with_ordinal, with_nominal=with_nominal, with_continuous=with_continuous
    )
    return frame, expected, kwargs


@pytest.fixture()
def make_data():
    def _make(**kwargs):
        data, expected, _ = make_clinical_kernel_data(PANDAS_BACKEND, **kwargs)
        return data, expected

    return _make


@pytest.fixture()
def make_backend_data(dataframe_backend):
    def _make(**kwargs):
        return make_clinical_kernel_data(dataframe_backend, **kwargs)

    return _make


class TestClinicalKernel:
    @staticmethod
    def test_clinical_kernel_1(make_backend_data):
        data, expected, kwargs = make_backend_data()
        mat = clinical_kernel(data, **kwargs)

        assert_array_almost_equal(expected, mat, 4)

    @staticmethod
    def test_clinical_kernel_no_ordinal(make_backend_data):
        data, expected, kwargs = make_backend_data(with_ordinal=False)
        mat = clinical_kernel(data, **kwargs)
        assert_array_almost_equal(expected, mat, 4)

    @staticmethod
    def test_clinical_kernel_no_nominal(make_backend_data):
        data, expected, kwargs = make_backend_data(with_nominal=False)
        mat = clinical_kernel(data, **kwargs)
        assert_array_almost_equal(expected, mat, 4)

    @staticmethod
    def test_clinical_kernel_no_continuous(make_backend_data):
        data, expected, kwargs = make_backend_data(with_continuous=False)
        mat = clinical_kernel(data, **kwargs)
        assert_array_almost_equal(expected, mat, 4)

    @staticmethod
    def test_clinical_kernel_only_nominal(make_backend_data):
        data, expected, kwargs = make_backend_data(with_continuous=False, with_ordinal=False)
        mat = clinical_kernel(data, **kwargs)
        assert_array_almost_equal(expected, mat, 4)

    @staticmethod
    def test_clinical_kernel_x_and_y(make_backend_data):
        data, m, kwargs = make_backend_data()
        mat = clinical_kernel(data[:3], data[3:], **kwargs)
        expected = m[:3:, 3:]

        assert_array_almost_equal(expected, mat, 4)

    @staticmethod
    def test_fit_column_classification(make_backend_data):
        data, _, kwargs = make_backend_data()
        t = ClinicalKernelTransform(**kwargs)
        t.fit(data)

        assert t.X_fit_.shape == data.shape
        assert list(t._numeric_columns) == [0, 1, 2]
        assert list(t._nominal_columns) == [3]

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
    def test_fit_once_fit_dataframe_raises(make_backend_data):
        data, _, kwargs = make_backend_data()
        t = ClinicalKernelTransform(fit_once=True, **kwargs)
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
    def test_bool_column_treated_as_numeric(dataframe_backend):
        age = [20.0, 23.0, 26.0, 54.0, 100.0]
        event_bool = [True, False, True, True, False]
        df_bool = dataframe_backend.make_frame({"age": age, "event": event_bool})
        df_uint = dataframe_backend.make_frame({"age": age, "event": np.array(event_bool, dtype=np.uint8)})

        assert_array_almost_equal(clinical_kernel(df_bool), clinical_kernel(df_uint))

        t_bool = ClinicalKernelTransform().fit(df_bool)
        t_uint = ClinicalKernelTransform().fit(df_uint)
        assert_array_almost_equal(t_bool._numeric_ranges, t_uint._numeric_ranges)
        assert_array_almost_equal(t_bool.X_fit_, t_uint.X_fit_)
        assert list(t_bool._numeric_columns) == [0, 1]
        assert list(t_bool._nominal_columns) == []

    @staticmethod
    def test_bool_only_columns_no_raise(dataframe_backend):
        values_a = [True, False, True, False]
        values_b = [False, False, True, True]
        df_bool = dataframe_backend.make_frame({"a": values_a, "b": values_b})
        df_uint = dataframe_backend.make_frame(
            {"a": np.array(values_a, dtype=np.uint8), "b": np.array(values_b, dtype=np.uint8)}
        )
        assert_array_almost_equal(clinical_kernel(df_uint), clinical_kernel(df_bool), 6)

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
    def test_feature_mismatch(dataframe_backend):
        x = dataframe_backend.make_frame({"a": [1.0, 2.0], "b": [3.0, 4.0]})

        y_renamed = dataframe_backend.make_frame({"a": [1.0, 2.0], "c": [3.0, 4.0]})
        with pytest.raises(ValueError, match="columns do not match"):
            clinical_kernel(x, y_renamed)

        y_fewer = dataframe_backend.make_frame({"a": [1.0, 2.0]})
        with pytest.raises(ValueError, match="different number of features"):
            clinical_kernel(x, y_fewer)

        y_array = np.zeros((10, 17))
        with pytest.raises(ValueError, match="different number of features"):
            clinical_kernel(x, y_array)


class TestClinicalKernelLazyFrame:
    """polars-specific: LazyFrame inputs must be rejected."""

    @staticmethod
    def test_clinical_kernel_lazyframe_rejected():
        data, _, kwargs = make_clinical_kernel_data(POLARS_BACKEND)
        with pytest.raises(TypeError, match=r"polars\.LazyFrame is not supported"):
            clinical_kernel(data.lazy(), **kwargs)

    @staticmethod
    def test_transform_lazyframe_rejected():
        """``ClinicalKernelTransform`` must reject a polars LazyFrame."""
        df = pl.DataFrame(
            {
                "age": [40.0, 50.0, 60.0, 70.0],
                "grade": pl.Series(["I", "II", "III", "I"], dtype=pl.Enum(["I", "II", "III", "IV"])),
            }
        )
        with pytest.raises(TypeError, match=r"polars\.LazyFrame is not supported"):
            ClinicalKernelTransform().fit(df.lazy())
        with pytest.raises(TypeError, match=r"polars\.LazyFrame is not supported"):
            ClinicalKernelTransform(fit_once=True).prepare(df.lazy())


class TestClinicalKernelTransformCrossLibrary:
    """fit/transform behavior across dataframe libraries."""

    @staticmethod
    def test_fit_transform_missing_numeric_matches_pandas():
        pd_x = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": [1.0, 1.0, 2.0, 2.0]})
        pl_x = pl.DataFrame({"a": [1.0, 2.0, None, 4.0], "b": [1.0, 1.0, 2.0, 2.0]})

        pd_t = ClinicalKernelTransform().fit(pd_x)
        pl_t = ClinicalKernelTransform().fit(pl_x)

        # The range of "a" is computed from the non-missing values only.
        assert_array_almost_equal(pd_t._numeric_ranges, [3.0, 1.0])
        assert_array_almost_equal(pl_t._numeric_ranges, pd_t._numeric_ranges)

        assert_array_almost_equal(pl_t.transform(pl_x), pd_t.transform(pd_x))

    @staticmethod
    @pytest.mark.parametrize("fit_backend,transform_backend", CROSS_LIBRARY_PAIRS)
    def test_transform_library_mismatch_raises(fit_backend, transform_backend):
        data = {"age": [20.0, 23.0, 26.0], "grade": ["low", "mid", "high"]}
        categories = {"grade": ["low", "mid", "high"]}
        t = ClinicalKernelTransform().fit(fit_backend.make_frame(data, categories=categories))
        with pytest.raises(TypeError, match="same dataframe library"):
            t.transform(transform_backend.make_frame(data, categories=categories))


class TestOrdinalColumnsOptIn:
    @staticmethod
    def _make_pl_df():
        return pl.DataFrame(
            {
                "age": pl.Series("age", [40.0, 50.0, 60.0, 70.0, 80.0], dtype=pl.Float64),
                "stage": pl.Series("stage", ["T1", "T2", "T3", "T1", "T2"], dtype=pl.Enum(["T1", "T2", "T3", "T4"])),
            }
        )

    @staticmethod
    def _make_pd_ordered():
        return pd.DataFrame(
            {
                "age": [40.0, 50.0, 60.0, 70.0, 80.0],
                "stage": pd.Categorical(
                    ["T1", "T2", "T3", "T1", "T2"],
                    categories=["T1", "T2", "T3", "T4"],
                    ordered=True,
                ),
            }
        )

    def test_default_is_nominal(self):
        df_pl = self._make_pl_df()
        df_pd_ordered = self._make_pd_ordered()
        K_pl = clinical_kernel(df_pl)
        K_pd_ord = clinical_kernel(df_pd_ordered)
        assert not np.allclose(K_pl, K_pd_ord)

    def test_opt_in_matches_pandas_ordered(self):
        df_pl = self._make_pl_df()
        df_pd_ordered = self._make_pd_ordered()
        K_pl = clinical_kernel(df_pl, ordinal_categories={"stage": ["T1", "T2", "T3", "T4"]})
        K_pd_ord = clinical_kernel(df_pd_ordered)
        np.testing.assert_allclose(K_pl, K_pd_ord, atol=1e-12)

    def test_opt_in_transform(self):
        df_pl = self._make_pl_df()
        t_default = ClinicalKernelTransform()
        t_opt_in = ClinicalKernelTransform(ordinal_categories={"stage": ["T1", "T2", "T3", "T4"]})
        t_default.fit(df_pl)
        t_opt_in.fit(df_pl)
        assert list(t_default._nominal_columns) == [1]
        assert list(t_opt_in._nominal_columns) == []
        assert list(t_opt_in._numeric_columns) == [0, 1]

    def test_unknown_column_raises(self):
        df_pl = self._make_pl_df()
        with pytest.raises(ValueError, match="unknown column names"):
            clinical_kernel(df_pl, ordinal_categories={"does_not_exist": ["a", "b"]})

    def test_categorical_column_can_be_declared_ordinal(self):
        # Under the explicit ``ordinal_categories`` API the user supplies the
        # order, so an unordered polars Categorical column can be declared
        # ordinal and is then treated differently from the nominal default.
        df_pl = pl.DataFrame(
            {
                "age": pl.Series("age", [40.0, 50.0, 60.0], dtype=pl.Float64),
                "label": pl.Series("label", ["x", "y", "z"], dtype=pl.Categorical),
            }
        )
        K_default = clinical_kernel(df_pl)
        K_ordinal = clinical_kernel(df_pl, ordinal_categories={"label": ["x", "y", "z"]})
        assert K_ordinal.shape == (3, 3)
        assert not np.allclose(K_default, K_ordinal)

    def test_non_mapping_raises(self):
        df_pl = self._make_pl_df()
        with pytest.raises(TypeError, match="must be a mapping"):
            clinical_kernel(df_pl, ordinal_categories=42)

    def test_non_string_key_raises(self):
        df_pl = self._make_pl_df()
        with pytest.raises(TypeError, match="keys must be strings"):
            clinical_kernel(df_pl, ordinal_categories={1: ["a", "b"]})


class TestNominalNullParity:
    """Missing nominal values must not match themselves."""

    @staticmethod
    def _frames_with_null():
        df_pd = pd.DataFrame(
            {
                "age": [40.0, 50.0, 60.0],
                "grade": pd.Categorical(["I", None, "II"], categories=["I", "II", "III"]),
            }
        )
        df_pl = pl.DataFrame(
            {
                "age": [40.0, 50.0, 60.0],
                "grade": pl.Series(["I", None, "II"], dtype=pl.Enum(["I", "II", "III"])),
            }
        )
        return df_pd, df_pl

    @staticmethod
    def test_one_hot_encoder_null_parity():
        df_pd, df_pl = TestNominalNullParity._frames_with_null()
        out_pd = OneHotEncoder().fit_transform(df_pd).to_numpy()
        out_pl = OneHotEncoder().fit_transform(df_pl).to_numpy()
        np.testing.assert_array_equal(out_pd, out_pl, strict=True)

    @staticmethod
    def test_clinical_kernel_null_parity():
        df_pd, df_pl = TestNominalNullParity._frames_with_null()
        K_pd = clinical_kernel(df_pd)
        K_pl = clinical_kernel(df_pl)
        np.testing.assert_allclose(K_pd, K_pl, atol=1e-12, strict=True)
        assert K_pl[1, 1] < 1.0

    @staticmethod
    def test_clinical_kernel_transform_null_parity():
        df_pd, df_pl = TestNominalNullParity._frames_with_null()
        K_pd = ClinicalKernelTransform().fit(df_pd)(df_pd, df_pd)
        K_pl = ClinicalKernelTransform().fit(df_pl)(df_pl, df_pl)
        np.testing.assert_allclose(K_pd, K_pl, atol=1e-12, strict=True)

    @staticmethod
    def test_pairwise_kernel_null_parity():
        df_pd, df_pl = TestNominalNullParity._frames_with_null()
        t_pd = ClinicalKernelTransform().fit(df_pd)
        t_pl = ClinicalKernelTransform().fit(df_pl)
        np.testing.assert_array_equal(np.isnan(t_pd.X_fit_), np.isnan(t_pl.X_fit_))
        assert np.isnan(t_pl.X_fit_[1, 1])

    @staticmethod
    def test_object_null_parity():
        df_pd = pd.DataFrame({"age": [40.0, 50.0, 60.0], "grade": ["I", None, "II"]})
        df_pl = pl.DataFrame({"age": [40.0, 50.0, 60.0], "grade": ["I", None, "II"]})
        K_pd = clinical_kernel(df_pd)
        K_pl = clinical_kernel(df_pl)
        np.testing.assert_allclose(K_pd, K_pl, atol=1e-12)
        assert K_pd[1, 1] < 1.0


class TestClinicalKernelTransformReplay:
    """Raw transform input must replay fit-time categorical semantics."""

    @staticmethod
    def test_transform_matches_clinical_kernel(dataframe_backend):
        data, _, kwargs = make_clinical_kernel_data(dataframe_backend)
        K_transform = ClinicalKernelTransform(**kwargs).fit(data).transform(data)
        K_direct = clinical_kernel(data, data, **kwargs)
        np.testing.assert_allclose(K_transform, K_direct, atol=1e-12)

    @staticmethod
    def test_ordinal_missing_and_unknown_transform_matches_clinical_kernel():
        df_pd = pd.DataFrame(
            {
                "stage": pd.Categorical(
                    ["T1", None, "T2"],
                    categories=["T1", "T2", "T3"],
                    ordered=True,
                )
            }
        )
        df_pl = pl.DataFrame({"stage": pl.Series(["T1", None, "T4", "T2"], dtype=pl.String)})
        ordinal_categories = {"stage": ["T1", "T2", "T3"]}

        K_pd_transform = ClinicalKernelTransform().fit(df_pd).transform(df_pd)
        K_pd_direct = clinical_kernel(df_pd, df_pd)
        np.testing.assert_allclose(K_pd_transform, K_pd_direct, atol=1e-12, strict=True)

        K_pl_transform = ClinicalKernelTransform(ordinal_categories=ordinal_categories).fit(df_pl).transform(df_pl)
        K_pl_direct = clinical_kernel(df_pl, df_pl, ordinal_categories=ordinal_categories)
        np.testing.assert_allclose(K_pl_transform, K_pl_direct, atol=1e-12, strict=True)

    @staticmethod
    def test_all_categorical_polars_transform():
        df_pl = pl.DataFrame({"stage": pl.Series(["T1", "T2", "T1"], dtype=pl.Enum(["T1", "T2", "T3"]))})
        K = ClinicalKernelTransform().fit(df_pl).transform(df_pl)
        assert K.shape == (3, 3)

    @staticmethod
    def test_polars_transform_subset_rows():
        df_pl = pl.DataFrame(
            {
                "age": [40.0, 50.0, 60.0],
                "stage": pl.Series("stage", ["T1", "T2", "T1"], dtype=pl.Enum(["T1", "T2", "T3"])),
                "label": pl.Series("label", ["x", "y", "x"], dtype=pl.Categorical),
            }
        )
        t = ClinicalKernelTransform(ordinal_categories={"stage": ["T1", "T2", "T3"]}).fit(df_pl)
        K_sub = t.transform(df_pl.head(2))
        assert K_sub.shape == (2, 3)

    @staticmethod
    def test_polars_transform_recasts_numeric_column_from_string():
        fit = pl.DataFrame({"score": [1.0, 2.0, 3.0]})
        transform = pl.DataFrame({"score": ["1.0", "2.0"]})
        t = ClinicalKernelTransform().fit(fit)
        K = t.transform(transform)
        assert K.shape == (2, 3)
        np.testing.assert_allclose(K, clinical_kernel(fit, fit.head(2)).T, strict=True)

    @staticmethod
    def test_polars_transform_all_numeric_no_nominal_columns():
        df = pl.DataFrame({"age": [40.0, 50.0, 60.0], "score": [1.0, 3.0, 5.0]})
        t = ClinicalKernelTransform().fit(df)
        K = t.transform(df.head(2))
        assert K.shape == (2, 3)
        np.testing.assert_allclose(K, clinical_kernel(df, df.head(2)).T, strict=True)


class TestClinicalKernelEdgeCases:
    @staticmethod
    def test_clinical_kernel_multiple_nominal_columns():
        df = pl.DataFrame(
            {
                "grade": pl.Series(["I", "II", "I"], dtype=pl.Categorical),
                "site": pl.Series(["A", "A", "B"], dtype=pl.Categorical),
            }
        )
        mat = clinical_kernel(df)
        expected = np.array(
            [
                [1.0, 0.5, 0.5],
                [0.5, 1.0, 0.0],
                [0.5, 0.0, 1.0],
            ]
        )
        np.testing.assert_allclose(mat, expected)

    @staticmethod
    def test_empty_polars_frame_fit_matches_pandas():
        df_pl = pl.DataFrame({"num": pl.Series([], dtype=pl.Float64)})
        df_pd = pd.DataFrame({"num": pd.Series([], dtype=np.float64)})
        t_pl = ClinicalKernelTransform().fit(df_pl)
        t_pd = ClinicalKernelTransform().fit(df_pd)
        np.testing.assert_array_equal(t_pl._numeric_ranges, t_pd._numeric_ranges, strict=True)
        assert t_pl.X_fit_.shape == t_pd.X_fit_.shape == (0, 1)

    @staticmethod
    @pytest.mark.parametrize("x_backend,y_backend", CROSS_LIBRARY_PAIRS)
    def test_mixed_backend_inputs_raise_typeerror(x_backend, y_backend):
        data = {"num": [1.0, 2.0], "cat": ["A", "B"]}
        categories = {"cat": ["A", "B"]}

        with pytest.raises(TypeError, match="must use the same dataframe library"):
            clinical_kernel(
                x_backend.make_frame(data, categories=categories), y_backend.make_frame(data, categories=categories)
            )

    @staticmethod
    def test_invalid_ordinal_categories_raise():
        df = pl.DataFrame(
            {
                "num": [1.0, 2.0],
                "grade": pl.Series(["A", "B"], dtype=pl.Enum(["A", "B"])),
            }
        )

        with pytest.raises(TypeError, match="must be a mapping"):
            clinical_kernel(df, ordinal_categories=1)
        with pytest.raises(TypeError, match="keys must be strings"):
            clinical_kernel(df, ordinal_categories={1: ["a"]})
        with pytest.raises(ValueError, match="unknown column names"):
            clinical_kernel(df, ordinal_categories={"unknown": ["a"]})
        with pytest.raises(ValueError, match="requires a categorical, string, or object column"):
            clinical_kernel(df, ordinal_categories={"num": ["1", "2"]})
        with pytest.raises(TypeError, match="must be an iterable of category labels"):
            clinical_kernel(df, ordinal_categories={"grade": 5})
        with pytest.raises(ValueError, match="must list at least one category"):
            clinical_kernel(df, ordinal_categories={"grade": []})
        with pytest.raises(ValueError, match="has duplicate categories"):
            clinical_kernel(df, ordinal_categories={"grade": ["A", "A"]})

    @staticmethod
    def test_clinical_kernel_all_missing_ordinal_column_range_zero(recwarn):
        df = pl.DataFrame(
            {
                "num": [1.0, 2.0, 3.0],
                "stage": pl.Series("stage", [None, None, None], dtype=pl.Utf8),
            }
        )
        # _ordinal_range maps an all-missing ordinal column to 0.0 instead of
        # calling nanmax-nanmin on an all-NaN array, which would warn.
        assert _ordinal_range(np.full(3, np.nan)) == 0.0

        # fit() exercises the _ordinal_range path; the all-NaN guard keeps the
        # warning from leaking.
        ClinicalKernelTransform(ordinal_categories={"stage": ["T1", "T2", "T3"]}).fit(df)
        assert not any("All-NaN" in str(w.message) for w in recwarn.list)

        # The functional path also handles an all-missing ordinal column end-to-end.
        mat = clinical_kernel(df, ordinal_categories={"stage": ["T1", "T2", "T3"]})
        assert mat.shape == (3, 3)
        assert np.all(np.isnan(mat))

    @staticmethod
    def test_unsupported_polars_dtype_raises():
        df = pl.DataFrame({"items": [[1], [2]]})
        with pytest.raises(TypeError, match="unsupported dtype"):
            clinical_kernel(df)

    @staticmethod
    def test_pairwise_kernel_polars_fit_with_nominal_column():
        df = pl.DataFrame(
            {
                "age": [40.0, 50.0],
                "grade": pl.Series(["I", "II"], dtype=pl.Categorical),
            }
        )
        transform = ClinicalKernelTransform().fit(df)
        value = transform.pairwise_kernel(transform.X_fit_[0], transform.X_fit_[1])
        expected = clinical_kernel(df)[0, 1]
        np.testing.assert_allclose(value, expected, atol=1e-12, strict=True)
