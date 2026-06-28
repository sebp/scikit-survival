"""Tests for ``sksurv.kernels.clinical`` with polars input."""

from dataframe_test_utils import make_clinical_kernel_pandas_data
import numpy as np
from numpy.testing import assert_array_almost_equal
import pandas as pd
import polars as pl
import pytest

from sksurv.kernels import ClinicalKernelTransform, clinical_kernel
from sksurv.kernels._clinical_dataframe import _ordinal_range
from sksurv.preprocessing import OneHotEncoder


def _clinical_polars_from_pandas(data):
    cols = {}
    if "age" in data:
        cols["age"] = pl.Series("age", data["age"].to_numpy(), dtype=pl.Float64)
    if "lymph node size" in data:
        categories = [str(cat) for cat in data["lymph node size"].cat.categories]
        values = [str(value) for value in data["lymph node size"].to_numpy()]
        cols["lymph node size"] = pl.Series("lymph node size", values, dtype=pl.Enum(categories))
    if "lymph node spread" in data:
        categories = list(data["lymph node spread"].cat.categories)
        values = data["lymph node spread"].astype(str).to_numpy()
        cols["lymph node spread"] = pl.Series("lymph node spread", values, dtype=pl.Enum(categories))
    if "metastasis" in data:
        cols["metastasis"] = pl.Series("metastasis", data["metastasis"].astype(str).to_numpy(), dtype=pl.Categorical)
    return pl.DataFrame(cols)


@pytest.fixture()
def make_polars_clinical_data():
    def _make(with_ordinal=True, with_nominal=True, with_continuous=True):
        data, expected = make_clinical_kernel_pandas_data(
            with_ordinal=with_ordinal, with_nominal=with_nominal, with_continuous=with_continuous
        )
        return _clinical_polars_from_pandas(data), expected

    return _make


ORDINAL_CATS_POLARS = {
    "lymph node size": ["1", "2", "3", "4"],
    "lymph node spread": ["none", "close", "distant"],
}


class TestClinicalKernelPolars:
    @staticmethod
    def test_clinical_kernel_full(make_polars_clinical_data):
        data, expected = make_polars_clinical_data()
        mat = clinical_kernel(data, ordinal_categories=ORDINAL_CATS_POLARS)
        assert_array_almost_equal(expected, mat, 4)

    @staticmethod
    def test_clinical_kernel_no_ordinal(make_polars_clinical_data):
        data, expected = make_polars_clinical_data(with_ordinal=False)
        mat = clinical_kernel(data)
        assert_array_almost_equal(expected, mat, 4)

    @staticmethod
    def test_clinical_kernel_no_nominal(make_polars_clinical_data):
        data, expected = make_polars_clinical_data(with_nominal=False)
        mat = clinical_kernel(data, ordinal_categories=ORDINAL_CATS_POLARS)
        assert_array_almost_equal(expected, mat, 4)

    @staticmethod
    def test_clinical_kernel_no_continuous(make_polars_clinical_data):
        data, expected = make_polars_clinical_data(with_continuous=False)
        mat = clinical_kernel(data, ordinal_categories=ORDINAL_CATS_POLARS)
        assert_array_almost_equal(expected, mat, 4)

    @staticmethod
    def test_clinical_kernel_lazyframe_rejected(make_polars_clinical_data):
        data, _ = make_polars_clinical_data()
        with pytest.raises(TypeError, match=r"polars\.LazyFrame is not supported"):
            clinical_kernel(data.lazy(), ordinal_categories=ORDINAL_CATS_POLARS)

    @staticmethod
    def test_clinical_kernel_x_and_y(make_polars_clinical_data):
        data, m = make_polars_clinical_data()
        mat = clinical_kernel(data.slice(0, 3), data.slice(3, 2), ordinal_categories=ORDINAL_CATS_POLARS)
        expected = m[:3, 3:]
        assert_array_almost_equal(expected, mat, 4)

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
    def test_clinical_kernel_boolean_treated_as_numeric():
        age = [20, 23, 26, 54, 100]
        event_bool = [True, False, True, True, False]
        event_int = [1, 0, 1, 1, 0]
        df_bool = pl.DataFrame({"age": pl.Series("age", age, dtype=pl.Float64), "event": event_bool})
        df_int = pl.DataFrame(
            {
                "age": pl.Series("age", age, dtype=pl.Float64),
                "event": pl.Series("event", event_int, dtype=pl.UInt8),
            }
        )

        mat_bool = clinical_kernel(df_bool)
        mat_int = clinical_kernel(df_int)
        assert_array_almost_equal(mat_int, mat_bool, 6)

    @staticmethod
    def test_clinical_kernel_boolean_only_no_raise():
        df_bool = pl.DataFrame(
            {
                "a": [True, False, True, False],
                "b": [False, False, True, True],
            }
        )
        df_int = pl.DataFrame(
            {
                "a": pl.Series("a", [1, 0, 1, 0], dtype=pl.UInt8),
                "b": pl.Series("b", [0, 0, 1, 1], dtype=pl.UInt8),
            }
        )
        mat_bool = clinical_kernel(df_bool)
        mat_int = clinical_kernel(df_int)
        assert_array_almost_equal(mat_int, mat_bool, 6)


class TestClinicalKernelTransformPolars:
    @staticmethod
    def test_fit_polars(make_polars_clinical_data):
        data, _ = make_polars_clinical_data()
        t = ClinicalKernelTransform(ordinal_categories=ORDINAL_CATS_POLARS)
        t.fit(data)
        assert t.X_fit_.shape == data.shape
        assert list(t._numeric_columns) == [0, 1, 2]
        assert list(t._nominal_columns) == [3]

    @staticmethod
    def test_fit_polars_boolean_is_numeric():
        df = pl.DataFrame(
            {
                "age": pl.Series("age", [20, 23, 26, 54, 100], dtype=pl.Float64),
                "event": [True, False, True, True, False],
            }
        )
        t = ClinicalKernelTransform()
        t.fit(df)
        assert list(t._numeric_columns) == [0, 1]
        assert list(t._nominal_columns) == []

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
    def test_fit_once_fit_polars_dataframe_raises(make_polars_clinical_data):
        data, _ = make_polars_clinical_data()
        t = ClinicalKernelTransform(fit_once=True, ordinal_categories=ORDINAL_CATS_POLARS)
        t.prepare(data)

        with pytest.raises(TypeError, match="fit_once=True expects a numeric array in fit"):
            t.fit(data)

    @staticmethod
    def test_fit_pandas_transform_polars_raises():
        df_pd = pd.DataFrame({"age": [20.0, 23.0, 26.0], "grade": pd.Categorical(["low", "mid", "high"])})
        df_pl = pl.DataFrame(
            {
                "age": [20.0, 23.0, 26.0],
                "grade": pl.Series(["low", "mid", "high"], dtype=pl.Categorical),
            }
        )
        t = ClinicalKernelTransform().fit(df_pd)
        with pytest.raises(TypeError, match="same dataframe library"):
            t.transform(df_pl)

    @staticmethod
    def test_fit_polars_transform_pandas_raises():
        df_pd = pd.DataFrame({"age": [20.0, 23.0, 26.0], "grade": pd.Categorical(["low", "mid", "high"])})
        df_pl = pl.DataFrame(
            {
                "age": [20.0, 23.0, 26.0],
                "grade": pl.Series(["low", "mid", "high"], dtype=pl.Categorical),
            }
        )
        t = ClinicalKernelTransform().fit(df_pl)
        with pytest.raises(TypeError, match="same dataframe library"):
            t.transform(df_pd)

    @staticmethod
    def test_lazyframe_rejected():
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
    def _frame_pair():
        df_pd = pd.DataFrame(
            {
                "age": [40.0, 50.0, 60.0],
                "stage": pd.Categorical(["T1", "T2", "T1"], categories=["T1", "T2", "T3"], ordered=True),
                "label": pd.Categorical(["x", "y", "x"]),
            }
        )
        df_pl = pl.DataFrame(
            {
                "age": [40.0, 50.0, 60.0],
                "stage": pl.Series("stage", ["T1", "T2", "T1"], dtype=pl.Enum(["T1", "T2", "T3"])),
                "label": pl.Series("label", ["x", "y", "x"], dtype=pl.Categorical),
            }
        )
        return df_pd, df_pl

    @staticmethod
    def test_polars_transform_raw_input_does_not_raise():
        _, df_pl = TestClinicalKernelTransformReplay._frame_pair()
        t = ClinicalKernelTransform(ordinal_categories={"stage": ["T1", "T2", "T3"]}).fit(df_pl)
        K = t.transform(df_pl)
        assert K.shape == (3, 3)

    @staticmethod
    def test_polars_transform_matches_clinical_kernel():
        _, df_pl = TestClinicalKernelTransformReplay._frame_pair()
        ordinal_categories = {"stage": ["T1", "T2", "T3"]}
        K_transform = ClinicalKernelTransform(ordinal_categories=ordinal_categories).fit(df_pl).transform(df_pl)
        K_direct = clinical_kernel(df_pl, df_pl, ordinal_categories=ordinal_categories)
        np.testing.assert_allclose(K_transform, K_direct, atol=1e-12)

    @staticmethod
    def test_pandas_transform_matches_clinical_kernel():
        df_pd, _ = TestClinicalKernelTransformReplay._frame_pair()
        K_transform = ClinicalKernelTransform().fit(df_pd).transform(df_pd)
        K_direct = clinical_kernel(df_pd, df_pd)
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
        _, df_pl = TestClinicalKernelTransformReplay._frame_pair()
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
    def test_empty_polars_frame_fit_matches_pandas():
        df_pl = pl.DataFrame({"num": pl.Series([], dtype=pl.Float64)})
        df_pd = pd.DataFrame({"num": pd.Series([], dtype=np.float64)})
        t_pl = ClinicalKernelTransform().fit(df_pl)
        t_pd = ClinicalKernelTransform().fit(df_pd)
        np.testing.assert_array_equal(t_pl._numeric_ranges, t_pd._numeric_ranges, strict=True)
        assert t_pl.X_fit_.shape == t_pd.X_fit_.shape == (0, 1)

    @staticmethod
    def test_mixed_backend_inputs_raise_typeerror():
        x_pd = pd.DataFrame({"num": [1.0, 2.0], "cat": pd.Categorical(["A", "B"])})
        x_pl = pl.DataFrame({"num": [1.0, 2.0], "cat": pl.Series(["A", "B"], dtype=pl.Categorical)})

        with pytest.raises(TypeError, match="must use the same dataframe library"):
            clinical_kernel(x_pd, x_pl)
        with pytest.raises(TypeError, match="must use the same dataframe library"):
            clinical_kernel(x_pl, x_pd)

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
    def test_polars_column_mismatch_raises():
        x = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        y = pl.DataFrame({"a": [1.0, 2.0], "c": [3.0, 4.0]})
        with pytest.raises(ValueError, match="columns do not match"):
            clinical_kernel(x, y)

    @staticmethod
    def test_polars_feature_count_mismatch_raises():
        x = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        y = pl.DataFrame({"a": [1.0, 2.0]})
        with pytest.raises(ValueError, match="different number of features"):
            clinical_kernel(x, y)

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
