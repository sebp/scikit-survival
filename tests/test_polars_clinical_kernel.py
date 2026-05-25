"""Tests for ``sksurv.kernels.clinical`` with polars input."""

import numpy as np
from numpy.testing import assert_array_almost_equal
import pandas as pd
import polars as pl
import pytest

from sksurv.kernels import ClinicalKernelTransform, clinical_kernel
from sksurv.preprocessing import OneHotEncoder


def _expected_clinical_kernel_matrix(with_ordinal=True, with_nominal=True, with_continuous=True):
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


@pytest.fixture()
def make_polars_clinical_data():
    age = [20, 23, 26, 54, 100]
    node_size = [2, 1, 3, 4, 1]
    node_spread = ["distant", "none", "distant", "close", "none"]
    metastasis = ["yes", "no", "yes", "yes", "no"]

    def _make(with_ordinal=True, with_nominal=True, with_continuous=True):
        cols = {}
        if with_continuous:
            cols["age"] = pl.Series("age", age, dtype=pl.Float64)
        if with_ordinal:
            cols["lymph node size"] = pl.Series(
                "lymph node size", [str(v) for v in node_size], dtype=pl.Enum(["1", "2", "3", "4"])
            )
            cols["lymph node spread"] = pl.Series(
                "lymph node spread", node_spread, dtype=pl.Enum(["none", "close", "distant"])
            )
        if with_nominal:
            cols["metastasis"] = pl.Series("metastasis", metastasis, dtype=pl.Categorical)
        df = pl.DataFrame(cols)
        expected = _expected_clinical_kernel_matrix(
            with_ordinal=with_ordinal, with_nominal=with_nominal, with_continuous=with_continuous
        )
        return df, expected

    return _make


ORDINAL_COLS_POLARS = ["lymph node size", "lymph node spread"]


class TestClinicalKernelPolars:
    @staticmethod
    def test_clinical_kernel_full(make_polars_clinical_data):
        data, expected = make_polars_clinical_data()
        mat = clinical_kernel(data, ordinal_columns=ORDINAL_COLS_POLARS)
        assert_array_almost_equal(expected, mat, 4)

    @staticmethod
    def test_clinical_kernel_no_ordinal(make_polars_clinical_data):
        data, expected = make_polars_clinical_data(with_ordinal=False)
        mat = clinical_kernel(data)
        assert_array_almost_equal(expected, mat, 4)

    @staticmethod
    def test_clinical_kernel_no_nominal(make_polars_clinical_data):
        data, expected = make_polars_clinical_data(with_nominal=False)
        mat = clinical_kernel(data, ordinal_columns=ORDINAL_COLS_POLARS)
        assert_array_almost_equal(expected, mat, 4)

    @staticmethod
    def test_clinical_kernel_no_continuous(make_polars_clinical_data):
        data, expected = make_polars_clinical_data(with_continuous=False)
        mat = clinical_kernel(data, ordinal_columns=ORDINAL_COLS_POLARS)
        assert_array_almost_equal(expected, mat, 4)

    @staticmethod
    def test_clinical_kernel_lazy(make_polars_clinical_data):
        data, expected = make_polars_clinical_data()
        mat = clinical_kernel(data.lazy(), ordinal_columns=ORDINAL_COLS_POLARS)
        assert_array_almost_equal(expected, mat, 4)

    @staticmethod
    def test_clinical_kernel_lazy_matches_eager():
        df = pl.DataFrame(
            {
                "age": [40.0, 50.0, 60.0, 70.0],
                "grade": pl.Series(["I", "II", "III", "I"], dtype=pl.Enum(["I", "II", "III", "IV"])),
            }
        )
        K_eager = clinical_kernel(df)
        K_lazy = clinical_kernel(df.lazy())
        np.testing.assert_allclose(K_eager, K_lazy)

    @staticmethod
    def test_clinical_kernel_x_and_y(make_polars_clinical_data):
        data, m = make_polars_clinical_data()
        mat = clinical_kernel(data.slice(0, 3), data.slice(3, 2), ordinal_columns=ORDINAL_COLS_POLARS)
        expected = m[:3, 3:]
        assert_array_almost_equal(expected, mat, 4)

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
        t = ClinicalKernelTransform(ordinal_columns=ORDINAL_COLS_POLARS)
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
    def test_fit_once_fit_polars_dataframe_raises(make_polars_clinical_data):
        data, _ = make_polars_clinical_data()
        t = ClinicalKernelTransform(fit_once=True, ordinal_columns=ORDINAL_COLS_POLARS)
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
    def test_call_lazyframe_matches_eager():
        """Regression: ``ClinicalKernelTransform.__call__(X, Y)`` with a
        LazyFrame Y must collect before accessing row count.
        """
        import warnings

        df = pl.DataFrame(
            {
                "age": [40.0, 50.0, 60.0, 70.0],
                "grade": pl.Series(["I", "II", "III", "I"], dtype=pl.Enum(["I", "II", "III", "IV"])),
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            K_eager = ClinicalKernelTransform().fit(df)(df, df)
            K_lazy = ClinicalKernelTransform().fit(df.lazy())(df.lazy(), df.lazy())
        np.testing.assert_allclose(K_eager, K_lazy)


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
        K_pl = clinical_kernel(df_pl, ordinal_columns=["stage"])
        K_pd_ord = clinical_kernel(df_pd_ordered)
        np.testing.assert_allclose(K_pl, K_pd_ord, atol=1e-12)

    def test_opt_in_transform(self):
        df_pl = self._make_pl_df()
        t_default = ClinicalKernelTransform()
        t_opt_in = ClinicalKernelTransform(ordinal_columns=["stage"])
        t_default.fit(df_pl)
        t_opt_in.fit(df_pl)
        assert list(t_default._nominal_columns) == [1]
        assert list(t_opt_in._nominal_columns) == []
        assert list(t_opt_in._numeric_columns) == [0, 1]

    def test_unknown_column_raises(self):
        df_pl = self._make_pl_df()
        with pytest.raises(ValueError, match="unknown column names"):
            clinical_kernel(df_pl, ordinal_columns=["does_not_exist"])

    def test_non_enum_column_raises(self):
        df_pl = pl.DataFrame(
            {
                "age": pl.Series("age", [40.0, 50.0, 60.0], dtype=pl.Float64),
                "label": pl.Series("label", ["x", "y", "z"], dtype=pl.Categorical),
            }
        )
        with pytest.raises(ValueError, match="requires a categorical dtype with declared category order"):
            clinical_kernel(df_pl, ordinal_columns=["label"])

    def test_non_iterable_raises(self):
        df_pl = self._make_pl_df()
        with pytest.raises(TypeError, match="must be an iterable"):
            clinical_kernel(df_pl, ordinal_columns=42)

    def test_non_string_entry_raises(self):
        df_pl = self._make_pl_df()
        with pytest.raises(TypeError, match="entries must be strings"):
            clinical_kernel(df_pl, ordinal_columns=[1])


class TestNominalNullParity:
    """Missing nominal values must not match themselves."""

    @staticmethod
    def _frames_with_null():
        import pandas as pd

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
        np.testing.assert_array_equal(out_pd, out_pl)

    @staticmethod
    def test_clinical_kernel_null_parity():
        df_pd, df_pl = TestNominalNullParity._frames_with_null()
        K_pd = clinical_kernel(df_pd)
        K_pl = clinical_kernel(df_pl)
        np.testing.assert_allclose(K_pd, K_pl, atol=1e-12)
        assert K_pl[1, 1] < 1.0

    @staticmethod
    def test_clinical_kernel_transform_null_parity():
        df_pd, df_pl = TestNominalNullParity._frames_with_null()
        K_pd = ClinicalKernelTransform().fit(df_pd)(df_pd, df_pd)
        K_pl = ClinicalKernelTransform().fit(df_pl)(df_pl, df_pl)
        np.testing.assert_allclose(K_pd, K_pl, atol=1e-12)

    @staticmethod
    def test_pairwise_kernel_null_parity():
        df_pd, df_pl = TestNominalNullParity._frames_with_null()
        t_pd = ClinicalKernelTransform().fit(df_pd)
        t_pl = ClinicalKernelTransform().fit(df_pl)
        np.testing.assert_array_equal(np.isnan(t_pd.X_fit_), np.isnan(t_pl.X_fit_))
        assert np.isnan(t_pl.X_fit_[1, 1])

    @staticmethod
    def test_object_null_parity():
        import pandas as pd

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
        import pandas as pd

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
        t = ClinicalKernelTransform(ordinal_columns=["stage"]).fit(df_pl)
        K = t.transform(df_pl)
        assert K.shape == (3, 3)

    @staticmethod
    def test_polars_transform_matches_clinical_kernel():
        _, df_pl = TestClinicalKernelTransformReplay._frame_pair()
        K_transform = ClinicalKernelTransform(ordinal_columns=["stage"]).fit(df_pl).transform(df_pl)
        K_direct = clinical_kernel(df_pl, df_pl, ordinal_columns=["stage"])
        np.testing.assert_allclose(K_transform, K_direct, atol=1e-12)

    @staticmethod
    def test_pandas_transform_matches_clinical_kernel():
        df_pd, _ = TestClinicalKernelTransformReplay._frame_pair()
        K_transform = ClinicalKernelTransform().fit(df_pd).transform(df_pd)
        K_direct = clinical_kernel(df_pd, df_pd)
        np.testing.assert_allclose(K_transform, K_direct, atol=1e-12)

    @staticmethod
    def test_all_categorical_polars_transform():
        df_pl = pl.DataFrame({"stage": pl.Series(["T1", "T2", "T1"], dtype=pl.Enum(["T1", "T2", "T3"]))})
        K = ClinicalKernelTransform().fit(df_pl).transform(df_pl)
        assert K.shape == (3, 3)

    @staticmethod
    def test_polars_transform_subset_rows():
        _, df_pl = TestClinicalKernelTransformReplay._frame_pair()
        t = ClinicalKernelTransform(ordinal_columns=["stage"]).fit(df_pl)
        K_sub = t.transform(df_pl.head(2))
        assert K_sub.shape == (2, 3)


class TestClinicalKernelEdgeCases:
    @staticmethod
    def test_empty_polars_frame_fit_matches_pandas():
        import pandas as pd

        df_pl = pl.DataFrame({"num": pl.Series([], dtype=pl.Float64)})
        df_pd = pd.DataFrame({"num": pd.Series([], dtype=np.float64)})
        t_pl = ClinicalKernelTransform().fit(df_pl)
        t_pd = ClinicalKernelTransform().fit(df_pd)
        np.testing.assert_array_equal(t_pl._numeric_ranges, t_pd._numeric_ranges)
        assert t_pl.X_fit_.shape == t_pd.X_fit_.shape == (0, 1)

    @staticmethod
    def test_mixed_backend_inputs_raise_typeerror():
        import pandas as pd

        x_pd = pd.DataFrame({"num": [1.0, 2.0], "cat": pd.Categorical(["A", "B"])})
        x_pl = pl.DataFrame({"num": [1.0, 2.0], "cat": pl.Series(["A", "B"], dtype=pl.Categorical)})

        with pytest.raises(TypeError, match="must use the same dataframe library"):
            clinical_kernel(x_pd, x_pl)
        with pytest.raises(TypeError, match="must use the same dataframe library"):
            clinical_kernel(x_pl, x_pd)
