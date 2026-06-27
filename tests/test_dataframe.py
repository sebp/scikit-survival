"""Tests for the internal dataframe boundary helpers."""

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest

from sksurv._dataframe import (
    ColumnSemantics,
    column_to_category_codes,
    column_to_one_hot_matrix,
    ensure_eager_dataframe,
    expand_dataframe_with_one_hot_columns,
    get_dataframe_library,
    get_semantic_categories,
    infer_column_semantics,
    is_supported_dataframe,
    is_supported_dataframe_or_series,
    polars_inputs,
    unsupported_dataframe_error,
)


def test_polars_inputs_predicates_separate_frames_and_series():
    frame = pl.DataFrame({"x": [1, 2]})
    lazy = frame.lazy()
    series = frame["x"]

    assert polars_inputs.LIBRARY.is_dataframe(frame)
    assert not polars_inputs.LIBRARY.is_dataframe(lazy)
    assert not polars_inputs.LIBRARY.is_dataframe(series)

    assert polars_inputs.LIBRARY.is_series(series)
    assert not polars_inputs.LIBRARY.is_series(frame)


def test_external_dataframe_library_lookup():
    frame = pl.DataFrame({"x": [1, 2]})
    series = frame["x"]

    frame_library = get_dataframe_library(frame)
    assert frame_library.name == "polars"
    assert frame_library.dataframe_display_name == "polars.DataFrame"
    assert get_dataframe_library(series) is None

    series_library = get_dataframe_library(series, allow_series=True)
    assert series_library.name == "polars"
    assert series_library.series_display_name == "polars.Series"
    assert get_dataframe_library(frame, allow_series=True) is frame_library


def test_ensure_eager_dataframe_rejects_lazyframe_and_passes_others_through():
    frame = pl.DataFrame({"x": [1, 2]})
    series = frame["x"]

    with pytest.raises(TypeError, match=r"polars\.LazyFrame is not supported"):
        ensure_eager_dataframe(frame.lazy())

    assert ensure_eager_dataframe(frame) is frame
    assert ensure_eager_dataframe(series) is series
    arr = np.arange(3)
    assert ensure_eager_dataframe(arr) is arr


def test_supported_input_predicates_are_explicit_to_sksurv():
    frame = pl.DataFrame({"x": [1, 2]})
    series = frame["x"]
    pd_frame = pd.DataFrame({"x": [1, 2]})
    pd_series = pd_frame["x"]

    assert is_supported_dataframe(frame)
    assert not is_supported_dataframe(frame.lazy())
    assert is_supported_dataframe(pd_frame)
    assert not is_supported_dataframe(series)
    assert is_supported_dataframe_or_series(frame)
    assert is_supported_dataframe_or_series(series)
    assert is_supported_dataframe_or_series(pd_frame)
    assert is_supported_dataframe_or_series(pd_series)
    assert not is_supported_dataframe_or_series([[1], [2]])


def test_unsupported_dataframe_error_names_supported_inputs():
    err = unsupported_dataframe_error([[1], [2]])
    assert isinstance(err, TypeError)
    assert "pandas.DataFrame" in str(err)
    assert "polars.DataFrame" in str(err)


def test_column_to_category_codes_numeric_semantics():
    semantics = ColumnSemantics(name="x", kind="numeric", categories=None, ordered=False)
    codes = column_to_category_codes(pl.Series("x", [1, 2, 3]), semantics)
    np.testing.assert_array_equal(codes, np.array([1.0, 2.0, 3.0]))


def test_column_to_one_hot_matrix_handles_unknown_and_empty_categories():
    semantics = ColumnSemantics(name="grade", kind="nominal", categories=("A", "B"), ordered=False)
    encoded = column_to_one_hot_matrix(pl.Series("grade", ["A", "C", None]), semantics, drop_first=False)
    np.testing.assert_array_equal(encoded[0], np.array([1.0, 0.0]))
    np.testing.assert_array_equal(np.isnan(encoded[1:]), np.ones((2, 2), dtype=bool))

    empty_semantics = ColumnSemantics(name="empty", kind="nominal", categories=(), ordered=False)
    empty = column_to_one_hot_matrix(pl.Series("empty", ["x", None]), empty_semantics)
    assert empty.shape == (2, 0)


def test_get_one_hot_column_names_without_dropping_first_category():
    from sksurv._dataframe import get_one_hot_column_names

    semantics = ColumnSemantics(name="grade", kind="nominal", categories=("A", "B"), ordered=False)
    assert get_one_hot_column_names(semantics, drop_first=False) == ["grade=A", "grade=B"]


def test_expand_dataframe_with_one_hot_columns_empty_frame_policy():
    nw_frame = nw.from_native(pl.DataFrame())
    result = expand_dataframe_with_one_hot_columns(
        nw_frame,
        columns_to_encode={},
        allow_drop=True,
        implementation=nw_frame.implementation,
        on_empty="empty_frame",
    )
    assert result.shape == (0, 0)


def test_expand_dataframe_with_one_hot_columns_empty_policy_errors():
    nw_frame = nw.from_native(pl.DataFrame())

    with pytest.raises(ValueError, match="No objects to concatenate"):
        expand_dataframe_with_one_hot_columns(
            nw_frame,
            columns_to_encode={},
            allow_drop=True,
            implementation=nw_frame.implementation,
            on_empty="raise",
        )

    with pytest.raises(ValueError, match="on_empty must be"):
        expand_dataframe_with_one_hot_columns(
            nw_frame,
            columns_to_encode={},
            allow_drop=True,
            implementation=nw_frame.implementation,
            on_empty="bad",
        )


def test_expand_dataframe_with_one_hot_columns_preserves_single_category_when_requested():
    df = pl.DataFrame({"grade": pl.Series(["A", "A"], dtype=pl.Enum(["A"]))})
    nw_frame = nw.from_native(df)
    col = nw_frame.get_column("grade")
    semantics = infer_column_semantics(col)

    result = expand_dataframe_with_one_hot_columns(
        nw_frame,
        columns_to_encode={"grade": (col, semantics)},
        allow_drop=False,
        implementation=nw_frame.implementation,
    )

    assert result.to_dict(as_series=False) == {"grade": ["A", "A"]}


def test_semantic_category_helpers_reject_unsupported_dtype():
    series = pl.Series("value", [1.0, 2.0])
    with pytest.raises(TypeError, match="get_semantic_categories"):
        get_semantic_categories(series)

    list_series = pl.Series("items", [[1], [2]])
    with pytest.raises(TypeError, match="unsupported dtype"):
        infer_column_semantics(list_series)


def test_infer_column_semantics_numeric_and_boolean():
    numeric = infer_column_semantics(pl.Series("x", [1.0, 2.0, 3.0]))
    assert numeric == ColumnSemantics(name="x", kind="numeric", categories=None, ordered=False)

    boolean = infer_column_semantics(pl.Series("b", [True, False, True]))
    assert boolean.name == "b"
    assert boolean.kind == "numeric"
    assert boolean.categories is None
    assert boolean.ordered is False
