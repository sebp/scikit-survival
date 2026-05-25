"""Tests for the internal dataframe boundary helpers."""

import pandas as pd
import polars as pl

from sksurv._dataframe import (
    collect_lazy_dataframe,
    get_dataframe_library,
    is_narwhals_dataframe,
    is_narwhals_dataframe_or_series,
    is_supported_dataframe,
    is_supported_dataframe_or_series,
    polars_inputs,
    to_narwhals_dataframe,
)


def test_polars_inputs_predicates_separate_frames_and_series():
    frame = pl.DataFrame({"x": [1, 2]})
    lazy = frame.lazy()
    series = frame["x"]

    assert polars_inputs.is_dataframe(frame)
    assert polars_inputs.is_dataframe(lazy)
    assert not polars_inputs.is_dataframe(series)

    assert polars_inputs.is_series(series)
    assert not polars_inputs.is_series(frame)


def test_external_dataframe_library_lookup():
    frame = pl.DataFrame({"x": [1, 2]})
    series = frame["x"]

    frame_library = get_dataframe_library(frame)
    assert frame_library.name == "polars"
    assert frame_library.dataframe_display_name == "polars.DataFrame or polars.LazyFrame"
    assert get_dataframe_library(series) is None

    series_library = get_dataframe_library(series, allow_series=True)
    assert series_library.name == "polars"
    assert series_library.series_display_name == "polars.Series"
    assert get_dataframe_library(frame, allow_series=True) is frame_library


def test_narwhals_predicates_route_backend_inputs():
    frame = pl.DataFrame({"x": [1, 2]})

    assert is_narwhals_dataframe(frame)
    assert is_narwhals_dataframe(frame.lazy())
    assert not is_narwhals_dataframe(frame["x"])
    assert not is_narwhals_dataframe([[1], [2]])

    assert is_narwhals_dataframe_or_series(frame)
    assert is_narwhals_dataframe_or_series(frame.lazy())
    assert is_narwhals_dataframe_or_series(frame["x"])
    assert not is_narwhals_dataframe_or_series([[1], [2]])


def test_collect_lazy_dataframe_collects_lazyframe_and_passes_others_through():
    frame = pl.DataFrame({"x": [1, 2]})
    lazy = frame.lazy()
    series = frame["x"]

    eager = collect_lazy_dataframe(lazy)
    assert isinstance(eager, pl.DataFrame)
    assert eager.to_dict(as_series=False) == frame.to_dict(as_series=False)
    assert collect_lazy_dataframe(frame) is frame
    assert collect_lazy_dataframe(series) is series


def test_to_narwhals_dataframe_returns_eager_dataframe_wrapper():
    frame = pl.DataFrame({"x": [1, 2]})

    nw_frame = to_narwhals_dataframe(frame.lazy())
    assert nw_frame.columns == ["x"]
    assert nw_frame.shape == (2, 1)
    assert isinstance(nw_frame.to_native(), pl.DataFrame)


def test_supported_input_predicates_are_explicit_to_sksurv():
    frame = pl.DataFrame({"x": [1, 2]})
    series = frame["x"]
    pd_frame = pd.DataFrame({"x": [1, 2]})
    pd_series = pd_frame["x"]

    assert is_supported_dataframe(frame)
    assert is_supported_dataframe(frame.lazy())
    assert is_supported_dataframe(pd_frame)
    assert not is_supported_dataframe(series)
    assert is_supported_dataframe_or_series(frame)
    assert is_supported_dataframe_or_series(series)
    assert is_supported_dataframe_or_series(pd_frame)
    assert is_supported_dataframe_or_series(pd_series)
    assert not is_supported_dataframe_or_series([[1], [2]])
