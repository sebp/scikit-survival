"""Internal dataframe helpers for Narwhals-backed input handling."""

from . import _polars as polars_inputs
from ._categorical_encoding import (
    column_to_category_codes,
    column_to_one_hot_matrix,
    expand_dataframe_with_one_hot_columns,
    get_one_hot_column_names,
)
from ._categorical_semantics import ColumnSemantics, get_semantic_categories, infer_column_semantics
from ._input import (
    SUPPORTED_DATAFRAME_INPUT_TYPES,
    ensure_eager_dataframe,
    get_dataframe_library,
    is_narwhals_dataframe,
    is_narwhals_dataframe_or_series,
    is_supported_dataframe,
    is_supported_dataframe_or_series,
    to_narwhals_dataframe,
    unsupported_dataframe_error,
)

__all__ = [
    "ColumnSemantics",
    "SUPPORTED_DATAFRAME_INPUT_TYPES",
    "to_narwhals_dataframe",
    "column_to_category_codes",
    "column_to_one_hot_matrix",
    "expand_dataframe_with_one_hot_columns",
    "ensure_eager_dataframe",
    "get_dataframe_library",
    "infer_column_semantics",
    "is_supported_dataframe",
    "is_supported_dataframe_or_series",
    "get_one_hot_column_names",
    "get_semantic_categories",
    "polars_inputs",
    "unsupported_dataframe_error",
    "is_narwhals_dataframe_or_series",
    "is_narwhals_dataframe",
]
