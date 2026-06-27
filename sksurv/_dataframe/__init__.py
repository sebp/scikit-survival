# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Internal dataframe helpers for Narwhals-backed input handling."""

from . import _polars as polars_inputs
from ._categorical_encoding import (
    column_to_category_codes,
    column_to_one_hot_matrix,
    expand_dataframe_with_one_hot_columns,
    get_one_hot_column_names,
)
from ._categorical_semantics import (
    ColumnSemantics,
    get_semantic_categories,
    infer_column_semantics,
    is_categorical_or_string_dtype,
)
from ._input import (
    SUPPORTED_DATAFRAME_INPUT_TYPES,
    ensure_eager_dataframe,
    get_dataframe_library,
    is_supported_dataframe,
    is_supported_dataframe_or_series,
    unsupported_dataframe_error,
)

__all__ = [
    "ColumnSemantics",
    "SUPPORTED_DATAFRAME_INPUT_TYPES",
    "column_to_category_codes",
    "column_to_one_hot_matrix",
    "expand_dataframe_with_one_hot_columns",
    "ensure_eager_dataframe",
    "get_dataframe_library",
    "infer_column_semantics",
    "is_categorical_or_string_dtype",
    "is_supported_dataframe",
    "is_supported_dataframe_or_series",
    "get_one_hot_column_names",
    "get_semantic_categories",
    "polars_inputs",
    "unsupported_dataframe_error",
]
