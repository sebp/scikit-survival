"""Column semantics for Narwhals-backed dataframe inputs.

The non-obvious policy is that ``nw.Enum`` is nominal, not ordinal. Its
declared category order is used for ARFF/category fidelity only; callers
that need statistical ordinality must opt in explicitly, for example via
``ordinal_columns=``.
"""

from dataclasses import dataclass
from typing import Literal

import narwhals.stable.v2 as nw

__all__ = [
    "ColumnSemantics",
    "infer_column_semantics",
    "get_semantic_categories",
]


def get_semantic_categories(series):
    """Return deterministic categories for Narwhals-backed categorical columns.

    Explicit category orders are preserved for ``Enum``. For inferred
    categoricals and strings, observed non-null values are sorted to avoid
    backend-specific discovery order and polars' global categorical string pool.
    """
    s = nw.from_native(series, series_only=True)
    dt = s.dtype
    if isinstance(dt, nw.Enum):
        return tuple(s.cat.get_categories().to_list())
    if isinstance(dt, (nw.Categorical, nw.String)):
        return tuple(sorted(s.drop_nulls().unique().to_list()))
    raise TypeError(f"get_semantic_categories: unsupported dtype {dt!r}; expected Enum, Categorical, or String")


@dataclass(frozen=True)
class ColumnSemantics:
    """Dataframe-library-neutral description of a single column's role in sksurv."""

    name: str
    kind: Literal["numeric", "nominal", "ordinal"]
    categories: tuple | None
    ordered: bool


def infer_column_semantics(column):
    """Infer :class:`ColumnSemantics` from a single column."""
    s = nw.from_native(column, series_only=True)
    dt = s.dtype

    if dt.is_numeric() or isinstance(dt, nw.Boolean):
        return ColumnSemantics(name=s.name, kind="numeric", categories=None, ordered=False)

    if isinstance(dt, (nw.Enum, nw.Categorical, nw.String)):
        return ColumnSemantics(
            name=s.name,
            kind="nominal",
            categories=get_semantic_categories(s),
            ordered=False,
        )

    raise TypeError(f"unsupported dtype for sksurv semantics: {dt!r}")
